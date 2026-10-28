import contextlib
import io
import json
import unittest
from unittest import mock

from bot.abi import word
from venues import (PoolRef, QUOTE_V3_SELECTOR, QUOTE_V4_SELECTOR,
                    QUOTE_V4_SELECTOR_FLAT, QUOTE_V4_SELECTOR_STRUCT,
                    VenueError, all_venues, encode_pool_id, encode_v4_quote,
                    get_venue, sort_pair)
from venues.chains import CHAINS, get_chain
from venues.pancake_v3 import pancake_v3
from venues.uniswap_v3 import uniswap_v3
from venues.uniswap_v4 import uniswap_v4

USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
WETH = "0x4200000000000000000000000000000000000006"
MUSD_MONAD = "0xacA92E438df0B2401fF60dA7E4337B687a2435DA"
USDC_MONAD = "0x754704Bc059F8C67012fEd69BC8A327a5aafb603"
ZERO_HOOKS = "0x0000000000000000000000000000000000000000"


class FakeRpc:
    """Returns canned eth_call payloads; records what was asked."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def eth_call(self, to, data, block, sender=None):
        self.calls.append((to, data, block))
        if not self.responses:
            raise AssertionError(f"unexpected eth_call to {to}")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def call(self, method, params):
        if method == "eth_gasPrice":
            return hex(6_000_000)  # 0.006 gwei
        if method == "eth_blockNumber":
            return hex(1000)
        raise AssertionError(f"unexpected rpc method {method}")


class SelectorTests(unittest.TestCase):
    def test_v4_selector_matches_the_deployed_bytecode(self):
        # Monad and Base quoters both contain this selector; the sqrtPriceLimit
        # variants are listed so a future probe can try them.
        self.assertEqual(QUOTE_V4_SELECTOR, "aa9d21cb")
        self.assertEqual(QUOTE_V4_SELECTOR_STRUCT, "97d73bdf")
        self.assertEqual(QUOTE_V4_SELECTOR_FLAT, "627bfac4")

    def test_v3_selector_is_the_quoter_v2_entry_point(self):
        self.assertEqual(QUOTE_V3_SELECTOR, "c6a5026a")

    def test_pool_id_matches_a_pool_recovered_from_an_initialize_event(self):
        # MUSD/USDC on Monad, recovered from the PoolManager Initialize log.
        key = (USDC_MONAD.lower(), MUSD_MONAD.lower(), 0, 1, ZERO_HOOKS)
        self.assertEqual(encode_pool_id(key).hex(),
                         "c4105dd13e1027d2db1a8b80bed2276378f1c57cc0cb69f9937cdffbea39635d")

    def test_pool_id_is_order_independent_because_callers_sort_first(self):
        a = encode_pool_id((*sort_pair(USDC, WETH), 500, 10, ZERO_HOOKS))
        b = encode_pool_id((*sort_pair(WETH, USDC), 500, 10, ZERO_HOOKS))
        self.assertEqual(a, b)

    def test_v4_calldata_carries_the_struct_offset(self):
        key = (USDC.lower(), WETH.lower(), 500, 10, ZERO_HOOKS)
        data = encode_v4_quote(key, True, 1_000)
        body = data[10:]
        self.assertEqual(int(body[:64], 16), 32, "head must offset to the struct")
        # word 0 is the struct offset, words 1-5 are PoolKey, 6 zeroForOne,
        # 7 amountIn, 8 the bytes offset, 9 the bytes length.
        inner_offset = int(body[8 * 64:9 * 64], 16)
        self.assertEqual(inner_offset, 256, "bytes offset is relative to the struct")
        self.assertEqual(len(body), (9 + 1) * 64, "8 static words, offset, length")
        # Without the leading offset the call reverts with empty data on mainnet.
        self.assertNotEqual(data[10:74], word(USDC.lower())[:64])

    def test_zero_for_one_flag_follows_the_token_order(self):
        key = (USDC.lower(), WETH.lower(), 500, 10, ZERO_HOOKS)
        forward = encode_v4_quote(key, True, 1)
        backward = encode_v4_quote(key, False, 1)
        self.assertNotEqual(forward, backward)
        self.assertEqual(int(forward[10 + 6 * 64:10 + 7 * 64], 16), 1)


class ChainTests(unittest.TestCase):
    def test_known_symbols_resolve(self):
        base = get_chain("base")
        self.assertEqual(base.chain_id, 8453)
        self.assertEqual(base.token("USDC")[1], 6)
        self.assertEqual(base.token("usdc")[0], USDC)

    def test_addresses_pass_through(self):
        base = get_chain("base")
        self.assertEqual(base.token(WETH), (WETH, 18))

    def test_unknown_symbol_and_chain_fail_closed(self):
        with self.assertRaises(KeyError):
            get_chain("base").token("NOPE")
        with self.assertRaises(KeyError):
            get_chain("nowhere")

    def test_monad_is_registered_with_its_own_tokens(self):
        monad = CHAINS["monad"]
        self.assertEqual(monad.chain_id, 143)
        self.assertEqual(monad.token("MUSD")[0], MUSD_MONAD)


class RegistryTests(unittest.TestCase):
    def test_three_venues_are_registered(self):
        ids = [venue.id for venue in all_venues()]
        self.assertEqual(ids, ["pancake-v3", "uniswap-v3", "uniswap-v4"])

    def test_unknown_venue_raises(self):
        with self.assertRaises(KeyError):
            get_venue("sushiswap")

    def test_chain_support_is_declared_not_assumed(self):
        self.assertEqual(uniswap_v3.supports, ("base", "monad"))
        self.assertEqual(uniswap_v4.supports, ("base", "monad"))
        self.assertEqual(pancake_v3.supports, ("base",))


class RegistryDriftTests(unittest.TestCase):
    """The legacy pipeline keeps its own Exchange table; it must not drift."""

    def test_bot_exchange_table_matches_the_venue_deployments(self):
        from bot.networks import EXCHANGES
        sources = {"UniswapV3": uniswap_v3, "PancakeV3": pancake_v3}
        self.assertEqual(len(EXCHANGES), 2)
        for exchange in EXCHANGES:
            deployment = sources[exchange.label].deployments["base"]
            with self.subTest(exchange=exchange.label):
                self.assertEqual(exchange.factory, deployment.factory)
                self.assertEqual(exchange.quoter, deployment.quoter)
                self.assertEqual(exchange.fees, deployment.fees)


class V3AdapterTests(unittest.TestCase):
    def test_discovery_skips_zero_addresses(self):
        rpc = FakeRpc([word(0), word(WETH), word(0), word(0)])
        pools = uniswap_v3.discover(rpc, "base", USDC, WETH, "latest")
        self.assertEqual([p.address for p in pools], [WETH])

    def test_quote_uses_the_fee_tier_and_returns_the_first_word(self):
        rpc = FakeRpc([word(12345)])
        pool = PoolRef(venue="uniswap-v3", chain="base", fee=500,
                       token0=USDC.lower(), token1=WETH.lower(), address=WETH)
        self.assertEqual(uniswap_v3.quote(rpc, "base", pool, USDC, 1_000, "latest"), 12345)
        data = rpc.calls[0][1]
        self.assertTrue(data.startswith("0x" + QUOTE_V3_SELECTOR))
        self.assertEqual(int(data[10 + 3 * 64:10 + 4 * 64], 16), 500)

    def test_wrong_token_is_rejected(self):
        pool = PoolRef(venue="uniswap-v3", chain="base", fee=500,
                       token0=USDC.lower(), token1=WETH.lower(), address=WETH)
        with self.assertRaises(VenueError):
            uniswap_v3.quote(FakeRpc([]), "base", pool, MUSD_MONAD, 1_000, "latest")

    def test_unsupported_chain_fails_closed(self):
        with self.assertRaises(VenueError):
            pancake_v3.discover(FakeRpc([]), "monad", USDC, WETH, "latest")


class V4AdapterTests(unittest.TestCase):
    def _pool(self):
        key = (*sort_pair(USDC, WETH), 500, 10, ZERO_HOOKS)
        return PoolRef(venue="uniswap-v4", chain="base", fee=500,
                       token0=key[0], token1=key[1], address=None,
                       extra=(10, ZERO_HOOKS, "0x" + encode_pool_id(key).hex()))

    def test_discovery_requires_price_and_liquidity(self):
        # fee 0 initialised, fee 100 initialised but empty, the rest uninitialised.
        responses = [word(1), word(1)] + [word(1), word(0)] + [word(0)] * 33
        pools = uniswap_v4.discover(FakeRpc(responses), "base", USDC, WETH, "latest")
        self.assertEqual([p.fee for p in pools], [0])

    def test_quote_returns_the_amount_out_word(self):
        rpc = FakeRpc([word(777)])
        self.assertEqual(uniswap_v4.quote(rpc, "base", self._pool(), USDC, 1_000, "latest"), 777)

    def test_quote_falls_back_across_abi_shapes(self):
        # First shape reverts, second succeeds; the working one is remembered.
        rpc = FakeRpc([RuntimeError("execution reverted"), word(42)])
        venue_pool = self._pool()
        self.assertEqual(uniswap_v4.quote(rpc, "base", venue_pool, USDC, 1_000, "latest"), 42)
        self.assertEqual(uniswap_v4._selector.get("base"), "97d73bdf")

    def test_all_shapes_reverting_is_a_venue_error(self):
        rpc = FakeRpc([RuntimeError("execution reverted")] * 4)
        with self.assertRaises(VenueError):
            uniswap_v4.quote(rpc, "base", self._pool(), USDC, 1_000, "latest")


class CliTests(unittest.TestCase):
    def test_subcommands_exist(self):
        from arbcheck.__main__ import build_parser
        parser = build_parser()
        for command in ("venues", "quote", "sweep", "replay"):
            with self.subTest(command=command):
                args = parser.parse_args({"venues": ["venues"],
                                          "quote": ["quote", "--pair", "USDC/WETH"],
                                          "sweep": ["sweep", "--pair", "USDC/WETH"],
                                          "replay": ["replay"]}[command])
                self.assertTrue(callable(args.func))

    def test_pair_parsing_rejects_malformed_input(self):
        from arbcheck.__main__ import _pair
        with self.assertRaises(SystemExit):
            _pair(get_chain("base"), "USDCWETH")
        (token_in, token_out) = _pair(get_chain("base"), "USDC/WETH")
        self.assertEqual(token_in[1], 6)
        self.assertEqual(token_out[1], 18)

    def test_sweep_json_is_one_stable_document_for_stubbed_quotes(self):
        from arbcheck import __main__ as cli

        class SweepRpc:
            def call(self, method, params):
                self.asserted = (method, params)
                return hex(1000)

        def quote(args):
            self.assertTrue(args.json)
            payload = {
                "quotes": [{"venue": "uniswap-v3", "pool": "pool-a", "price": args.size}],
                "best": {"buy": "uniswap-v3", "sell": "uniswap-v4",
                         "gross_usd": 1.25, "net_usd": None},
            }
            print("diagnostic text that must stay off the JSON stream")
            print(json.dumps(payload))
            return 0

        args = cli.build_parser().parse_args([
            "sweep", "--pair", "USDC/WETH", "--sizes", "1,2", "--blocks", "1", "--json",
        ])
        output = io.StringIO()
        with mock.patch.object(cli, "_rpc", return_value=SweepRpc()), \
                mock.patch.object(cli, "cmd_quote", side_effect=quote), \
                contextlib.redirect_stdout(output):
            self.assertEqual(cli.cmd_sweep(args), 0)

        result = json.loads(output.getvalue())
        self.assertEqual(result["chain"], "base")
        self.assertEqual(result["pair"], "USDC/WETH")
        self.assertEqual(result["sizes"], [1.0, 2.0])
        self.assertEqual(len(result["blocks"]), 1)
        self.assertEqual(result["blocks"][0]["block"], "0x3e8")
        self.assertEqual([quote["size"] for quote in result["blocks"][0]["quotes"]], [1.0, 2.0])
        best_by_size = result["blocks"][0]["best"]["by_size"]
        self.assertEqual(set(best_by_size), {"1.0", "2.0"})
        self.assertEqual(best_by_size["1.0"]["net_usd_reason"],
                         "gas could not be priced")
        self.assertIsNone(result["blocks"][0]["gas_usd"])
        self.assertEqual(result["blocks"][0]["gas_usd_reason"], "gas could not be priced")
        self.assertEqual(result["summary"], {
            "samples": 2, "successful_samples": 2, "failed_samples": 0,
        })


if __name__ == "__main__":
    unittest.main()

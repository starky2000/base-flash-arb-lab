"""arbcheck — verify whether an on-chain arbitrage route pays before you deploy it.

Read-only. No keys, no signing, no broadcasts. Every number comes from a venue's
own on-chain quoter at a pinned block, so pool fee and price impact are inside the
quote rather than estimated.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys

from bot.rpc import RPC, RpcError
from venues import VenueError, all_venues, get_venue
from venues.chains import CHAINS, get_chain

DEFAULT_GAS_UNITS = 450_000


def _rpc(args) -> RPC:
    chain = get_chain(args.chain)
    return RPC(args.rpc or chain.rpc_default)


def _pair(chain, pair: str) -> tuple[tuple[str, int], tuple[str, int]]:
    if "/" not in pair:
        raise SystemExit(f"--pair expects BASE/QUOTE, got {pair!r}")
    base, quote = pair.split("/", 1)
    return chain.token(base.strip()), chain.token(quote.strip())


def _selected_venues(args, chain_key: str):
    wanted = [v.strip() for v in args.venues.split(",")] if args.venues else None
    venues = all_venues() if not wanted else [get_venue(v) for v in wanted]
    out = []
    for venue in venues:
        if chain_key in venue.supports:
            out.append(venue)
        elif wanted:
            raise SystemExit(f"{venue.id} is not available on {chain_key}")
    if not out:
        raise SystemExit(f"no venue available on {chain_key}")
    return out


def _quote_venues(rpc, chain, venues, token_in, token_out, amount_in, block):
    """One pass over every pool of every venue, in the token_in -> token_out direction."""
    quotes = []
    for venue in venues:
        try:
            pools = venue.discover(rpc, chain.key, token_in[0], token_out[0], block)
        except (VenueError, RpcError) as exc:
            quotes.append({"venue": venue.id, "error": str(exc)})
            continue
        for pool in pools:
            try:
                amount_out = venue.quote(rpc, chain.key, pool, token_in[0], amount_in, block)
            except (VenueError, RpcError) as exc:
                quotes.append({"venue": venue.id, "pool": pool.label(), "error": str(exc)})
                continue
            quotes.append({
                "venue": venue.id,
                "pool": pool.label(),
                "ref": pool,
                "amount_out": amount_out,
                "price": amount_out / 10 ** token_out[1],
            })
    return quotes


def _gas_usd(chain, rpc, block, gas_units, eth_usd):
    if eth_usd is None:
        # Derive the ETH price from a real quote rather than a hardcoded number.
        try:
            weth = chain.token("WETH")
            usdc = chain.token("USDC")
            best = None
            for venue in all_venues():
                if chain.key not in venue.supports:
                    continue
                for pool in venue.discover(rpc, chain.key, weth[0], usdc[0], block):
                    out = venue.quote(rpc, chain.key, pool, weth[0], 10 ** weth[1], block)
                    if out > 0 and (best is None or out > best):
                        best = out
            if best:
                eth_usd = best / 10 ** usdc[1]
        except (VenueError, RpcError, KeyError):
            pass
    if eth_usd is None:
        return None
    gas_price = int(rpc.call("eth_gasPrice", []), 16)
    return gas_price * gas_units / 1e18 * eth_usd


def cmd_venues(args):
    chain = get_chain(args.chain)
    print(f"chain  {chain.key} ({chain.chain_id})  tokens: {', '.join(sorted(chain.tokens))}")
    for venue in all_venues():
        mark = "yes" if chain.key in venue.supports else "no"
        print(f"  {venue.id:14} {venue.label:16} on {chain.key}: {mark}")
    print()
    print("chains:", ", ".join(f"{c.key}({c.chain_id})" for c in CHAINS.values()))
    return 0


def cmd_quote(args):
    chain = get_chain(args.chain)
    token_in, token_out = _pair(chain, args.pair)
    venues = _selected_venues(args, chain.key)
    rpc = _rpc(args)
    block = args.block or "latest"
    amount_in = int(round(args.size * 10 ** token_in[1]))

    raw_quotes = _quote_venues(rpc, chain, venues, token_in, token_out, amount_in, block)
    errors = [q for q in raw_quotes if "amount_out" not in q]
    quotes = [q for q in raw_quotes if "amount_out" in q]
    if not quotes:
        print("no pool returned a quote; nothing can be concluded from this block")
        for err in errors:
            print(f"  {err['venue']}: {err['error']}")
        return 1

    # A pool can hold non-zero liquidity and still be nearly empty at the current
    # tick, which quotes as an absurd price. Anything more than 5% away from the
    # median is a depth artifact, not an opportunity, so it is excluded rather than
    # ranked.
    prices = sorted(q["price"] for q in quotes)
    median = prices[len(prices) // 2]
    plausible = [q for q in quotes if median and abs(q["price"] / median - 1) <= 0.05]
    dropped = len(quotes) - len(plausible)
    if not plausible:
        print("every quote was more than 5% from the median; the pools are too thin ")
        return 1

    best_out = max(q["amount_out"] for q in plausible)
    print(f"{args.pair} on {chain.label} @ block {block} | {args.size:g} in")
    print(f"  {'venue':16}{'pool':22}{'out':>16}{'vs best':>10}")
    for q in sorted(plausible, key=lambda q: -q["amount_out"]):
        delta = (q["amount_out"] / best_out - 1) * 10_000
        print(f"  {q['venue']:16}{q['pool']:22}{q['price']:>16.8f}{delta:>9.1f}bp")
    if dropped:
        print(f"  excluded {dropped} quote(s) more than 5% from the median (depth artifact)")

    # A cycle needs the exact intermediate amount, so the sell leg is re-quoted
    # against the buy output rather than against the same dollar size.
    cycles = []
    for buy in plausible:
        for sell in plausible:
            if sell["ref"] is buy["ref"]:
                continue
            try:
                back = get_venue(sell["venue"]).quote(
                    rpc, chain.key, sell["ref"], token_out[0], buy["amount_out"], block)
            except (VenueError, RpcError):
                continue
            gross = (back - amount_in) / 10 ** token_in[1]
            cycles.append({"buy": buy, "sell": sell, "back": back,
                           "gross_usd": gross, "gross_bps": gross / args.size * 10_000})
    if not cycles:
        print("\nno cross-venue cycle could be quoted between the plausible pools")
        return 1

    cycles.sort(key=lambda c: -c["gross_usd"])
    best = cycles[0]
    gas = _gas_usd(chain, rpc, block, args.gas_units, args.eth_usd)
    net = None if gas is None else best["gross_usd"] - gas

    print(f"\nbest cycle: {best['buy']['pool']} -> {best['sell']['pool']}")
    print(f"  gross   ${best['gross_usd']:+.4f}  ({best['gross_bps']:+.1f} bp)")
    if net is None:
        print("  gas     unknown (no ETH price available) - net not computed")
    else:
        print(f"  gas     ${gas:.4f} ({args.gas_units} gas)")
        print(f"  net     ${net:+.4f}  ({net / args.size * 10_000:+.1f} bp)")
    verdict = "profitable" if (net is not None and net > 0) else "not profitable"
    print(f"  verdict {verdict}")

    if args.json:
        print(json.dumps({
            "quotes": [{k: v for k, v in q.items() if k != "ref"} for q in plausible],
            "best": {"buy": best["buy"]['venue'], "sell": best["sell"]['venue'],
                     "gross_usd": best["gross_usd"], "net_usd": net},
        }, indent=2))
    return 0


def cmd_sweep(args):
    """Quote historical samples.

    With ``--json``, stdout is one stable document: ``chain``, ``pair``, and
    ``sizes`` describe the request; ``blocks`` has one item per sampled block
    with ``quotes`` (tagged by size), ``best.by_size`` (one result per size),
    and ``gas_usd``; ``summary`` reports successful and failed block/size samples.
    A missing gas price is represented by ``gas_usd: null`` and a sibling
    ``gas_usd_reason`` string. No timestamp or human progress text is emitted.
    """
    chain = get_chain(args.chain)
    rpc = _rpc(args)
    head = int(rpc.call("eth_blockNumber", []), 16)
    spacing = args.spacing
    blocks = [hex(head - i * spacing) for i in range(args.blocks)]
    sizes = [float(s) for s in args.sizes.split(",")]
    blocks_by_number = {}
    successes = 0
    failures = 0
    for block in blocks:
        block_result = {"block": block, "quotes": [], "best": {"by_size": {}},
                        "gas_usd": None, "gas_usd_reason": "gas could not be priced"}
        for size in sizes:
            ns = argparse.Namespace(**vars(args))
            ns.block, ns.size = block, size
            ns.json = args.json
            if not args.json:
                print(f"\n=== block {block} · {size:g} {args.pair.split('/')[0]}")
                status = cmd_quote(ns)
                successes += status == 0
                failures += status != 0
                continue

            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                status = cmd_quote(ns)
            output = captured.getvalue()
            report = None
            if status == 0:
                start = output.rfind("\n{")
                if start >= 0:
                    report, _ = json.JSONDecoder().raw_decode(output[start + 1:])
            if report is None:
                failures += 1
                block_result["quotes"].append({"size": size, "error": output.strip() or
                                               "quote did not produce a report"})
                block_result["best"]["by_size"][str(size)] = {"error": "no complete quote"}
                continue

            successes += 1
            block_result["quotes"].extend({"size": size, **quote}
                                          for quote in report.get("quotes", []))
            best = report.get("best", {})
            best_result = {"size": size, **best}
            if best.get("net_usd") is None:
                best_result["net_usd_reason"] = "gas could not be priced"
            block_result["best"]["by_size"][str(size)] = best_result
            net = best.get("net_usd")
            gross = best.get("gross_usd")
            if net is not None and gross is not None:
                block_result["gas_usd"] = gross - net
                block_result["gas_usd_reason"] = None
            elif net is None:
                block_result["gas_usd"] = None
                block_result["gas_usd_reason"] = "gas could not be priced"
        blocks_by_number[block] = block_result

    if args.json:
        print(json.dumps({
            "chain": chain.key,
            "pair": args.pair,
            "sizes": sizes,
            "blocks": list(blocks_by_number.values()),
            "summary": {"samples": len(blocks) * len(sizes),
                        "successful_samples": successes, "failed_samples": failures},
        }, indent=2))
    else:
        print(f"\nswept {successes + failures} combinations; raw quotes above are per block, "
              "not an aggregate claim")
    return 0


def cmd_replay(args):
    cmd = (f"forge script script/ForkProof.s.sol:ForkProof -vvv")
    print("Fork replay is performed by the Solidity fork proof, which never broadcasts.")
    print(f"  candidate block : {args.block or '(pick a quoted block)'}")
    print(f"  command         : {cmd}")
    if not args.execute:
        print("\nDry run. Add --execute to run it (still no broadcast).")
        return 0
    return subprocess.call(cmd.split())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arbcheck", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--chain", default="base", help="chain key (default: base)")
        p.add_argument("--rpc", default=None, help="RPC URL (default: chain public RPC)")
        p.add_argument("--block", default=None, help="pinned block (default: latest)")
        p.add_argument("--venues", default=None, help="comma-separated venue ids")

    v = sub.add_parser("venues", help="list chains and registered venues")
    v.add_argument("--chain", default="base")
    v.set_defaults(func=cmd_venues)

    q = sub.add_parser("quote", help="exact quotes and the best cross-venue cycle")
    common(q)
    q.add_argument("--pair", required=True, help="BASE/QUOTE, e.g. USDC/WETH")
    q.add_argument("--size", type=float, default=5_000, help="input size in BASE units")
    q.add_argument("--gas-units", type=int, default=DEFAULT_GAS_UNITS)
    q.add_argument("--eth-usd", type=float, default=None, help="override the ETH price")
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=cmd_quote)

    s = sub.add_parser("sweep", help="quote several historical blocks and sizes")
    common(s)
    s.add_argument("--pair", required=True)
    s.add_argument("--sizes", default="1000,5000,20000")
    s.add_argument("--blocks", type=int, default=3)
    s.add_argument("--spacing", type=int, default=64, help="blocks between samples")
    s.add_argument("--gas-units", type=int, default=DEFAULT_GAS_UNITS)
    s.add_argument("--eth-usd", type=float, default=None)
    s.add_argument("--json", action="store_true", help="emit one machine-readable JSON document")
    s.set_defaults(func=cmd_sweep)

    r = sub.add_parser("replay", help="print (or run) the no-broadcast fork proof")
    common(r)
    r.add_argument("--pair", default="USDC/WETH")
    r.add_argument("--size", type=float, default=5_000)
    r.add_argument("--execute", action="store_true")
    r.set_defaults(func=cmd_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (KeyError, ValueError, RpcError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

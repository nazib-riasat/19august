"""Print the three fingerprints that must agree across machines.

Run this after bootstrapping on a new laptop or in a Kaggle notebook, and
compare the output with a teammate's::

    python scripts/verify_handoff.py

* **config hash** — the identity of an experimental condition.  Differs only if
  a config *value* differs; comments, whitespace and CRLF cannot move it.
* **log digest** — a hash of the ``(seq, op, payload)`` stream, timestamps
  excluded.  Same digest means the same work produced the same log.
* **manifest fingerprint** — a hash of the manifest's ``reproducibility`` block.
  Differs when the config, the seed, the commit, or a pinned package version
  differs — and *not* when the hostname, OS or GPU differs.
* **lattice fingerprints** (Phase-2 gap G9) — one ``environment_fingerprint`` per
  suite, over the spec, the pool including ``feat``, the obligations, the
  snapshot digest, the gold set, both proof templates **and the enumerated
  graph**.  Two machines must confirm they enumerated the same environment
  *before* comparing any Gate-2 number; binding only the generator's inputs
  would let two machines whose masks or checker differ enumerate different
  graphs and still agree.

  Excludes β on purpose: ``config_hash`` moves when the Phase-3 sweep freezes β,
  so a fingerprint containing it would change after the suites were frozen.  The
  β-dependent layer is ``target_fingerprint``, printed separately.

If any of these disagree between two machines running the same commit, that is a
real bug and worth chasing before it contaminates a result.

The lattice section costs a few seconds (it enumerates thirty lattices), so
``--no-lattice`` skips it when only the config and log digests are wanted.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from graft.canonical import digest_of
from graft.config import config_hash, load_config
from graft.eventlog import EventLog
from graft.graphstore import ReplayGraphStore
from graft.ledger import Ledger
from graft.runtime import REPRO_KEY, run_manifest, set_seed
from graft.schemas import Edge, Node

TS = "2026-08-08T00:00:00+00:00"


def build_reference_log(path: Path, n_nodes: int = 50) -> EventLog:
    """Write a small, fully deterministic log — no clock, no RNG in the content."""
    log = EventLog.open(path)
    for i in range(n_nodes):
        log.append("node.add", Node(node_id=f"n{i}", ntype="Entity", payload={"i": i}).to_dict())
    for i in range(n_nodes - 1):
        log.append(
            "edge.add",
            Edge(
                edge_id=f"e{i}",
                etype="same_as",
                src=f"n{i}",
                dst=f"n{i + 1}",
                t_created=TS,
                provenance=(f"s{i}",),
            ).to_dict(),
        )
    log.append("edge.invalidate", {"edge_id": "e0", "t_invalid": TS, "superseded_by": "e1"})
    return log


def lattice_fingerprints() -> list[tuple[str, str, str]]:
    """``(suite, environment_fingerprint, target_fingerprint)`` per suite.

    Aggregated over each suite's instances in order, so one digest per suite
    stands for the whole thing: a single instance differing anywhere moves it.
    """
    from graft.canonical import digest_of
    from graft.synth.enumerate import environment_fingerprint, reachable_states
    from graft.synth.exact import target_distribution, target_fingerprint
    from graft.synth.lattice import benchmark_suite, probe_suite, tuning_suite

    out: list[tuple[str, str, str]] = []
    for name, build in (
        ("main", benchmark_suite),
        ("probe", probe_suite),
        ("tuning", tuning_suite),
    ):
        environments: list[str] = []
        targets: list[str] = []
        for instance in build():
            graph = reachable_states(instance, instance.cfg)
            env = environment_fingerprint(instance, graph)
            environments.append(env)
            targets.append(
                target_fingerprint(
                    target_distribution(instance, instance.cfg, graph=graph), env
                )
            )
        out.append((name, digest_of(environments), digest_of(targets)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="default", help="config preset to fingerprint")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--no-lattice",
        action="store_true",
        help="skip the Phase-2 suites (they enumerate thirty lattices)",
    )
    args = parser.parse_args()

    cfg = load_config(preset=args.preset)
    set_seed(args.seed)

    with tempfile.TemporaryDirectory() as tmp:
        log = build_reference_log(Path(tmp) / "events.jsonl")

        ledger = Ledger.from_config(cfg, log=log)
        with ledger.query_scope("reference"):
            while not ledger.would_exceed("terminal_checks"):
                ledger.count("terminal_checks")

        snapshot = ReplayGraphStore(log).at()
        log_digest = log.digest()
        graph_digest = snapshot.state_digest()
        log.close()

    manifest = run_manifest(cfg, args.seed)
    repro = manifest[REPRO_KEY]

    print(f"graft            {__import__('graft').__version__}")
    print(f"python           {repro['python']}")
    print(f"commit           {repro['git']['sha']}" + ("  (DIRTY)" if repro["git"]["dirty"] else ""))
    print(f"preset           {args.preset}  seed {args.seed}")
    print()
    print(f"config hash      {config_hash(cfg)}")
    print(f"log digest       {log_digest}")
    print(f"graph digest     {graph_digest}")
    print(f"manifest print   {digest_of(repro)}")
    print()
    print(f"terminal_checks  {ledger.snapshot()['totals']['terminal_checks']} "
          f"(cap {cfg.checker_budget})")
    print(f"graph            {snapshot.counts()}")

    if not args.no_lattice:
        print()
        for suite, environment, target in lattice_fingerprints():
            print(f"lattice {suite:<8} env {environment[:32]}  target(beta={cfg.beta:g}) "
                  f"{target[:32]}")

    if repro["git"]["dirty"]:
        print("\nNote: the working tree is dirty, so the commit alone does not "
              "identify this code. Commit before comparing with a teammate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

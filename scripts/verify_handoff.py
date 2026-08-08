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

If any of the three disagree between two machines running the same commit, that
is a real bug and worth chasing before it contaminates a result.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="default", help="config preset to fingerprint")
    parser.add_argument("--seed", type=int, default=13)
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

    if repro["git"]["dirty"]:
        print("\nNote: the working tree is dirty, so the commit alone does not "
              "identify this code. Commit before comparing with a teammate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

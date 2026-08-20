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

* **ingestion fingerprint** (Phase-5 gap G11) — a hash of Stage A's
  *configuration*: extractor model id, revision and decoding config, the NLI pin,
  the context recipe, the grounding ladder and the prompt-registry SHA.
  **This one binds the config, not the output, and the difference is the point.**
  A bf16 LLM forward pass is not bit-stable across GPU architectures, so the
  "identical log digest on any machine" promise above holds for Phases 0–4 and
  **is explicitly not made for Stage A**.  What two machines must agree on is
  this hash; if they do and their ingestion logs still differ, that is expected
  and is recorded as a limitation rather than chased as a bug.

  Printing it here costs nothing: ``graft.ingest.pins`` imports no ML library,
  which a structural test enforces precisely so this script keeps running on a
  bare interpreter.

* **Stage-B fingerprint** (Phase-6 exit criterion 16) — the same idea for graph
  construction: embedder id and revision, the candidate/pair constants, the
  training budget, the commit floors, and a hash of the **endpoint table**.  The
  last one belongs in configuration identity in the strongest sense: it decides
  *which graphs are constructible*, so two machines agreeing on the embedder and
  disagreeing here would build different graphs from the same log and never
  notice.  Encoder *weights* are not bit-identical across machines and are not
  promised to be; the setup that produced them is.

* **Stage-C fingerprint** (Phase-7 exit criterion 15) — the same idea for
  retrieval: the fusion arithmetic (normalisation, per-channel weights, the
  combining rule and the tie-break), the BM25 constants, the expansion bounds,
  the scorer's declared configuration, and the **shared** embedder pin.  Two
  machines that fuse channels with different arithmetic produce recall numbers
  that are not comparable, and nothing else in the run would say so.
  ``pool_cap`` is deliberately absent: it lives in the config tree and is already
  covered by ``config hash``, and giving a frozen value two homes is the failure
  mode `CLAUDE.md` §5 catalogues.

* **stage-G fingerprint** (Phase-8 exit criterion 12) — the gate's identity: the
  two ablation arms, the model classes and their parameter cap, the feature-block
  names *and their order* (an ablation is a column mask over that order, so a
  reordering silently changes which columns an arm sees), the threshold rule, the
  two prevalences, the class handling, and the abstain-cause vocabulary. Weights
  are not bit-identical across machines and are not promised to be; the setup
  that produced them is, and so is the arithmetic that turns a probability into
  an abstention.

* **stage-D fingerprint** (Phase-9 exit criterion 16) — Stage D's identity: the
  training corpora and the subset decision 2 pinned, the obligation-synthesis
  rules, the featurizer's feature **names in vector order** (the Phase-8
  correction, adopted here from the start rather than after an audit found two
  different experiments sharing one identity), the distillation and portfolio
  constants, the training ladder, and the Gate-3 decision rule itself. ``beta``,
  ``K``, ``checker_budget`` and the seeds stay absent — they are the config
  tree's. ``n_real`` is bound though it is ``None`` until build step 4 derives
  it: a run at 50,000 trajectories and one at 200,000 are different experiments,
  and ``None`` marks a fingerprint taken before the budget was known.

* **stage-E fingerprint** (Phase-10 exit criterion 5) — Stage E's identity: the
  frozen reader and its dtype, the **prompt SHA**, the decoding config, the
  serialisation ordering rule and its declared signals, the claim-id format, the
  budget ladder, the answer-equivalence and scoring rules, the abstention
  aggregation convention and the declined post-hoc monitor. The prompt SHA is the
  mechanism v1.2 §3.5's "same frozen SLM, same prompt, same budget for every
  compared system" is enforced by — Phase 11's baselines reuse the same template,
  so one that reworded it shows up as a different fingerprint rather than as a
  better score. ``serialization_budget_tokens`` is deliberately absent: it is the
  config tree's and reaches identity through ``config hash``.

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
from graft.graphbuild.pins import endpoint_table_hash, stage_b_fingerprint
from graft.gate.pins import PRIMARY_METRIC, stage_g_fingerprint
from graft.setgen.pins import stage_d_fingerprint, training_blocked_reason
from graft.reader.pins import PROMPT_SHA, stage_e_fingerprint
from graft.retrieve.pins import SCORER, stage_c_fingerprint
from graft.ingest.pins import EXTRACTOR, ingestion_fingerprint
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
    # **Measured, after a first version of this comment overstated it.**  These
    # docstrings carry U+2192 and U+03B2, Windows consoles default to cp1252, and
    # `--help` died on the description before printing a word of it -- that part
    # is reproduced, on six runners.
    #
    # The guard also covers `print` of *data*, but the original justification
    # ("a curly apostrophe would kill the run") was **wrong**: U+2019 and U+2014
    # are cp1252 0x92/0x97 and encode fine.  What LoCoMo actually holds outside
    # cp1252 is 18 occurrences of 11 characters -- 8 zero-width spaces and 9
    # emoji -- across 7 turns and 1 gold answer.  And no current print path in
    # these runners emits corpus text, so this is insurance against a future
    # debug print, not a live crash averted.  `scripts/phase3_calibrate.py` set
    # the convention; extended here 19 Aug 2026.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
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
    print(f"stage-B print    {stage_b_fingerprint()}"
          f"   (endpoints {endpoint_table_hash()})")
    print(f"stage-C print    {stage_c_fingerprint()}"
          + ("" if SCORER["built"] else "   (scorer declared, not built - Gate 0 unsigned)"))
    print(f"stage-G print    {stage_g_fingerprint()}"
          f"   (gate primary: {PRIMARY_METRIC})")
    blocked = training_blocked_reason()
    print(f"stage-D print    {stage_d_fingerprint()}"
          f"   (Stage-D training: {'BLOCKED' if blocked else 'unblocked'})")
    print(f"stage-E print    {stage_e_fingerprint()}"
          f"   (prompt {PROMPT_SHA[:16]})")
    print(f"ingestion print  {ingestion_fingerprint()}"
          + ("" if EXTRACTOR is not None else "   (no extractor frozen yet - G2 bakeoff)"))
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

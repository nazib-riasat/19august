# GRAFT thesis research-paper library

This folder is the local literature library for the GRAFT thesis. It contains the papers cited in the project research plans, evidence audits, merged-plan review, and the associated research discussion.

## Folder map

- `01_GFlowNet` — foundations, objectives, credit assignment, exploration, constraints, symmetry, and flow/RL comparators.
- `02_Agent_Memory` — long-term agent memory systems and the two end-to-end memory benchmarks.
- `03_Graph_Construction` — heterogeneous/temporal GNNs, entity linking, joint extraction, and incremental graph computation.
- `04_Retrieval_and_Search` — GraphRAG, graph retrieval, iterative retrieval, combinatorial search, MCTS, A-star, PCST, MIP, beam, and submodular selection.
- `05_Citation_and_Evaluation` — citation generation, attribution, verification, long-context behavior, selective prediction, calibration, and statistical testing.
- `06_RL_and_Set_Baselines` — permutation-invariant set models and canonical PPO/GRPO/RL-retrieval baselines.
- `07_Datasets` — dataset and annotation-scheme papers supplying the supervision required by the Gate-0 data contract.

`papers.csv` is the searchable master index. It records each paper's title, venue, year, local filename, source URL, priority, and thesis relevance.

## Priority labels

- `Core` — directly defines the proposed architecture, objective, evaluation, or a mandatory baseline.
- `Baseline` — needed for a fair learning or search comparison.
- `Supporting` — supports an ablation, design decision, diagnostic, or failure analysis.
- `Provisional` — useful preprint, workshop, or lower-confidence evidence that must not carry a central claim alone.

A twelfth flag, **Marginal**, is recorded in the `Relevance` column (not `Priority`) and mirrored in `INDEX.md`. Marginal papers are retained as related work only; they weakened when plan v1.1 dropped dual epistemic mass, de-emphasised counterfactual credit, and removed incremental delta training as a contribution. Do not spend reading time on them. Currently marginal: SynFlowNet, Multi-Objective GFlowNets, Distributional GFlowNets, Hierarchical GFlowNet Data Synthesis, Amortizing Intractable Inference, InstantGNN, InkStream, FLARE, RARR, Data Shapley.

## Title accuracy

Titles in `INDEX.md` and `papers.csv` are the **verified published titles**, checked against the publisher record. Short names (SubTB, FL-GFN, LED-GFN, SEEM, ARM, TaG, HGERE, SynCheck, ConEL-2, 2WikiMultiHopQA, …) appear in parentheses and must **not** be used as titles in the thesis bibliography.

## Re-download or repair the library

Run from PowerShell (the explicit process-level bypass is needed on systems whose default policy disables local scripts):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\Research papers\download_papers.ps1"
```

The script skips valid existing PDFs. Add `-Force` after the filename to download every file again.

## Scope rule

The library deliberately includes papers that disagree with or weaken the proposed design. Counterevidence such as lightweight memory, simple graph baselines, full-context limitations, and strong combinatorial search is necessary for a defensible thesis.

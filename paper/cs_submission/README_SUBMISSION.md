# Overleaf package — Computers & Security submission

## Compile
1. Upload this folder (or the zip) to Overleaf → "New Project" → "Upload Project".
2. Main document `main.tex`, compiler **pdfLaTeX**.
3. Overleaf runs pdfLaTeX → BibTeX → pdfLaTeX ×2 automatically. `elsarticle.cls`,
   `elsarticle-harv.bst`, `algorithm`, `algpseudocode`, `siunitx` and `booktabs` all ship with
   Overleaf's TeX Live; nothing needs installing.

## Contents
```
main.tex     manuscript (elsarticle, preprint, author-year)
refs.bib     32 references, all cited
figs/        10 figures, all generated from experiment outputs (no hand-drawn diagrams, no TikZ)
```

| Figure | File | Shows |
|---|---|---|
| 1 | architecture.pdf | full system architecture: framing, two ID-agnostic branches, detectors, alarm logic, calibration path, evaluation harness |
| 2 | carhacking_schedule.pdf | Car-Hacking attack schedule + timer-vs-graph-network forecasting |
| 3 | model_comparison.pdf | events detected at two budgets, all models, 5 seeds |
| 4 | road_event_curves.pdf | events vs false alarms/h trade-off on ROAD |
| 5 | latency_tiers.pdf | latency by alarm rule; detection by attack tier |
| 6 | threshold_transfer.pdf | requested vs achieved false-alarm rate under two calibration protocols |
| 7 | cross_vehicle.pdf | cross-vehicle transfer per representation, four pairs |
| 8 | adaptation.pdf | unlabelled target-vehicle adaptation (0/1/5/15 min) |
| 9 | pair_union.pdf | paired detector vs individual branches |
| 10 | ablation.pdf | detector ablation (ROAD) and representation ablation (cross-vehicle) |

Tables 1–11 cover: corpora properties, reporting practice of related work, corpora as used,
notation, ROAD results, latency by alarm rule, the five evaluation traps, cross-vehicle results,
paired detector, ablations, and inference cost.

## Sections
Introduction (~990 words) · Related work (~1450 words, 2 tables) · Threat model · System
architecture (Fig. 1) · Mathematical modelling (notation table, Eqs. 1–8) · Proposed algorithms
(Algorithms 1–3) · Datasets · Experimental setup · Results (benchmarks, traps, cross-vehicle, paired
detector, ablation, cost) · Discussion · Reproducibility · Conclusion · Declarations.

## Reproducibility
Code and result summaries: **https://github.com/shishirbit/can-ids-evaluation** (public).
Every number in the manuscript is recorded in `experiments/paper_numbers.json` in that repository
together with the script that produced it; the figures are generated from that file by
`scripts/make_figures.py` and `scripts/make_figures2.py`, so text, tables and plots cannot diverge.

## Before you submit — please check
* **Bibliography completeness.** Authors/titles/venues/years are taken from indexed records. Several
  2025–2026 entries lack volume/pages/DOI and a few use "and others" where the indexed record did
  not expose the full author list. Complete them from the publisher records.
* **Author block, ORCIDs, funding, acknowledgements** are placeholders.
* **Journal fit.** Written for *Computers & Security*; switching to *Vehicular Communications* needs
  only the `\journal{}` line. The Taylor & Francis `interact.cls` template supplied earlier targets
  T&F journals (e.g. Transportmetrica B), whose scope is transport dynamics rather than in-vehicle
  security.
* **Anonymisation.** If the journal requires double-blind review, remove the GitHub URL from
  Sections "Reproducibility" and "Data availability" and replace with an anonymised link.
* **Abstract length** is ~330 words; check the journal limit (commonly 250).

## Verified build
Compiled locally with pdfLaTeX (TeX Live 2026) before delivery:
* **0 overfull boxes**, no undefined citations or references.
* 36 pages; all 10 figures and 11 tables placed within their own sections (pages 6-30), none
  pushed past the references.
* Figures are sized for this text width, so LaTeX scales them by less than 10% and the labels
  stay legible.

A compiled `main.pdf` is included in the project repository for reference; Overleaf will
regenerate it from `main.tex`.

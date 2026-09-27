# Overleaf package — Computers & Security submission

## Compile
1. Upload this whole folder (or the zip) to Overleaf → "New Project" → "Upload Project".
2. Set the main document to `main.tex`, compiler **pdfLaTeX**.
3. Compile order is handled by Overleaf (pdfLaTeX → BibTeX → pdfLaTeX ×2). `elsarticle.cls` and
   `elsarticle-harv.bst` ship with Overleaf's TeX Live; nothing needs installing.

## Contents
```
main.tex                 manuscript (elsarticle, preprint, author-year)
refs.bib                 bibliography
figs/carhacking_schedule.pdf   Fig. 1  attack schedule + timer-vs-model forecasting
figs/road_event_curves.pdf     Fig. 2  events detected vs false alarms/h on ROAD
figs/cross_vehicle.pdf         Fig. 3  cross-vehicle transfer per representation
figs/pair_union.pdf            Fig. 4  paired ID-agnostic detectors
```
Figures are regenerated from the experiment outputs by `scripts/make_figures.py` in the project
root; they are not hand-drawn.

## Before you submit — please check
* **Bibliography completeness.** `refs.bib` has verified authors/titles/venues/years, but several
  entries lack volume, pages or DOI (marked in the file header). Complete them from the publisher
  records. Two recent entries (`khreasat2026relational`, `hegde2026residualization`) use
  "and others" where the full author list was not in the indexed record.
* **Journal fit.** Written for *Computers & Security* (Elsevier). Switching to *Vehicular
  Communications* needs only the `\journal{}` line changed; both use `elsarticle`.
  The Taylor & Francis `interact.cls` template you supplied is for T&F journals
  (e.g. Transportmetrica B), whose scope is transport dynamics rather than in-vehicle security —
  see the note in the covering message.
* **Author block, ORCID, funding, acknowledgements** are placeholders.
* **Word limits / structured abstract**: not applied, because the publisher's author instructions
  page could not be retrieved automatically. Verify abstract length (currently ~300 words) and any
  required declarations against the live guide for authors.
* **Anonymisation**: the data-availability statement contains a placeholder repository URL.

## Numbers in the manuscript
Every figure and table value comes from files under `experiments/` in the project root
(`event_curves.md`, `vehicle_pairs.json`, `ecu_cost.json`, `road_detectors_v4/summary.json`,
`per_vehicle_calib/plain_s0.json`, `pair_detector/*.json`, `id_agnostic/*.json`). The README in the
project root maps each claim to the script that produced it.

# GALAXY-ITS — Phase 1: Spatio-Temporal Threat Forecaster (STTF)

Implementation of the STTF component of the GALAXY-ITS proposal
(`deepseek_text_20260922_bae663.txt`), scoped to run on a single 8 GB GPU (RTX 5060 Laptop).

## Setup

```bash
uv venv .venv --python 3.11
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv -e .
```
RTX 50-series (sm_120) needs a PyTorch build for CUDA 12.8 or newer.

## Data

```bash
bash scripts/download_car_hacking.sh                 # resumable, HCRL official Dropbox share
.venv/Scripts/python scripts/preprocess_car_hacking.py
```

## Run

```bash
# synthetic sanity checks (no data needed)
.venv/Scripts/python scripts/train_sttf.py --synthetic dos --epochs 20
.venv/Scripts/python scripts/train_sttf.py --synthetic dos --precursor --epochs 20

# real data
.venv/Scripts/python scripts/train_sttf.py --data data/processed/car_hacking --epochs 50
# ablation (a): no graph structure
.venv/Scripts/python scripts/train_sttf.py --data data/processed/car_hacking --no-graph
```
Results (metrics per horizon, thresholds, training history) go to `experiments/<name>/<timestamp>/results.json`.

## Layout

```
src/galaxy_its/data/can_io.py     CAN log readers -> CanLog (Car-Hacking CSV, normal-run TXT)
src/galaxy_its/data/graphs.py     100 ms windows -> dynamic CAN-ID graphs, node features, nested forecast labels
src/galaxy_its/data/dataset.py    chronological splits with purge gaps, GPU-resident batching
src/galaxy_its/data/synthetic.py  synthetic CAN traffic, optional attack precursors
src/galaxy_its/models/sttf.py     dense GAT + Transformer + multi-horizon heads + MC dropout
src/galaxy_its/eval/metrics.py    F1/AUC/PR-AUC/Brier/FPR/FNR/MCC per horizon, all vs pre-onset
src/galaxy_its/eval/baselines.py  persistence and periodicity sanity baselines
scripts/                          download, preprocess, train
```

## Design decisions vs. the proposal

| Proposal | Implemented | Why |
|---|---|---|
| Horizons 1/5/10 s (C1) vs 1/2/5/9 s (3.2.2) | 0 (detection), 1, 2, 5, 9 s | Resolves the inconsistency; 0 s gives a detection reference |
| "Attack likelihood at lead time H" (undefined) | y = any attack in [t, t+H] (nested) | Well-defined, monotone in H, enables the consistency loss |
| "Auxiliary temporal consistency loss" (undefined) | penalty on P(H_k) > P(H_k+1) | Follows directly from nested labels |
| PyTorch Geometric GAT | dense GAT batched over B*L graphs | Graphs have ~30–50 nodes; faster, no compiled extensions on Windows |
| d_model 256 | 128 (configurable) | 8 GB VRAM |
| SMOTE on training data | class-weighted BCE (pos_weight) | SMOTE does not apply to sequences of graphs |
| Stratified + temporal split | chronological 70/15/15 per log with purge gap L + max(H) | Stratification would leak future into train |
| — | **pre-onset** evaluation subset | Separates true forecasting from detection of ongoing attacks |
| — | persistence & periodicity baselines | Exposes gains that come from detection or from the dataset's attack schedule |

## Findings so far

* **Synthetic, no precursor:** STTF pre-onset AUC ≈ 0.5 (cannot forecast), while the periodicity
  baseline reaches 0.67–0.79 from the injection schedule alone.
* **Synthetic, 3 s precursor:** STTF pre-onset AUC 0.98 at 1–2 s, 0.76 at 5 s, fails at 9 s
  (beyond the precursor lead) — the model forecasts when a signal exists.
* **Car-Hacking Fuzzy:** 300 attacks of ~5 s separated by gaps of exactly 3.0 s (CV 0.003).
  The file also holds two recordings joined by a 2540 s silence (attack session 0–2466 s,
  attack-free drive afterwards); logs are now split at recording gaps before the temporal split.
* **Fuzzy test results (pre-onset = genuine forecasting windows):**

  | horizon | STTF F1 / AUC | periodicity (time-since-last-attack only) F1 / AUC |
  |---|---|---|
  | 0 s (detection) | 0.995 / 1.000 | — |
  | 1 s | 0.591 / 0.851 | **0.993 / 1.000** |
  | 2 s | 0.801 / 0.877 | **0.993 / 0.996** |
  | 5 s | 0.870 / 0.847 | **0.994 / 0.988** |

  A single-feature timer beats STTF at every forecast horizon.
* **Data quality (all Car-Hacking files):** every attack file has 300 bursts with 3.0 s gaps
  (gap CV 0.001–0.003), and the same 481.6 s attack-free recording (891,068 identical frames) is
  appended to all four attack files; duplicates are dropped at load time.
* **Full Car-Hacking (DoS + Fuzzy + gear + RPM + normal), test split, pre-onset AUC-ROC:**

  | horizon | STTF (graph) | STTF no-graph (ablation a) | periodicity timer |
  |---|---|---|---|
  | 0 s detection (F1 / AUC, all windows) | 0.9997 / 1.000 | 0.9995 / 1.000 | — |
  | 1 s | 0.901 | 0.967 | **0.980** |
  | 2 s | 0.840 | **0.962** | 0.924 |
  | 5 s | 0.694 | **0.933** | 0.856 |

  Detection is saturated, as in the literature.
* **Direction A (chosen 2026-09-22): early detection of stealthy attacks on ROAD.**
  ROAD: one injection per capture (15 events + 14 masquerade copies), 3-fold capture-level CV,
  an attack and its masquerade copy always share a split. Attack detectability falls in three tiers:
  rate (injection doubles an ID's rate), value (never-seen byte values: speedometer, coolant,
  correlated signal, fuzzing) and contextual (reverse light on/off: a normal value that is
  wrong only given other IDs' signals).
  Non-deep baselines, pooled over folds (`scripts/detect_road.py`):

  | detector | AUC | false alarms/h | value attacks detected | contextual (reverse light) detected |
  |---|---|---|---|---|
  | rate exceedance (unsup.) | 0.753 | 772 | 15/17 | 10/12 |
  | payload surprise exceedance (unsup.) | 0.681 | 6269 | 17/17 | **0/12** |
  | window GBDT (sup., all IDs, no graph/history) | **0.882** | 159 | 15/17 | **12/12** |

  Contextual attacks need cross-ID evidence but not necessarily a GNN: a flat GBDT over all IDs
  finds them. Thresholds tuned for window-F1 give unusable false-alarm rates; evaluation must
  move to operating points (detection rate at a false-alarm budget, k-of-n alarm smoothing).
  STTF on ROAD (3-fold CV, 29 events; `--payload --per-node-norm` = v2):

  | model | events detected | window AUC (0 s) | false alarms/h at F1 threshold |
  |---|---|---|---|
  | v1 graph (timing features only) | 20/29 | 0.915 | 292 |
  | v2 no-graph | 23/29 | — | 2 |
  | v2 graph | **27/29** | **0.986** | 785 |
  | window GBDT | 29/29 | 0.882 | 159 |

  Payload features add +7 events, the graph adds +4 more, and v2 beats the GBDT on AUC.
  **v3 = v2 + `--ambient-holdout`** (whole ambient captures held out per split instead of time
  slices of the same drives), evaluated with event-level alarms (k-of-n rule, false alarm = rising
  edge on attack-free traffic) and an alarm duty-cycle cap. Two evaluation traps found here:
  a threshold so low that the alarm never clears has one rising edge per recording and looks like
  a *low* false-alarm rate (hence the duty cap), and calibrated single operating points confound
  model quality with threshold transfer (hence the sweep below).

  Event-level trade-off on test, 3-of-5 rule, duty <= 2%, pooled over folds (`scripts/event_curve.py`,
  seed 0, 29 events, 1.2 h held-out normal driving):

  | model | detected @10 FA/h | @30 FA/h | @60 FA/h |
  |---|---|---|---|
  | rate exceedance (unsup.) | 3/29 | 3/29 | 3/29 |
  | payload novelty (unsup.) | 0/29 | 0/29 | 0/29 |
  | window GBDT (no graph/history) | 20/29 | 20/29 | 20/29 |
  | STTF v1 graph (timing features only) | **20/29** | 20/29 | 20/29 |
  | STTF v3 no-graph (payload) | 18/29 | 22/29 | 22/29 |
  | STTF v3 graph (payload) | 7/29 | **27/29** | **27/29** |

  Over 3 seeds (mean +- std of events detected):

  | model | @10 FA/h | @30 FA/h |
  |---|---|---|
  | STTF v3 graph | 19.3 +- 10.9 (7, 27, 24) | **26.7 +- 0.6** |
  | STTF v3 no-graph | 13.7 +- 5.9 (18, 16, 7) | 15.3 +- 7.6 (22, 17, 7) |

  **Seed ensembling (mean probability over the 3 seeds) fixes the tight-budget regime**
  (`scripts/ensemble_norm.py`):

  | model | @5 FA/h | @10 FA/h | @30 FA/h |
  |---|---|---|---|
  | graph, 3-seed ensemble | **27/29** | **27/29** | 27/29 |
  | graph + causal per-capture normalisation | 19/29 | 19/29 | 23/29 |
  | no-graph, 3-seed ensemble | 0/29 | 20/29 | 23/29 |

  27/29 events at 5 false alarms per hour is the first deployable operating point here, and the
  graph advantage is largest exactly there (27 vs 0). Causal per-capture score normalisation
  (running median/IQR over the preceding 60 s) *hurts*: near-binary scores give a tiny running IQR,
  so dividing by it amplifies noise - tested and rejected.

  At >= 30 alarms/h the graph model is better *and* stable (26.7/29, std 0.6) even single-seed.

  **Re-implemented literature baselines** on raw frame windows (64 consecutive frames, stride 16,
  scores mapped onto the same 100 ms windows; `scripts/baseline_frames.py`):

  | model | @5 FA/h | @30 FA/h | per-seed @5/h |
  |---|---|---|---|
  | STTF v3 graph, 3-seed ensemble | **27/29** | 27/29 | 7 / 27 / 24 |
  | ConvGRU (CNN+GRU on frames), 3-seed ensemble | 21/29 | 27/29 | 25 / 25 / 22 |
  | window GBDT | 20/29 | 20/29 | - |
  | FrameAE (reconstruction, unsup.) | 0/29 | 0/29 | - |

  **5 seeds, fine threshold grid (200 candidates), significance tests** (`scripts/compare_models.py`),
  3-of-5 alarm rule, events detected of 29:

  | model | params | @5 FA/h | @30 FA/h |
  |---|---|---|---|
  | STTF graph (payload + graph attention) | 1.05 M | 22.4 +- 7.8 | 26.8 +- 0.4 |
  | ConvGRU on raw frames | 98 k | **26.2 +- 1.2** | **27.0 +- 0.0** |
  | ConvGRU + payload novelty | 98 k | 26.2 +- 1.0 | 27.0 +- 0.0 |

  * **All three models are statistically indistinguishable** (ConvGRU vs ConvGRU+payload p=1.000,
    identical event sets; vs STTF p=0.81/0.42) and hit the same ceiling of 27/29 events. The two
    events nobody detects are the coolant-temperature attack and its masquerade copy (43 injected
    frames at ~10 Hz - the sparsest attack in ROAD).
  * A **98 k-parameter CNN+GRU matches the 1.05 M-parameter graph model**, and is more stable
    across seeds (+-1.2 vs +-7.8 events at 5 alarms/h). Neither graph attention nor the
    payload-novelty features add measurable detection at these budgets.
  * Earlier readings of "graph helps" (+11 events) and "payload features help" (p=0.029) were both
    **artefacts of a coarse threshold grid** (60 candidates); with 200 they vanish. Grid resolution,
    seed count, the alarm duty cap and the ambient-split design each changed conclusions here -
    that fragility is the paper's central methodological message.

  Durable contributions: the evaluation protocol, the detectability tiers, the dataset artefacts,
  and the negative result that model complexity does not pay on ROAD.

  **Latency / alarm-rule trade-off** (ConvGRU+payload at 5 alarms/h, 5 seeds). The alarm rule, not
  the model, sets the floor on detection latency (k windows x 100 ms):

  | rule | events detected | median latency | note |
  |---|---|---|---|
  | 1-of-1 | 16.2 +- 13.2 (27,27,27,0,0) | 0 ms (<= 100 ms) | 2 seeds have no feasible threshold |
  | 2-of-3 | 21.2 +- 10.6 (26,27,27,26,0) | 112 ms | 1 seed infeasible |
  | 3-of-5 | **26.2 +- 1.0** | 220 ms | stable |

  The proposal's "<100 ms detection latency" is reachable only with an unsmoothed alarm, which is
  unstable across seeds; the stable configuration costs ~200-300 ms. Earlier "0 ms median latency"
  figures came from a raw single-window threshold rather than the alarm rule actually reported.

  **ECU cost** (`scripts/ecu_cost.py`, 1 CPU thread, batch 1):

  | model | params | int8 size | latency (median / p99) | required rate | duty on one core |
  |---|---|---|---|---|---|
  | ConvGRU + payload | 98 k | 0.10 MB | 1.35 / 1.69 ms | 150/s | 20% |
  | STTF graph | 1.05 M | 1.05 MB | 38.9 / 70.1 ms | 10/s | 39% |
  | STTF + 50x MC dropout (as proposed) | 1.05 M | 1.05 MB | 1945 ms | 10/s | **1945% (~20 cores)** |

  **Cross-vehicle generalisation** (can-train-and-test-v1.5 `set_01`: train on Chevrolet Impala,
  test on Chevrolet Silverado; the repository's own split; `scripts/cross_vehicle.py`, ConvGRU with
  and without payload features, 3-of-5 rule):

  | test set | events | FA/h | alarm duty | median latency |
  |---|---|---|---|---|
  | known vehicle, known attacks | 7/8 | 0-22 | 0.7% | 200 ms |
  | unknown vehicle, threshold from known vehicle | 8/8 | 21.9 | **99.9%** | 0 ms |
  | unknown vehicle, threshold from its own clean traffic | **0/8** | 0.0 | 0.0% | - |
  | masquerade (known vehicle) | 6-8/8 | 0-28 | 0.0% | 450-1000 ms |

  **The model does not transfer across vehicles at all.** On the unknown vehicle the alarm is
  active 99.9% of the time, so "8/8 detected" is an artefact of flagging everything; forced to meet
  a false-alarm budget on that vehicle's own clean traffic it detects nothing. Removing the
  payload-novelty features changes nothing, so this is not a feature artefact - the learned ID
  embeddings and traffic patterns are vehicle-specific. Per-vehicle calibration (or per-vehicle
  training) is a prerequisite, not an optimisation.

  **Per-vehicle calibration does not rescue it** (`scripts/per_vehicle_calib.py`): giving the
  Impala-trained model N minutes of the Silverado's *attack-free* traffic (IDs added to the
  vocabulary, frames added as negatives, threshold calibrated on a held-out 20%, evaluated on target
  attack files never used for adaptation):

  | adaptation | events detected with the budget actually honoured on test |
  |---|---|
  | 0 min | 0/8 |
  | 1 min | 0/8 (raw 4/8 at 175-391 FA/h - budget violated) |
  | 5 min | 0/8 @5/h, 1/8 @30/h |
  | 15 min | 0/8 (raw 3/8 at 249-376 FA/h - violated) |

  No trend with more traffic, so this is not a data-volume problem. Mechanism: the target vehicle's
  ID embeddings are trained only on normal traffic, so the classifier has never seen an attack on
  those IDs (target has 98 IDs vs the source's 52). Unlabelled normal traffic from a new vehicle is
  therefore insufficient; per-vehicle training with labelled attacks, or an ID-agnostic
  representation, is required. Caveats: one vehicle pair, 8 events, one architecture.

  **ID-agnostic features do transfer** (`src/galaxy_its/data/relative.py`, `scripts/id_agnostic.py`).
  Each frame is described only relative to its own arbitration ID (cadence ratio vs that ID's
  warm-up median inter-arrival, payload bit/byte change rates, excursion beyond that ID's warm-up
  byte range, per-ID and bus rate ratios, DLC change) - 12 features, no ID embedding, no
  cross-vehicle vocabulary. Per-ID references come from a 30 s warm-up prefix of each capture
  (unsupervised, label-free). ConvGRU, 3 seeds, threshold calibrated on the target's own clean
  traffic (no target labels):

  | test set | ID-based model | ID-agnostic model | FA/h | median latency |
  |---|---|---|---|---|
  | known vehicle, known attacks | 7/8 | 7.0 +- 0.0 / 8 | 1.2 | 200 ms |
  | **unknown vehicle**, known attacks | **0/8** | **6.3 +- 1.7 / 8** (7.0 +- 0.8 @30/h) | 0.0 | 450 ms |
  | masquerade (known vehicle) | 6-8/8 | 0.0-0.7 / 7 | 0-14 | - |

  **Replicated on all four vehicle pairs** (`scripts/summarize_vehicles.py`), unknown-vehicle events
  detected at <= 5 FA/h, threshold from the target's own clean traffic, 3 seeds:

  | pair | ID-based | ID-agnostic |
  |---|---|---|
  | Impala -> Silverado | 0/8 | 6.3 +- 1.7 / 8 |
  | Traverse -> Forester (cross-make) | 0/12 | **12.0 +- 0.0 / 12** |
  | Silverado -> Forester (cross-make) | 0/10 | **10.0 +- 0.0 / 10** |
  | Forester -> Traverse (cross-make) | 0/14 | 6.0 +- 0.0 / 14 |
  | total | **0/44** | **34.3/44 (78%)** |

  Masquerade attacks remain the ID-agnostic blind spot (0-4.7 of 4-8 per pair), and the ID-based
  model beats it there on two pairs (7/8, 5/8). Caveat: the ID-agnostic pipeline excludes windows
  inside the 30 s warm-up, so attacks starting in the first 30 s are dropped - some event
  denominators therefore differ between the two representations (e.g. known-vehicle set_04: 12 vs 14).

  So the vehicle gap is a *representation* problem, not an inherent limit: dropping ID identity
  recovers most cross-vehicle detection (0/8 -> 6.3/8) at <= 0 false alarms/h, with no labels from
  the target vehicle. The cost is masquerade attacks, which ID-relative features miss entirely
  (timing stays normal and substituted payloads stay inside the warm-up range) - the same
  contextual-tier blind spot seen on ROAD. Calibrating on the *source* vehicle's clean traffic still
  fails (22-50% alarm duty), so per-vehicle threshold calibration remains necessary.

  **Comparison with published per-ID behavioural residualisation** (re-implemented from its method
  description in `src/galaxy_its/data/residual.py`: 14 temporal/protocol/payload window statistics
  per ID, z-scored against that ID's warm-up baseline), same model and protocol, unknown vehicle at
  <= 5 FA/h, 3 seeds:

  | pair | ID-based | residualisation | ours (relative) |
  |---|---|---|---|
  | Impala -> Silverado | 0/8 | 2.7 +- 0.5 | **6.3 +- 1.7** |
  | Traverse -> Forester | 0/12 | **12.0** | **12.0** |
  | Silverado -> Forester | 0/10 | **10.0** | **10.0** |
  | Forester -> Traverse | 0/14 | 5.7 +- 0.5 | 6.0 +- 0.0 |
  | total | **0/44** | 30.4/44 (69%) | **34.3/44 (78%)** |
  | masquerade (26 events) | - | **10.3/26 (40%)** | 4.7/26 (18%) |

  Both ID-agnostic representations transfer across vehicles, so the finding "ID-agnostic features
  transfer, ID-embedding features do not" is robust rather than specific to our feature set - the
  published residualisation representation would transfer too, which nobody had tested. Our
  per-frame ratios are better at transfer (78% vs 69%, difference driven entirely by set_01) while
  residualisation is clearly better on masquerade (40% vs 18%), i.e. the two are **complementary and
  both transferable** - a better pairing than relative + ID-based.

  **Concatenating the two feature sets fails** (`--features both`, 26 features/frame, 4 pairs x 3 seeds):
  unknown-vehicle detection drops to 23.3/44 (53%) at 5 FA/h and 60% even at 30 FA/h, versus 78%/80%
  for the relative features alone; masquerade lands between the two (31%). On set_03 the combined
  model collapses to 0/10 because its scores on the *target vehicle's clean traffic* saturate near
  1.0, so no threshold meets the budget - with the source vehicle's threshold it still finds 10/10,
  i.e. discrimination partly survives while calibration fails. Complementary representations
  therefore have to be combined as **separate detectors with OR-ed alarms**, not as one feature
  vector.

  **Paired ID-agnostic detectors are the design that works** (`scripts/pair_detector.py
  --branches rel+res`, alarms OR-ed, each branch given half the budget, thresholds calibrated on the
  target vehicle's own clean traffic - no target labels). Unknown vehicle, pooled 4 pairs x 2 seeds:

  | branch | @5 FA/h | FA/h | @30 FA/h |
  |---|---|---|---|
  | A: ID-relative | 67% (59/88) | 1.2 | 70% |
  | B: per-ID residualisation | 69% (61/88) | 1.3 | 72% |
  | **A OR B** | **81% (71/88)** | **0.3** | **83% (73/88)** |
  | masquerade (52 events) | A 19%, B 37% | | **A OR B 38% @5/h, 52% @30/h** |

  The union beats either transferable branch by 12-14 points at a *lower* false-alarm rate, and
  roughly doubles masquerade coverage relative to ID-relative features alone - on a vehicle never
  seen in training and with no labels from it. Note: absolute numbers are not comparable across
  scripts (the paired script evaluates only windows both branches score, and uses 2 seeds), so
  compare branches within a table, not across tables.

  **Paired detector, ID-based branch variant** (`scripts/pair_detector.py`): run both branches and alarm if either fires -
  branch A (ID-agnostic, transfers to any vehicle) and branch B (ID-based + payload novelty, trained
  per vehicle, catches masquerade). Each branch gets half the false-alarm budget; evaluated on the
  known vehicle where both are valid, pooled over 4 vehicle pairs x 2 seeds:

  | test | budget | A id-agnostic | B id-based | **A OR B** | union FA/h |
  |---|---|---|---|---|---|
  | known attacks (84 events) | 5/h | 88% | 80% | **92%** | 0.4 |
  | known attacks | 30/h | 90% | 92% | **93%** | 19.3 |
  | masquerade (52 events) | 5/h | 29% | 58% | **63%** | 0.0 |
  | masquerade | 30/h | 35% | 65% | **69%** | 12.5 |

  The union beats the better branch in all four cases *and* has a lower false-alarm rate than either
  branch alone (each runs at half the budget), so it is strictly better rather than a trade-off. It
  also keeps branch A's property of working on an unseen vehicle with no labels, while branch B adds
  masquerade coverage once the vehicle has labelled attacks. Masquerade coverage is still only
  63-69%, the main open problem.

  Fifth evaluation trap: the alarm duty cap must be enforced on *test* metrics too, not only when
  choosing the threshold - otherwise a permanently-firing detector reports perfect detection.

  Feature extraction is negligible (0.3% of a core for graph + payload features over real traffic).
  The proposal's 50-pass MC-dropout uncertainty is infeasible by ~20x on a single core; a laptop
  core is also several times faster than a typical automotive core, so these are optimistic bounds.
  Below that both variants are seed-unstable (the seed-0 graph run scored 7/29, seed 1 scored 27/29),
  so single-seed results at tight budgets are not interpretable: error bars are mandatory here.
  Open: per-capture score normalisation and more diverse ambient training drives for the tight-budget
  regime; ensembling to damp seed variance.

  With time-sliced ambient (v2), **thresholds did not transfer**: calibrated on validation for 60 alarms/h they give
  1244-2866/h on test, because validation holds only ~10 min of normal driving (middle slice of
  each ambient capture) while test is the final slice. Next: hold out *whole* ambient captures
  for calibration/test, and per-capture score normalisation. No configuration is deployable yet.
  Pitfall found: byte-value novelty with exact 256-value histograms flags normal signal drift
  (84% of held-out normal windows); coarse, smoothed value bins fix this. For forecasting, the graph structure *hurts*
  (ablation a reverses the proposal's expectation), and the gain of the no-graph model over the
  timer at 2–5 s is small and most plausibly also schedule-driven (a 2 s history can see the tail
  of the previous burst). Neither result supports a precursor-forecasting claim on this dataset. Car-Hacking supports
  **detection** results only; forecasting claims need datasets with irregular attack timing
  (e.g. ROAD, can-train-and-test) or must be reported against this baseline.

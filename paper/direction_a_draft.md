# Revision plan and draft sections — GALAXY-ITS → direction A (early detection)

Status: 3-fold capture-level CV on ROAD; STTF rows are mean ± std over 3 seeds, baseline rows are
single-run (deterministic given the fold). Remaining before submission: per-capture score
per-vehicle calibration experiments (given §5c), and 2–3 further published baselines cited
rather than re-run.

## 1. What changes relative to `deepseek_text_20260922_bae663.txt`

| Original claim | Evidence | Revision |
|---|---|---|
| C1 STTF *forecasts* attacks 1–9 s ahead | On Car-Hacking a 1-feature timer beats STTF at every horizon (pre-onset AUC 0.98 vs 0.90 at 1 s); on ROAD the timer reaches 0.91–0.94 pre-onset from "time since capture start" while STTF gets 0.58–0.68 | **Drop forecasting.** Contribution becomes *early detection*: 26.7 ± 0.6 of 29 ROAD attacks within one 100 ms window at 30 false alarms/h |
| "Graph structure is essential" | Car-Hacking: removing the graph *improves* results. ROAD, 5 seeds: STTF graph 22.4 ± 7.8 vs a plain CNN+GRU 22.0 ± 4.1 at 5 alarms/h (p=0.34), identical ensemble event sets | **Drop the architecture claim.** Cross-ID *features* are what contextual attacks need; graph attention is one way to use them and is not required |
| ">98% accuracy" targets | Car-Hacking detection is saturated (AUC 1.000 for both variants; a published baseline already reports 100%) | Report Car-Hacking as a sanity check only; headline results on ROAD |
| Six reproduced SOTA baselines | 2 re-implemented on identical splits (ConvGRU, FrameAE) + GBDT + 2 unsupervised detectors; ConvGRU matches STTF | Report these 5 as re-implementations; cite the rest as published numbers, clearly marked as not re-run |
| Gaps 1, 2, 5 (post-factum detection, flat temporal modelling, cross-domain) | — | Replace Gap 1 with the *evaluation* gap: no public CAN dataset supports onset forecasting, and CAN IDS papers rarely report false alarms per hour or threshold transfer |

## 2. New abstract (draft)

In-vehicle intrusion detection is usually evaluated on datasets whose attacks follow a fixed
timetable, with thresholds tuned per window and no reported false-alarm rate. We show that this
protocol rewards models for learning the injection schedule rather than the attack: on the
HCRL Car-Hacking dataset, whose 300 attacks are separated by gaps of exactly 3.0 s
(CV 0.001–0.003), a logistic model with a single feature — time since the last attack — matches
or beats a spatio-temporal graph network at every forecast horizon, and window-level detection is
saturated (AUC 1.000). On the ROAD dataset, where each capture contains one physically verified
injection, we show that attack detectability falls into three tiers — rate, value and contextual —
and that the third tier (a normal signal value that is wrong only given other IDs) is invisible to
per-ID models: a per-(ID, byte) novelty detector finds 17/17 value attacks and 0/12 contextual ones.
We then present STTF, a spatio-temporal detector combining per-ID payload-novelty features with
cross-ID graph attention and Monte Carlo dropout, evaluated under a deployment-oriented protocol:
capture-level cross-validation, whole held-out ambient drives for calibration, k-of-n alarm
smoothing, an alarm duty-cycle cap, and event-level detection swept against a false-alarm budget.
On ROAD, a 98 k-parameter CNN+GRU over raw frames detects 26.2 ± 1.2 of 29 attack events —
including all masquerade variants — at 5 false alarms per hour of held-out normal driving, matching
a 1.05 M-parameter spatio-temporal graph model (22.4 ± 7.8; p = 0.81 over 5 seeds, identical event
sets) and reaching the same 27/29 ceiling; the only events no model detects are a coolant-temperature
injection of 43 frames at 10 Hz and its masquerade copy. Detection latency is set by the alarm rule
rather than the model: an unsmoothed alarm detects within one 100 ms window but has no feasible
threshold for 2 of 5 seeds, while a stable 3-of-5 rule costs 220 ms. The cheap model needs 0.10 MB
int8 and 20% of one CPU core, whereas the 50-pass Monte-Carlo-dropout uncertainty specified in
earlier work would need ~20 cores. We further show that five evaluation choices — threshold-grid resolution,
seed count, an alarm duty-cycle cap during threshold selection, the same cap applied to reported
metrics, and whether held-out normal driving comes from different drives — each individually reverse
the apparent ranking of these models. On a second
dataset (can-train-and-test) the same detectors do not transfer between vehicles at all: with the
source vehicle's threshold the alarm is active 99.9% of the time on the target vehicle, and under a
false-alarm budget detection falls to 0 of 8 events, which neither more adaptation traffic (up to
15 min) nor the payload features repair. Replacing arbitration-ID identity with ID-relative frame
features — cadence and payload statistics measured against each ID's own warm-up behaviour —
recovers 6.3 ± 1.7 of 8 target-vehicle events at zero measured false alarms per hour with no labels
from that vehicle, at the cost of masquerade attacks, which the relative view cannot see. We release
code, splits and per-event results.

## 3. Method (Section III, revised)

3.1 Windowing and graph construction — unchanged from the proposal (100 ms windows, dynamic
CAN-ID graph, transition-frequency adjacency), with the ID vocabulary fitted on training traffic
only and one UNK node absorbing unseen IDs (DoS 0x000, fuzzing IDs).

3.2 Node features (32 per ID per window)
  * timing/volume (8): log count, presence, inter-arrival mean/std, bit-flip fraction, unchanged-
    payload fraction, mean byte value, mean DLC
  * payload (24): per-byte mean, per-byte change fraction, and **per-byte novelty**
    −log p(bin | ID)/log B against a coarse-binned, neighbour-smoothed profile of normal traffic.
    Exact 256-value histograms flag ordinary signal drift as novel (84% of held-out normal
    windows); 32 smoothed bins remove this while keeping far-out values (0xFF on a byte that is
    normally 0x02) maximally surprising.

3.3 Architecture — dense multi-head graph attention over IDs (edge weights bias attention,
non-edges masked), Transformer over the window sequence, per-horizon heads, MC dropout for
uncertainty. 1.0 M parameters; trains in ~20 min on one 8 GB laptop GPU.

3.4 What we removed — LZTO and GADT are out of scope for this paper (see §7 for why: the LLM
policy engine cannot meet the enforcement latency budget, and adversarial co-evolution needs a
CAN simulator that SUMO/CARLA/OMNeT++ do not provide).

## 4. Evaluation protocol (Section IV, revised) — the second contribution

1. **Capture-level CV.** ROAD has 15 injection events; each has up to 3 instances. Fold f tests
   instance f+1, validates on the next, trains on the rest. An attack and its `_masquerade`
   variant (the same event with conflicting frames removed) always share a split.
2. **Whole held-out ambient drives.** Calibration and test use *different drives*, not slices of
   the same drive. With time-sliced ambient, a threshold calibrated for 60 alarms/h produced
   1244–2866/h on test; with whole-drive holdout it produces 29/h.
3. **Event-level alarms.** k-of-n smoothing; a false alarm is a rising edge on attack-free
   traffic; detection means the rule fires inside the attack episode; latency is measured from
   episode onset.
4. **Report at a false-alarm budget** (60/h and 10/h here), not at the window-F1 optimum, which
   yields 159–785 alarms/h.
5. **Sanity baselines** that must be beaten: persistence (current window's label), a timing-only
   model, per-ID rate exceedance, per-(ID, byte) novelty exceedance, and a GBDT over all IDs'
   features in the current window (no graph, no history).

Threshold-transfer resolution caveat: the budget is only as precise as the held-out normal
driving allows (~1.2 h pooled → ±1 alarm ≈ 0.8/h).

## 5. Results (Section V) — 5 seeds, 3-fold capture-level CV

Table 1. ROAD, pooled over 3 folds, 29 attack events, 1.2 h held-out normal driving. Event-level
detection at a false-alarm budget (3-of-5 alarm rule, alarm duty <= 2% of clean time), threshold
swept on test so the comparison is not confounded by threshold transfer.

| model | window AUC (100 ms) | detected @10 FA/h | @30 FA/h | @60 FA/h |
|---|---|---|---|---|
| rate exceedance (unsup.) | 0.748 | 3/29 | 3/29 | 3/29 |
| payload novelty (unsup.) | 0.342 | 0/29 | 0/29 | 0/29 |
| window GBDT (no graph/history) | 0.963 | 20/29 | 20/29 | 20/29 |
| STTF timing features only (1 seed) | 0.915 | 20/29 | 20/29 | 20/29 |
| STTF + payload, no graph | — | 13.7 ± 5.9 | 15.3 ± 7.6 | 15.3 ± 7.6 |
| **STTF + payload + graph** | **0.986** | 19.3 ± 10.9 | **26.7 ± 0.6** | **26.7 ± 0.6** |

STTF rows: mean ± std over 3 seeds (per-seed values — graph @10: 7, 27, 24; @30: 27, 27, 26;
no-graph @10: 18, 16, 7; @30: 22, 17, 7).

Table 1b. Post-hoc variance reduction (no retraining), events detected at a budget.

| model | @5 FA/h | @10 FA/h | @30 FA/h |
|---|---|---|---|
| graph, 3-seed ensemble | **27/29** | **27/29** | 27/29 |
| graph, single seed (min–max) | 7–27 | 7–27 | 26–27 |
| graph + causal per-capture normalisation | 19/29 | 19/29 | 23/29 |
| no-graph, 3-seed ensemble | 0/29 | 20/29 | 23/29 |

Averaging seeds' per-window probabilities helped STTF at 3 seeds (27/29 at 5 alarms/h) but does
not generalise: with 5 seeds ConvGRU+payload is *worse* ensembled (23) than per seed
(26.2 ± 1.0), so Table 1d reports per-seed statistics. Causal per-capture score normalisation
(running median/IQR over the preceding 60 s, online) was tested and *rejected*: the detector's
scores are near-binary, so the running IQR is tiny and dividing by it amplifies noise (23/29 at
30 alarms/h vs 27/29 raw).

Table 1c. Re-implemented literature baselines on raw frame windows (64 consecutive frames,
stride 16, scores mapped onto the same 100 ms evaluation windows).

| model | @5 FA/h | @30 FA/h | per-seed @5/h |
|---|---|---|---|
| STTF v3 graph, 3-seed ensemble | **27/29** | 27/29 | 7 / 27 / 24 |
| ConvGRU (CNN+GRU on frames), 3-seed ensemble | 21/29 | 27/29 | 25 / 25 / 22 |
| window GBDT (our features, no graph/history) | 20/29 | 20/29 | - |
| FrameAE (reconstruction, unsupervised) | 0/29 | 0/29 | - |

Table 1d. 5 seeds, events detected of 29 (mean ± std over seeds; significance in the text).

| model | @5 FA/h | @30 FA/h | ensemble @5/h |
|---|---|---|---|
| STTF (payload + graph attention) | 22.4 ± 7.8 | 26.8 ± 0.4 | 27 |
| ConvGRU on raw frames | 22.0 ± 4.1 | 25.0 ± 1.9 | 27 |
| **ConvGRU + payload novelty** | **26.2 ± 1.0** | 26.2 ± 1.0 | 23 |

**No model claim survives.** With 5 seeds and a 200-point threshold grid, STTF graph, ConvGRU and
ConvGRU+payload are statistically indistinguishable (ConvGRU vs ConvGRU+payload p = 1.000 with
identical event sets; STTF vs either p = 0.81 at 5 alarms/h, p = 0.42 at 30) and share a 27/29
ceiling. The 98 k-parameter model is 10× smaller, 29× faster per inference, and more stable across
seeds (±1.2 vs ±7.8 events). Unsupervised reconstruction (FrameAE) and per-ID novelty thresholding
detect nothing at any usable budget.

Four evaluation choices each reversed the apparent ranking during this study, which is the core
methodological result:
1. **threshold-grid resolution** — 60 candidates gave "payload features help, p = 0.029"; 200 gave
   p = 1.000;
2. **seed count** — 3 seeds gave "graph attention adds 11 events"; 5 seeds gave p = 0.81;
3. **alarm duty cap** — without it, a permanently-on alarm scores as "29/29 at 29 false alarms/h";
4. **ambient split** — time slices of the same drives made thresholds appear to transfer
   (60/h budget → 1244–2866/h on unseen drives).
Any of these alone is enough to publish a spurious architectural improvement.

Two regimes (single seed):
* **≥ 30 alarms/h — the headline claim.** Payload features plus cross-ID graph attention detect
  26.7 ± 0.6 of 29 events, including all 13 masquerade variants, within a single 100 ms window.
  Removing the graph costs 11 events and multiplies the seed spread by 12× (15.3 ± 7.6).
* **≤ 10 alarms/h — not yet deployable.** Both variants are seed-unstable (graph: 7, 27, 24), so
  single-seed numbers in this regime are uninterpretable. Two causes, separable in future work:
  novelty scores are elevated on *unseen drives* (drive-to-drive variation), and run-to-run
  variance is large at strict thresholds. Remedies to test: per-capture score normalisation, more
  diverse ambient training drives, and ensembling.

Methodological point for reviewers: a single calibrated operating point would have reported either
27/29 or 7/29 for the same model depending on the seed, and a threshold low enough to keep the
alarm permanently on scores as "29/29 at 29 false alarms/h" unless an alarm duty-cycle cap is
imposed. Both traps are avoided by the swept event-level curve plus the duty cap.

Note: the unsupervised detectors' per-tier results (17/17 value, 0/12 contextual) come from the
window-F1 threshold; at a false-alarm budget they detect almost nothing, which is itself the
argument for learned models.

Table 2. Detectability tiers (payload-novelty detector, pooled).

| tier | example | events | detected |
|---|---|---|---|
| value | max speedometer (byte 5 → 0xFF), correlated signal, coolant, fuzzing | 17 | 17 |
| contextual | reverse light on/off (normal value, wrong given gear/speed) | 12 | 0 |

Ablations: (a) no graph → −11.4 events at 30 alarms/h and 12× the seed spread; (b) no payload features → −6.7 events (single seed);
(c) exact vs smoothed byte profile → 84% vs 33% of normal windows flagged.

## 5b. Latency, alarm rule and ECU cost

The alarm rule, not the model, sets the latency floor (k windows x 100 ms). ConvGRU+payload at
5 alarms/h, 5 seeds:

| rule | events detected | median latency | note |
|---|---|---|---|
| 1-of-1 | 16.2 ± 13.2 (27,27,27,0,0) | 0 ms (≤ 100 ms) | no feasible threshold for 2 seeds |
| 2-of-3 | 21.2 ± 10.6 (26,27,27,26,0) | 112 ms | 1 seed infeasible |
| 3-of-5 | **26.2 ± 1.0** | 220 ms | stable |

So the "< 100 ms detection latency" target is reachable only with an unsmoothed alarm that is
unstable across seeds; the stable configuration costs ~200–300 ms. (Our earlier 0 ms medians came
from a raw single-window threshold, not from the reported alarm rule.)

Deployment cost, measured on one CPU thread at batch 1 (`scripts/ecu_cost.py`):

| model | params | int8 | median / p99 latency | required rate | duty on one core |
|---|---|---|---|---|---|
| ConvGRU + payload | 98 k | 0.10 MB | 1.35 / 1.69 ms | 150/s | 20% |
| STTF graph | 1.05 M | 1.05 MB | 38.9 / 70.1 ms | 10/s | 39% |
| STTF + 50× MC dropout (as proposed) | 1.05 M | 1.05 MB | 1945 ms | 10/s | **1945% (~20 cores)** |

Feature extraction costs 0.3% of a core over real traffic. A laptop core is several times faster
than a typical automotive Cortex-A/R core, so these are optimistic bounds; the duty column scales.
The 50-pass MC-dropout uncertainty quantification specified in the original proposal is infeasible
by roughly 20× on a single core — 5 passes or a deep-ensemble-free calibration method is needed.


## 5c. Cross-vehicle generalisation (can-train-and-test-v1.5, set_01)

Train on the Chevrolet Impala, test on the Chevrolet Silverado, using the dataset's own
known/unknown vehicle split. ConvGRU, 3-of-5 alarm rule, with and without payload-novelty features
(results indistinguishable):

| test set | events | FA/h | alarm duty | median latency |
|---|---|---|---|---|
| known vehicle, known attacks | 7/8 | 0–22 | 0.7% | 200 ms |
| unknown vehicle, threshold from the known vehicle | 8/8 | 21.9 | **99.9%** | 0 ms |
| unknown vehicle, threshold from its own clean traffic | **0/8** | 0.0 | 0.0% | — |
| masquerade attacks (known vehicle) | 6–8/8 | 0–28 | 0.0% | 450–1000 ms |

**Detection does not transfer between vehicles.** With the source vehicle's threshold the alarm is
active 99.9% of the time on the target vehicle — "8/8 detected" means "everything flagged". When the
threshold must meet a false-alarm budget on the target vehicle's own attack-free traffic, detection
falls to 0/8. Dropping the payload-novelty features does not change this, so the cause is not
feature novelty but the vehicle-specific ID vocabulary and traffic patterns the model learns.
Practical consequence: an in-vehicle IDS of this family needs per-vehicle calibration (at minimum)
or per-vehicle training; published cross-vehicle numbers that do not report alarm duty or a
false-alarm budget should be read with this in mind.

This also gives the fifth evaluation trap: the duty cap must be applied to reported test metrics,
not only when selecting the threshold.


## 5d. Does per-vehicle calibration rescue transfer? (no)

We gave the Impala-trained detector N minutes of the target vehicle's *attack-free* traffic — the
data a workshop or fleet operator could realistically record — adding its IDs to the vocabulary and
its frames as negatives, calibrating the alarm threshold on a held-out 20%, and evaluating on target
attack files never used for adaptation (two-way file split, all 8 target events).

| adaptation traffic | events detected with the false-alarm budget honoured on test |
|---|---|
| 0 min | 0/8 |
| 1 min | 0/8 (raw 4/8 at 175–391 FA/h — budget violated) |
| 5 min | 0/8 @5/h; 1/8 @30/h |
| 15 min | 0/8 (raw 3/8 at 249–376 FA/h — violated) |

Detection does not improve with more adaptation traffic, so this is not a data-volume problem. The
mechanism is that the target vehicle's ID embeddings are learned exclusively from normal traffic
(the target has 98 arbitration IDs against the source's 52), so the classifier has no evidence of
what an attack looks like on any of them. **Unlabelled normal traffic from a new vehicle is not
enough.** Either per-vehicle training with labelled attacks is required, or the representation must
drop vehicle-specific ID identity in favour of ID-agnostic statistics — which §5e shows recovers
most of the gap.

Caveats: one vehicle pair, 8 attack events, one architecture, adaptation budgets up to 15 min.
The ID-agnostic pipeline (§5e) excludes windows inside the 30 s warm-up, so attacks beginning in the
first 30 s of a capture are dropped; event denominators therefore differ between the two
representations on some subsets (e.g. known-vehicle set_04: 12 vs 14).


## 5e. ID-agnostic features restore cross-vehicle detection (the positive result)

If vehicle-specific ID identity is what blocks transfer (§5c–d), removing it should help. We
describe each frame only *relative to its own arbitration ID*: cadence ratio against that ID's
warm-up median inter-arrival, payload bit/byte change rates, excursion beyond that ID's warm-up
byte range, per-ID and bus-wide rate ratios, DLC change — 12 features, no ID embedding and no
cross-vehicle vocabulary. Per-ID references come from a 30 s warm-up prefix of each capture, so
adapting to a new vehicle needs no labels and no retraining.

ConvGRU (same architecture, no embedding), 3 seeds, threshold calibrated on the target vehicle's
own attack-free traffic:

| test set | ID-based | **ID-agnostic** | FA/h | median latency |
|---|---|---|---|---|
| known vehicle, known attacks | 7/8 | 7.0 ± 0.0 / 8 | 1.2 | 200 ms |
| **unknown vehicle**, known attacks | **0/8** | **6.3 ± 1.7 / 8** (7.0 ± 0.8 at 30 FA/h) | 0.0 | 450 ms |
| masquerade (known vehicle) | 6–8/8 | 0.0–0.7 / 7 | 0–14 | — |

Replicated on all four vehicle pairs in can-train-and-test (unknown-vehicle events detected at
≤ 5 false alarms/h, threshold from the target's own clean traffic, 3 seeds):

| pair | ID-based | ID-agnostic |
|---|---|---|
| Impala → Silverado | 0/8 | 6.3 ± 1.7 / 8 |
| Traverse → Forester (cross-manufacturer) | 0/12 | **12.0 ± 0.0 / 12** |
| Silverado → Forester (cross-manufacturer) | 0/10 | **10.0 ± 0.0 / 10** |
| Forester → Traverse (cross-manufacturer) | 0/14 | 6.0 ± 0.0 / 14 |
| **total** | **0/44** | **34.3/44 (78%)** |

**The vehicle gap is a representation problem, not an inherent limit.** Dropping ID identity
recovers 78% of target-vehicle attack events (0/44 → 34.3/44) at a few false alarms per hour,
across four pairs including three cross-manufacturer ones, without a single label from the target
vehicle. The price is the
contextual tier: masquerade attacks stay weak (0–4.7 of 4–8 per pair, and the ID-based model beats
it on two pairs with 7/8 and 5/8), because their timing is normal and their substituted payloads
stay inside the warm-up range. This mirrors the ROAD reverse-light
result and motivates the paired design evaluated in §5f — ID-relative features for
transferable coverage of rate/value attacks, plus a vehicle-specific model for contextual ones.

Calibrating the threshold on the source vehicle's clean traffic still fails (22–50% alarm duty), so
per-vehicle threshold calibration remains necessary even with transferable features.


## 5f. Paired detector: the two representations are complementary

The two blind spots are disjoint — ID-relative features transfer between vehicles but miss
masquerade attacks, ID-based features catch masquerade on their own vehicle but transfer to nothing
— so we run both and alarm if either branch fires, giving each branch half the false-alarm budget.
Evaluated on the known vehicle, where both branches are valid, pooled over 4 vehicle pairs × 2 seeds:

| test | budget | A (ID-agnostic) | B (ID-based + payload) | **A OR B** | union FA/h |
|---|---|---|---|---|---|
| known attacks (84 events) | 5/h | 88% | 80% | **92%** | 0.4 |
| known attacks | 30/h | 90% | 92% | **93%** | 19.3 |
| masquerade (52 events) | 5/h | 29% | 58% | **63%** | 0.0 |
| masquerade | 30/h | 35% | 65% | **69%** | 12.5 |

The union beats the better branch at every operating point *and* fires less often than either branch
alone, because each runs at half the budget — it is strictly better, not a trade-off. Operationally
it also degrades gracefully: on a vehicle with no labelled attacks only branch A is available (78% of
events across four pairs, §5e), and branch B is added once that vehicle has labelled data, raising
masquerade coverage from ~29% to ~63%. Masquerade detection at 63–69% remains the main open problem,
consistent with the contextual tier being the hardest case throughout this study.

Cost: two 98 k-parameter models, i.e. 0.20 MB int8 and ~40% of one CPU core at the rates in §5b.


## 5g. Two transferable detectors, OR-ed: the proposed design

§5e–f leave one gap: the transferable representation misses masquerade attacks, and the branch that
catches them (ID-based) does not transfer. We therefore re-implemented the published per-ID
behavioural residualisation representation (14 window statistics per arbitration ID, z-scored
against that ID's warm-up baseline) and evaluated it under the same protocol. It also transfers —
30.4/44 unknown-vehicle events (69%) against 34.3/44 (78%) for our per-frame ratios — and it is
markedly better on masquerade (40% vs 18%). The two ID-agnostic representations are therefore
complementary, and both transfer.

Concatenating them into one 26-feature input *fails*: unknown-vehicle detection falls to 53% at
5 false alarms/h (60% at 30/h), and on one pair the combined model collapses to 0/10 because its
scores on the target vehicle's clean traffic saturate near 1.0 so no threshold meets the budget
(with the source vehicle's threshold it still finds 10/10 — discrimination survives, calibration
does not). Complementary representations must be combined as separate detectors.

Running them as two branches with OR-ed alarms, each given half the budget and calibrated on the
target vehicle's own attack-free traffic (no target labels), pooled over 4 vehicle pairs × 2 seeds:

| branch | unknown vehicle @5 FA/h | FA/h | @30 FA/h | masquerade @5/h | @30/h |
|---|---|---|---|---|---|
| A: ID-relative (ours) | 67% (59/88) | 1.2 | 70% | 19% | 31% |
| B: per-ID residualisation | 69% (61/88) | 1.3 | 72% | 37% | 48% |
| **A OR B** | **81% (71/88)** | **0.3** | **83%** | **38%** | **52%** |

The union gains 12–14 points over either transferable branch *at a lower false-alarm rate*, and
doubles masquerade coverage relative to ID-relative features alone — on a vehicle never seen in
training, with no labels from it, using two 98 k-parameter models (0.20 MB int8, ~40% of one core).
This is the design we propose: two ID-agnostic representations, separate detectors, OR-ed alarms,
per-vehicle label-free threshold calibration.

Numbers are not comparable across our scripts (the paired script scores only windows both branches
cover and uses 2 seeds); branch comparisons within a table are.


## 6. Car-Hacking as a negative result (Section V.x)

300 attacks per file, 3.0 s gaps (CV 0.001–0.003); detection AUC 1.000 for all variants; a
1-feature timer reaches pre-onset F1 0.993 at 1 s while STTF reaches 0.591. Two dataset artefacts
that affect anyone using it: a 2540 s recording gap inside `Fuzzy_dataset.csv`, and the same
481.6 s attack-free recording (891,068 identical frames) appended to all four attack files.

## 7. Limitations (Section VI, revised)

* ROAD results are 15 attack events on one vehicle. The cross-vehicle test (§5c) is on a different
  dataset and vehicle pair, and shows the models do not transfer; we therefore make no
  vehicle-agnostic claims.
* Latency is quantised to the 100 ms window; sub-100 ms claims need finer windows.
* Seed variance is the dominant source of uncertainty at budgets below ~10 alarms/h for models
  without payload-novelty features (STTF graph: ±7.8 events; ConvGRU: ±4.1). With those features a
  single model is stable (±1.0), so no ensemble — and no extra ECU inference cost — is needed.
* The coolant attack (43 injected frames at ~10 Hz) and its masquerade copy are detected by no
  model at any usable budget: the 27/29 ceiling is a property of the data, not the models.
* No onset forecasting is demonstrated, and we argue no public CAN dataset can demonstrate it:
  attacks begin abruptly, so "forecasting" reduces to learning capture protocol or schedule.
* LZTO/GADT deferred: an 8B LLM cannot meet a <100 ms enforcement budget on vehicle hardware;
  Dilithium (ML-DSA, FIPS 204) signatures are ~2.4 kB against an 8-byte CAN frame; and the
  proposed simulator stack does not model the CAN bus.

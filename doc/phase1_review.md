# Phase 1 review — EDA & sensor analysis
 
Source of truth: `doc/plan.md` (Phase 1 section) and `doc/requirements_spec.md`. Reviewed
artifact: `notebook/phase1_eda_sensor_analysis.ipynb`.
 
---
 
## 1. TO DO — Phase 1 activities per plan.md
 
| # | Activity | Spec detail |
|---|---|---|
| 1 | Trajectory visualization | Plot all 21 sensors over cycle for 5–10 randomly sampled training engines. Identify which sensors trend monotonically toward failure vs. show no systematic pattern. |
| 2 | Near-constant sensor identification | Compute std of each sensor across all training data. Expected near-constant set for FD001: sensors 1, 5, 6, 10, 16, 18, 19. Confirm from data, not assumption. |
| 3 | Correlation with RUL | Pearson correlation of each sensor vs. raw (uncapped) linear RUL, pooled across engines. Rank by \|r\|. Cross-check ranking against trajectory plots since some sensors degrade non-monotonically. |
| 4 | ACF/PACF on a stationary version of the signal | (a) Confirm non-stationarity first: ADF/KPSS on 2–3 example engines for a top sensor, expect both to agree "non-stationary." (b) First-difference within engine, re-test, expect both to flip to "stationary." (c) Compute ACF/PACF on the differenced series for the top 3–4 sensors. (d) Interpret cautiously — differenced ACF/PACF describes noise memory, treat as a consistency check on Phase 3 rolling-window size, not the sole basis for it. |
| 5 | Operating condition check | Verify `op_1`, `op_2`, `op_3` are effectively constant in FD001 (single operating condition, no normalization needed — unlike FD002/FD004). |
 
## 2. Deliverables — Phase 1 (plan.md)
 
> "A ranked sensor list with correlation values, a confirmed exclusion set, ADF/KPSS results
> confirming raw-sensor non-stationarity, and ACF/PACF plots computed on the differenced series
> for the top sensors. These directly inform Phase 3 design decisions."
 
Unpacked, that's four concrete artifacts:
 
1. Ranked sensor list with Pearson `r` values.
2. Confirmed near-constant exclusion set.
3. ADF/KPSS results demonstrating raw sensors are non-stationary.
4. ACF/PACF plots on the differenced series for the top sensors.
---
 
## 3. Summary of work done (notebook)
 
The notebook (`phase1_eda_sensor_analysis.ipynb`) covers all five activities, generally at more
rigor than the plan's minimum:
 
- **Trajectory visualization** — 3 independent random samples (seeds 42/7/123) of 8 engines each,
  all 21 sensors, plotted against **cycles-before-failure** (not raw cycle) so every engine's
  endpoint aligns at failure. Using 3 samples instead of 1 checks that a pattern isn't a lucky
  draw. This surfaced an unplanned finding: `sensor_9` and `sensor_14` don't have a consistent
  trend direction across engines (some rise toward failure, some fall) — quantified in a follow-up
  section (1b) by fitting a per-engine linear slope and counting sign: 71/29 and 60/40 rising/
  falling splits respectively.
- **Near-constant sensor identification** — std of each sensor over all training rows, log-scale
  bar chart, threshold at `std < 0.01`. Result exactly matches the plan's expected set: sensors 1,
  5, 6, 10, 16, 18, 19 excluded, 14 sensors kept. A ~27x gap at the threshold boundary is reported
  as evidence the cut isn't sensitive to its exact value.
- **Correlation with RUL** — pooled Pearson `r` vs. raw RUL for all 14 informative sensors, ranked
  by \|r\|. Top 4 (`sensor_11`, `sensor_4`, `sensor_12`, `sensor_7`) match the plan's stated
  expectation. Goes beyond the spec's "cross-check against trajectory plots" instruction by adding
  a **per-engine correlation consistency check** (Pearson `r` computed independently within each
  of the 100 engines, plotted as a distribution against the pooled value) — this is what actually
  caught and quantified the `sensor_9`/`sensor_14` sign-flip issue with a second, independent
  method, cross-confirming the slope-based finding from Section 1b.
- **ACF/PACF** — full sequence per the plan: (a) ADF/KPSS on raw `sensor_11` for 3 engines, both
  agree non-stationary; (b) first-differenced, re-tested, both flip to stationary; extended beyond
  the plan's "a top sensor" (singular) to also confirm `sensor_4`, `sensor_12`, `sensor_7` are
  stationary once differenced (with `sensor_12` correctly flagged as a borderline KPSS case rather
  than swept under the rug); (c) ACF/PACF plotted for all 4 top sensors × 3 engines; (d) result
  interpreted as an MA(1)-like signature (sharp lag-1 ACF cutoff, gradually decaying PACF) — i.e.
  no memory beyond 1 cycle — with an explicit caution that this doesn't by itself dictate Phase 3
  window size, consistent with the plan's Activity 4 guidance.
- **Operating condition check** — std/min/max/nunique for `op_1`–`op_3` reported on **both** train
  and test (plan only asks to verify FD001 in general). `op_3` is exactly constant; `op_1`/`op_2`
  have many unique values but negligible range, correctly interpreted as measurement noise around
  one operating point rather than the multi-regime behavior expected in FD002/FD004.
## 4. Mapping: done work → deliverables
 
| Deliverable required | Status | Evidence in notebook | Notes |
|---|---|---|---|
| Ranked sensor list with correlation values | ✅ Met | Section 3, cell `5f7c0c75` (pooled `pearson_r`/`abs_r` table) + Section 3b consistency table | Also includes per-engine consistency (`std_engine_r`, `pct_sign_match`) not required by the plan but directly useful for Phase 3 feature design. |
| Confirmed near-constant exclusion set | ✅ Met | Section 2, cell `d2a87c08` | Matches plan's expected set exactly: `{1, 5, 6, 10, 16, 18, 19}`. |
| ADF/KPSS results confirming raw-sensor non-stationarity | ✅ Met | Section 4a, cell `ab16300a` | Done for `sensor_11` (top-ranked) on 3 engines, both tests agree non-stationary — matches plan's minimum ("2–3 example engines for a top sensor"). |
| ACF/PACF plots on the differenced series for the top sensors | ✅ Met | Section 4c, cell `d3d6258a` (+ quantified summary in `746075f7`) | Covers top 4 sensors (plan asks for top 3–4), 3 engines each; includes both the plots and a numeric significance summary. |
 
All four required deliverables are present. The plan's five activities are each represented by a
dedicated, clearly-labeled section, and the notebook closes with an explicit "Phase 1 complete"
checkpoint cell summarizing what's carried forward (`INFORMATIVE_SENSORS`,
`NEAR_CONSTANT_SENSORS`, `pooled_corr_ranked`, plus the consistency/stationarity findings).
 
### Where the notebook exceeds the plan
 
- 3 independent trajectory samples instead of 1 (robustness check on visual conclusions).
- Per-engine correlation consistency analysis (Section 3b) — not explicitly requested, but
  directly operationalizes the plan's own warning that "some CMAPSS sensors degrade
  non-monotonically."
- Stationarity confirmed for 4 sensors instead of 1 before proceeding to ACF/PACF.
- Operating-condition check reported for both train and test splits, not just train.
### Open items to carry into Phase 3
 
These are flagged in the notebook itself and should inform Phase 3 feature engineering:
 
1. **`sensor_9` / `sensor_14` sign inconsistency** (Sections 1b, 3b) — per-engine trend direction
   splits ~70/30 and ~60/40. A single pooled-sign rolling feature will misrepresent the minority
   engines; the notebook suggests a sign-corrected or per-engine-normalized feature (e.g.
   `abs(slope)`-weighted) as an alternative. Since these two sensors are not in the "top 8 by
   |r|" that Phase 3 uses for rolling stats, this mainly matters only if they're revisited later
   (e.g. for the lag features, if selection criteria change) — worth a explicit decision (include
   with a correction, or exclude) rather than defaulting silently.
2. **`sensor_8` / `sensor_13` near-duplication** (Sections 2, 3b) — near-identical mean/std and
   100% sign agreement but ~3–4x noisier per-engine correlation than the top 10. Flagged as
   redundant with each other; neither is in the top-8 set Phase 3 will use, so no immediate
   action, but worth remembering if the sensor-selection cutoff is ever revisited.
3. **Rolling-window size** — the ACF/PACF "no memory beyond 1 cycle" finding is explicitly
   scoped as a consistency check, not a source for picking window size directly. Phase 3's
   candidate window set `{5, 10, 20, 30}` should still be chosen against how much trend each
   window preserves (per Section 4's closing note), not the noise-memory result alone.
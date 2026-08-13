# Phase 3 review — feature engineering
 
Source of truth: `doc/plan.md` (Phase 3 section) and `doc/requirements_spec.md`. Reviewed
artifact: `notebook/phase3_feature_engineering.ipynb`.
 
---
 
## 1. TO DO — Phase 3 activities per plan.md
 
| # | Activity | Spec detail |
|---|---|---|
| 1 | Sensor selection | Use the **top 8 sensors by \|correlation\| with RUL** from Phase 1 for rolling statistics — not the full ~14 non-near-constant set. Drop the 7 near-constant sensors. Exclude `engine_id` from the feature set entirely (design decision #6). |
| 2 | Rolling statistics | For each selected sensor × window size, compute rolling mean/std/min/max, trailing-only (no look-ahead). Candidate window set `{5, 10, 20, 30}`, but provisional — revisit against Phase 1's differenced-series ACF/PACF and trajectory-smoothness findings before finalizing. `min_periods=1` so early cycles still produce a value. |
| 3 | Lag features | For the top 3–4 sensors by correlation, add lag-1, lag-5, lag-10. Use `groupby('engine_id').shift(k)` so lags never cross an engine boundary. |
| 4 | Cycle as a feature | Include raw `cycle` — an engine at cycle 150 is categorically different from one at cycle 30. |
| 5 | Construct test features | Apply identical transforms to `test_df`; each test engine's stats from its own history only. Keep only the last row per engine as the model input vector. |
| 6 | Drop NaN rows | Drop rows with NaN lag/rolling features from `X_train` (early-cycle warm-up). Not expected to affect `X_test` since only the last row per engine is kept. |
 
## 2. Deliverables — Phase 3 (plan.md)
 
> "`X_train` feature matrix with `y_train` target vector. `X_test` feature matrix (100 rows, one
> per test engine). Column count expected: 8 sensors × 4 windows × 4 stats = 128 rolling features,
> + 9–12 lag features (top 3–4 sensors × 3 lags), + 1 cycle feature ≈ **138–141 features total** —
> recompute this if the candidate window set from Activity 2 changes."
 
Unpacked, three concrete artifacts plus a reconciled feature count:
 
1. `X_train` (feature matrix) and `y_train` (capped RUL target).
2. `X_test` — exactly 100 rows, one per test engine, at each engine's own last observed cycle.
3. Feature count consistent with the sensor/window/lag choices actually made.
---
 
## 3. Summary of work done (notebook)
 
The notebook (`phase3_feature_engineering.ipynb`) covers all six plan activities, plus three
redundancy/collinearity checks (pairwise correlation matrix, hierarchical clustering, VIF) that
the plan doesn't explicitly request but that directly inform Activity 1's cutoff decision:
 
- **Sensor selection (Activity 1) — deliberate, documented deviation from the plan default.**
  The plan specifies "top 8." The notebook recomputes the Phase 1 pooled-correlation ranking fresh
  and finds the gap in `|r|` from rank 12 (`sensor_13`, 0.563) to rank 13 (`sensor_9`, 0.390) is
  0.172 — about 7.5× the next-largest adjacent-rank gap anywhere in the list (0.023). It uses this
  elbow to justify **top 12** instead of top 8, and cross-checks the cut against Phase 1's
  independent finding that `sensor_9`/`sensor_14` (ranks 13–14) have inconsistent per-engine
  correlation sign — two independent signals agreeing on where the reliable-sensor tier ends. Near-
  constant exclusion (`std < 0.01`) reproduces Phase 1's set exactly: 7 sensors dropped
  (`sensor_1, 5, 6, 10, 16, 18, 19`).
- **Redundancy checks beyond the plan's Activity 1** — three methods applied to the 12 selected
  sensors before building rolling features from them:
  - Pairwise Pearson correlation matrix: every one of the 66 pairs has `|r| ≥ 0.60`; standout
    pairs `sensor_11`/`sensor_12` (0.85) and `sensor_8`/`sensor_13` (0.83, matching a Phase 1 flag).
  - Hierarchical clustering (average linkage, distance = `1 − |r|`): confirms the same two pairs
    merge first; at threshold `|r| > 0.8` the 12 sensors form 8 groups (a 4-sensor cluster of the
    top-ranked sensors, a 2-sensor cluster, 6 singletons).
  - VIF (multivariate collinearity): only `sensor_11` (5.91) and `sensor_12` (5.26) exceed the
    moderate-concern threshold (`>5`); none exceeds the severe threshold (`>10`) — correctly
    interpreted as "safe to use together for a linear model," a different question from the
    pairwise checks, and not a reason to drop any sensor from the tree-based Phase 5 model.
- **Rolling statistics (Activity 2)** — mean/std/min/max over `{5, 10, 20, 30}`, `min_periods=1`,
  computed via `groupby('engine_id').rolling(...)`. The window set is kept as-is per the plan's own
  "provisional, revisit before finalizing" instruction: the notebook notes Phase 1's differenced-
  series ACF/PACF found no autocorrelation past lag 1 (no noise structure to size a window
  around), so window choice is left on trend-preservation grounds rather than revised. Explicit
  correctness notes: `min_periods=1` produces `NaN` `std` only at each engine's first cycle
  (expected, handled with lag NaNs later); sorting by `(engine_id, cycle)` before rolling removes
  reliance on the raw files already being cycle-ordered.
- **Lag features (Activity 3)** — top 4 sensors (`sensor_11, sensor_4, sensor_12, sensor_7`, the
  same set Phase 1 used for ACF/PACF) × lags `{1, 5, 10}`, via `groupby('engine_id').shift(k)`.
- **Cycle feature (Activity 4)** — no transform needed; `cycle` is added directly to `FEATURE_COLS`.
- **Test features (Activity 5)** — identical rolling/lag transforms applied to `test_df`. Notable
  deviation from the plan's snippet: the plan uses `groupby('engine_id').last()`, which the
  notebook flags as a latent bug — `.last()` aggregates per-column independently and takes the
  last **non-NaN** value in each column, so a single NaN on an engine's true final row would
  silently pull that column from an earlier cycle while the rest come from the true last cycle,
  mixing two points in time. The notebook instead sorts by `(engine_id, cycle)` and takes
  `.tail(1)`, which is correct by construction. It also verifies the minimum last-observed test
  cycle (31) comfortably exceeds the 10-cycle warm-up `lag_10` needs, confirming no test engine is
  short enough to produce a NaN feature at its own truncation point.
- **Assembly + NaN drop (Activity 6)** — `y_train` recomputed as `min(max_cycle - cycle, 125)`
  matching Phase 2's cap convention. 205 features assembled (192 rolling + 12 lag + 1 cycle); 1,000
  / 20,631 training rows (4.85%) dropped for NaN, confirmed to be exactly the first 10 cycles of
  every one of the 100 engines (bounded by `lag_10`, the last feature to become defined, at cycle
  11) — not a scattered pattern. `engine_id` is asserted absent from `FEATURE_COLS` (design
  decision #6). Zero remaining NaN in `X_train` or `X_test`, asserted.
## 4. Mapping: done work → deliverables
 
| Deliverable required | Status | Evidence in notebook | Notes |
|---|---|---|---|
| `X_train` + `y_train` | ✅ Met | Section 9, cell `bff3fa5a` | `X_train` (19,631, 205), `y_train` (19,631,) — capped-RUL target, NaN rows dropped. |
| `X_test` — 100 rows, one per test engine, last observed cycle | ✅ Met | Section 8 (`24bcce1b`) + Section 9 | `X_test_full` built via sorted `.tail(1)`, not the plan's `.last()` — a correctness improvement, not a deviation from intent. Shape confirmed (100, 205). |
| Feature count reconciled with sensor/window/lag choices | ✅ Met, but count differs from plan | Intro cell + Section 9 (`bff3fa5a`) | 205 features (192 rolling + 12 lag + 1 cycle), vs. the plan's 138–141 — expected once the sensor set changes from 8 to 12; the notebook explicitly recomputes and states this rather than leaving the mismatch unexplained. |
 
All three required deliverables are present and correct. Every one of the plan's six numbered
activities has a dedicated section, and the notebook closes with an explicit "Phase 3 complete"
checkpoint identifying `X_train`, `y_train`, `X_test` as ready for Phase 4/5.
 
### Where the notebook deviates from or exceeds the plan
 
- **Top 12 sensors instead of top 8** (Activity 1) — the one substantive deviation from a plan
  default. Justified with a quantified gap analysis (7.5× larger than any other adjacent-rank gap)
  and cross-checked against an independent Phase 1 finding, so this reads as a considered call
  rather than scope creep — but it's worth confirming this tradeoff (205 vs. ~140 features on
  ~19.6k rows) is intentional before Phase 5, since more rolling/lag features from highly
  correlated sensors (Section 2: all 66 pairs `|r| ≥ 0.60`) raises overfitting risk for a
  boosted-tree model without necessarily adding independent signal — LightGBM's feature-importance
  chart (Phase 5, plan Activity 4) is the natural place to check whether the extra 4 sensors'
  features actually earn their keep.
- **Three redundancy/collinearity checks** (correlation matrix, dendrogram, VIF) not requested by
  the plan's Phase 3 spec, added specifically to justify and stress-test the top-8→top-12 decision.
- **`.tail(1)` instead of `.last()`** for extracting each test engine's final row — fixes a subtle
  correctness gap in the plan's own snippet (silent column-level time-mixing on partial NaN rows),
  verified empirically to not have mattered for this dataset but correct independent of that check.
### Open items to carry into Phase 4/5
 
1. **Feature-to-row ratio.** 205 features against 19,631 training rows (~96 rows/feature) is not
   alarming on its own for LightGBM, but is materially higher-dimensional than the plan's 138–141
   baseline. Worth watching CV RMSE (Phase 5) for signs of overfitting relative to what the top-8
   set would have given — the notebook doesn't build the top-8 variant as a comparison point, so
   this is untested rather than confirmed safe.
2. **`sensor_11`/`sensor_12` and `sensor_8`/`sensor_13` redundancy** (Sections 2–4) — flagged
   repeatedly (highest pairwise `r`, first dendrogram merges, highest VIF) but no feature was
   dropped as a result, which is a reasonable call for a tree-based model (LightGBM handles
   correlated features natively via split selection) but not for the Phase 7 stretch CNN if that's
   attempted later — normalization there doesn't remove redundant channels the way tree splits do.
3. **Window set carried over unchanged from the plan's candidate `{5, 10, 20, 30}`** — the
   notebook's stated reason (no ACF/PACF memory structure past lag 1 to size a window around) is a
   process match to the plan's Activity 2 instruction, but it's a decision to *not* revisit, not
   independent validation that these four window sizes are optimal. Nothing to fix here, just worth
   remembering if Phase 5/6 diagnostics point to a specific window size underperforming.
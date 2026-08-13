# Phase 4 review — baseline models
 
Reviews [notebook/phase4_baseline.ipynb](../notebook/phase4_baseline.ipynb) against the Phase 4
spec in [plan.md](plan.md#phase-4--baseline-models) and the evaluation requirements in
[requirements_spec.md §4](requirements_spec.md).
 
---
 
## 1. TO DO — Phase 4 activities per plan.md
 
| # | Activity | Spec detail |
|---|---|---|
| 1 | Implement the NASA scoring function | `nasa_score(y_true, y_pred)`: `d = pred - true`; `exp(-d/13) - 1` for `d < 0`, `exp(d/10) - 1` for `d >= 0`. Must exist before any model evaluation. Verify on a toy example that `d = -10` scores lower than `d = +10`. |
| 2 | Naive baseline | `RUL_pred = mean(max_cycle over training engines) - last_cycle_of_test_engine`, clipped to ≥ 0. Report RMSE and NASA score. |
| 3 | Linear degradation baseline | Fit `RUL (capped) ~ cycle` per training engine on that engine's own cycles; predict test engines at their last cycle using the average slope from training engines. Report RMSE and NASA score. |
| 4 | Record results | Comparison table (Naive, Linear degradation) — to be extended in Phase 6 with LightGBM. |
 
## 2. Deliverables — Phase 4 (plan.md)
 
> "Implemented `nasa_score` function (to be reused in every subsequent phase). Baseline RMSE and
> NASA score recorded. Comparison table started."
 
Three concrete artifacts:
 
1. `nasa_score` (and, implicitly, `rmse`) — the metric functions every later phase reuses unchanged.
2. RMSE + NASA score for both the naive and linear-degradation baselines.
3. A comparison table (`Model | RMSE | NASA score`) seeded for Phase 6.
---
 
## 3. Summary of work done (notebook)
 
The notebook covers all four activities in order, with each metric double-checked before use:
 
- **Data reload (Section 0)** — `train_df`, `test_df`, `rul_true` reloaded fresh, matching the
  "reload rather than depend on prior notebook state" convention from Phases 1–3.
- **Capped RUL label (Section 1)** — recomputes `rul = min(max_cycle - cycle, 125)` on `train_df`,
  identical to the Phase 2 label, needed here to fit the linear-degradation baseline.
- **NASA scoring function (Section 2)** — `nasa_score` implemented exactly to spec (`d = pred -
  true`, `np.where(d < 0, exp(-d/13)-1, exp(d/10)-1)`, summed). A `rmse` helper is added alongside
  it (not required by the plan, but needed immediately and reused identically to `nasa_score` in
  every later section). The toy sanity check uses `d = -10` vs. `d = +10` on a single point exactly
  as the plan specifies, asserts `score_early < score_late`, and prints the confirmation — this
  matches Activity 1's verification step precisely.
- **Naive baseline (Section 3)** — `mean_train_max_cycle - test_last_cycle`, clipped to ≥ 0,
  reindexed to align with `rul_true`'s engine order before scoring. Result: **RMSE 40.20, NASA
  score 25527.83**.
- **Linear degradation baseline (Section 4)** — per-engine `np.polyfit(cycle, rul, 1)` for all 100
  training engines, then slope and intercept both averaged and applied to each test engine's last
  cycle, clipped to ≥ 0. Result: **RMSE 36.30, NASA score 17234.08**.
- **Comparison table + chart (Section 5)** — a two-row `Model | RMSE | NASA score` table as
  specified, plus a bar-chart comparison (not required by the plan, but a reasonable presentation
  add-on that will need a third bar in Phase 6).
- **Written summary (Section 6)** — states both results, notes the linear baseline lands close to
  the spec's ~35 RMSE floor using only `cycle` (no sensor data), and observes the NASA-score gap
  between the two baselines is proportionally larger than the RMSE gap — attributed to the
  exponential penalty being more sensitive to the naive baseline's larger optimistic misses.
### Interpretation choice worth flagging
 
Activity 3's wording — "fit ... using that engine's own training cycles ... predict ... using the
*average slope* from training engines" — is ambiguous about the intercept, since it only names the
slope explicitly. The notebook documents its own resolution in the `p4-act3-md` cell: test engines
have no RUL labels to fit against, so *both* slope and intercept are averaged across the 100
training-engine fits and applied as one shared line. This is a sensible, clearly-justified reading
(there's no other source for an intercept), and it's called out explicitly rather than silently
assumed — but it's worth noting explicitly here since a different reader could parse the plan text
as "slope only, per-engine intercept," which is not implementable without test-set RUL. No action
needed; flagging for awareness before Phase 6 extends this table.
 
## 4. Mapping: done work → deliverables
 
| Deliverable required | Status | Evidence in notebook | Notes |
|---|---|---|---|
| `nasa_score` function, verified on a toy example | ✅ Met | Section 2, cells `p4-nasa-score`, `p4-nasa-toy` | Matches the plan's formula exactly; toy check passes the specified `d=-10` vs `d=+10` comparison. |
| Naive baseline RMSE + NASA score | ✅ Met | Section 3, cell `p4-naive` | RMSE 40.20, NASA 25527.83; predictions clipped to ≥ 0 per requirements_spec.md §4.3. |
| Linear degradation baseline RMSE + NASA score | ✅ Met | Section 4, cells `p4-linear-fit`, `p4-linear-predict` | RMSE 36.30, NASA 17234.08; clipped to ≥ 0. Intercept-averaging interpretation documented in-notebook (see above). |
| Comparison table started | ✅ Met | Section 5, cell `p4-compare-table` | Two rows (Naive, Linear degradation); ready for a third row in Phase 6. |
 
All four required deliverables are present and correctly executed.
 
### Where the notebook exceeds the plan
 
- Adds a `rmse` helper function alongside `nasa_score`, reused identically for both baselines
  (and presumably every later phase) rather than recomputing RMSE ad hoc each time.
- Bar-chart visualization of the comparison table (RMSE and NASA score side by side), beyond the
  plan's plain-table requirement.
- The written summary (Section 6) doesn't just report numbers — it interprets *why* the NASA-score
  gap is proportionally larger than the RMSE gap (fewer large optimistic misses in the linear
  baseline), connecting back to the asymmetric-penalty rationale from requirements_spec.md §4.2.
### Open items to carry into Phase 5 / Phase 6
 
None block moving on — both deliverables are met — but worth keeping in mind:
 
1. **Linear degradation baseline (36.30) is the number every later model must beat**, per
   requirements_spec.md §4.1's "≈35 cycles" floor — Phase 5's LightGBM CV/test RMSE should be
   compared directly against this 36.30, not the spec's rounded "≈35."
2. **The comparison table and chart will need a third row/bar in Phase 6** for LightGBM (and a
   fourth in Phase 7 if the CNN stretch is attempted) — worth reusing this notebook's `comparison`
   DataFrame structure rather than rebuilding it.
3. **The intercept-averaging interpretation for the linear baseline** (see Section 3 above) is
   reasonable and documented, but since it's an interpretation rather than a literal reading of the
   plan text, it's worth a quick second look if Phase 6's baseline comparison numbers ever look
   off.
---
 
## 5. Verdict
 
**Phase 4 deliverable is met.** `nasa_score` and `rmse` are implemented once, verified against the
spec's asymmetry requirement, and reused unchanged across both baselines. The naive baseline (RMSE
40.20) and linear degradation baseline (RMSE 36.30) are both computed correctly, clipped to ≥ 0,
and recorded in a comparison table ready for Phase 6 extension. The linear baseline's RMSE lands
close to the spec's ~35-cycle floor using only `cycle` as a feature, giving a credible reference
point for judging Phase 5's LightGBM model. Ready to proceed to Phase 5.
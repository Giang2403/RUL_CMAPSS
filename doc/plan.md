# Implementation plan — turbofan RUL prediction
## NASA CMAPSS predictive maintenance project
 
---
 
## Overview
 
| Item | Detail |
|---|---|
| Dataset | NASA CMAPSS FD001 (primary), FD002 / FD004 (stretch) |
| Approach | EDA → feature engineering → baselines → LightGBM → (optional CNN) |
| Total phases | 8 (phases 0–6 core, phase 7 stretch) |
| Estimated core effort | 34 hours |
| Estimated stretch effort | 8 additional hours (42 hours total) |
| Weeks to complete | 2 weeks at 22 hours/week |
 
---
 
## Phase descriptions
 
---
 
### Phase 0 — Setup & data loading
**Estimated effort: 2 hours**
 
**Objective.** Get clean, well-named DataFrames into memory. Everything downstream depends on
this being correct. A wrong column assignment here silently corrupts every subsequent step.
 
**Activities.**
 
1. Download FD001 files from the NASA data repository or Kaggle mirror. Files needed:
   `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`.
2. Parse the space-delimited format, assign column names:
   `['engine_id', 'cycle', 'op_1', 'op_2', 'op_3', 'sensor_1', ..., 'sensor_21']`.
3. Verify integrity: confirm no missing values, check min/max cycle counts per engine,
   confirm 100 training engines and 100 test engines.
4. Load `RUL_FD001.txt` into a Series indexed by engine ID (1-indexed, one value per engine).
5. Print a summary: shape of each DataFrame, dtypes, min/max values for a few key columns.
**Deliverable.** Three clean Python objects — `train_df`, `test_df`, `rul_true` — verified
and ready to use for the rest of the project.
 
---
 
### Phase 1 — EDA & sensor analysis
**Estimated effort: 7 hours**
 
**Objective.** Understand the dataset before building anything. Identify which sensors carry
degradation signal and which are near-constant noise. Characterize the autocorrelation structure
of the informative sensors — correctly, on a stationary version of the signal.
 
**Activities.**
 
1. **Trajectory visualization.** For 5–10 randomly sampled training engines, plot all 21 sensors
   over cycle. This is the most important plot in the entire project — you should be able to see
   which sensors trend monotonically toward failure and which show no systematic pattern.
2. **Near-constant sensor identification.** Compute the standard deviation of each sensor across
   all training data. Sensors with near-zero variance carry no signal. Expected result for FD001:
   sensors 1, 5, 6, 10, 16, 18, 19 are near-constant. Confirm from data rather than assumption.
3. **Correlation with RUL.** Compute the Pearson correlation between each sensor value and the
   corresponding remaining cycles (using the raw linear label for this analysis). Rank sensors
   by absolute correlation. The top-ranked sensors are your primary modeling features. Pearson
   only captures linear association, and some CMAPSS sensors degrade non-monotonically —
   cross-check the ranking against the trajectory plots from Activity 1 rather than trusting the
   correlation number alone.
4. **ACF / PACF — on a stationary version of the signal, not the raw trajectory.** A degrading
   sensor trajectory trends toward failure by construction, which makes it non-stationary.
   Running ACF/PACF directly on raw values (the natural first instinct) mostly measures trend
   persistence, not genuine lag-dependency — every lag will show slow decay regardless of the
   underlying noise process, which is not an informative result. This connects directly to the
   stationarity material already covered in the theory curriculum, and the fix reuses the same
   tool from there (differencing):
   - **Confirm non-stationarity first.** Run ADF/KPSS on 2–3 example engine trajectories for a
     top sensor. Expect ADF to fail to reject its null (non-stationary) and KPSS to reject its
     null (also non-stationary) — agreement on "non-stationary" is the expected result here.
   - **First-difference within engine**, then re-test: `df.groupby('engine_id')[sensor].diff()`.
     Differencing removes the trend, exactly as in the stationarity chapter.
   - **Compute ACF/PACF on the differenced series** for the top 3–4 sensors. This is now a fair
     question to ask of the tool: it characterizes the sensor's short-term noise structure
     (close to white noise vs. genuinely autocorrelated cycle-to-cycle noise), not the
     degradation trend itself.
   - **Interpret cautiously.** The differenced-series ACF/PACF describes noise memory, not trend
     memory — treat it as a consistency check on rolling-window size in Phase 3 (a window
     shouldn't be shorter than the noise's own memory), not as the sole basis for choosing it.
     Cross-check against how visually jagged vs. smooth each sensor's trajectory looked in
     Activity 1.
5. **Operating condition check.** Verify that `op_1`, `op_2`, `op_3` are effectively constant
   for FD001 (single operating condition). This confirms no normalization is needed for this
   sub-dataset (unlike FD002/FD004).
**Deliverable.** A ranked sensor list with correlation values, a confirmed exclusion set,
ADF/KPSS results confirming raw-sensor non-stationarity, and ACF/PACF plots computed on the
differenced series for the top sensors. These directly inform Phase 3 design decisions.
 
---
 
### Phase 2 — Label construction
**Estimated effort: 3 hours**
 
**Objective.** Construct the training target correctly. This is the single most impactful
modeling decision: get it wrong and the model trains on false signal for the first 60–70% of
each engine's life.
 
**Activities.**
 
1. **Linear label.** Add a `rul` column to `train_df`:
   ```python
   max_cycle = train_df.groupby('engine_id')['cycle'].transform('max')
   train_df['rul'] = max_cycle - train_df['cycle']
   ```
 
2. **Apply piecewise-linear cap.** Clip the `rul` column at 125 cycles:
   ```python
   train_df['rul'] = train_df['rul'].clip(upper=125)
   ```
 
3. **Visualize the cap effect.** For 3–4 engines, plot raw RUL (linear countdown) vs. capped
   RUL side by side over cycle. The capped label is flat at 125 for the healthy phase, then
   descends linearly once the engine enters the degradation phase.
4. **Justify the 125 value.** The cap threshold is not arbitrary — it corresponds to the
   approximate point where sensor readings in the training set begin to diverge across engines
   of different health states. Verify this visually: plot mean sensor values (for an informative
   sensor) binned by raw RUL. The signal should be flat above ~125 cycles and trend below it.
**Deliverable.** `train_df` augmented with a capped `rul` column. A visualization showing the
cap effect. A one-paragraph written justification of why the cap is correct.
 
---
 
### Phase 3 — Feature engineering
**Estimated effort: 7 hours**
 
**Objective.** Convert the raw per-cycle sensor readings into a feature matrix where each row
represents the state of one engine at one point in its life, expressed as statistics over its
observable history up to that cycle.
 
**Activities.**
 
1. **Sensor selection.** Use the **top 8 sensors by |correlation| with RUL** from Phase 1 for
   rolling statistics — not the full set of ~14 non-near-constant sensors. Restricting further
   than "everything that isn't near-constant" keeps the feature count controlled (see the
   deliverable count below) and avoids redundant, highly-correlated features from adjacent
   window sizes on marginal sensors. Drop the 7 near-constant sensors (1, 5, 6, 10, 16, 18, 19)
   entirely, and also exclude `engine_id` from the feature set — training and test engine IDs
   both run 1–100 but refer to different physical units, so the model must not be given the
   opportunity to key off raw ID.
2. **Rolling statistics.** For each of the top 8 sensors × each window size, compute rolling
   mean, std, min, and max — always using only cycles strictly before or at the current cycle
   (no look-ahead). Start from the candidate window set {5, 10, 20, 30}, but treat it as
   provisional: revisit it against the differenced-series ACF/PACF and trajectory-smoothness
   findings from Phase 1 Activity 4 before finalizing.
   ```python
   for sensor in informative_sensors:
       for w in [5, 10, 20, 30]:
           train_df[f'{sensor}_mean_{w}'] = (
               train_df.groupby('engine_id')[sensor]
               .transform(lambda x: x.rolling(w, min_periods=1).mean())
           )
   ```
   Note `min_periods=1`: this allows rolling stats to be computed even in the early cycles
   where full window history is not yet available (using whatever is available up to that point).
3. **Lag features.** For the top 3–4 sensors by correlation (Phase 1), add lag-1, lag-5, and
   lag-10 features. Use `groupby('engine_id').shift(k)` to ensure lags do not cross engine
   boundaries.
4. **Add cycle as a feature.** The raw cycle number carries useful information: an engine at
   cycle 150 is categorically different from one at cycle 30. Include it.
5. **Construct test features.** Apply the exact same transformations to `test_df`. Each test
   engine's rolling stats must be computed from its own cycle history only — no cross-engine
   contamination. Since we need the feature vector at the final observed cycle per test engine,
   after computing features keep only the last row per engine:
   ```python
   X_test = test_df.groupby('engine_id').last().reset_index()
   ```
 
6. **Drop rows with NaN.** A few early rows may have NaN lag features (lag-10 at cycle 5, for
   example). Drop these from the training set. They are not present in the test feature matrix
   since we only use the last cycle per engine.
**Deliverable.** `X_train` feature matrix with `y_train` target vector. `X_test` feature
matrix (100 rows, one per test engine). Column count expected: 8 sensors × 4 windows × 4 stats
= 128 rolling features, + 9–12 lag features (top 3–4 sensors × 3 lags), + 1 cycle feature ≈
**138–141 features total** — recompute this if the candidate window set from Activity 2 changes.
 
---
 
### Phase 4 — Baseline models
**Estimated effort: 4 hours**
 
**Objective.** Establish a floor that every subsequent model must beat. Without a baseline, any
RMSE number is uninterpretable — you don't know if you've built something useful or something
that barely beats guessing.
 
**Activities.**
 
1. **Implement the NASA scoring function.** This must be done before any model evaluation:
   ```python
   def nasa_score(y_true, y_pred):
       d = y_pred - y_true
       scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
       return scores.sum()
   ```
   Verify on a toy example: `d = -10` should give a smaller penalty than `d = +10`.
2. **Naive baseline.** For each test engine at its last observed cycle, predict:
   ```
   RUL_pred = mean(max_cycle for all training engines) − last_cycle_of_test_engine
   ```
   Clip to ≥ 0. Evaluate: RMSE and NASA score.
3. **Linear degradation baseline.** For each test engine, fit a simple linear regression of
   RUL (capped) against cycle number using that engine's own training cycles — then predict at
   the test engine's last cycle by assuming the same linear degradation rate. Use the average
   slope from training engines. Evaluate: RMSE and NASA score.
4. **Record results.** Enter these into a comparison table that will be expanded in Phase 6:
   | Model | RMSE | NASA score |
   |---|---|---|
   | Naive | ? | ? |
   | Linear degradation | ? | ? |
**Deliverable.** Implemented `nasa_score` function (to be reused in every subsequent phase).
Baseline RMSE and NASA score recorded. Comparison table started.
 
---
 
### Phase 5 — LightGBM
**Estimated effort: 7 hours**
 
**Objective.** Train a gradient boosted tree model on the engineered feature matrix. Validate
correctly using leave-engines-out cross-validation. Analyze feature importance to confirm the
model is learning genuine degradation physics.
 
**Activities.**
 
1. **Leave-engines-out CV setup.** Split the 100 training engines into 5 folds of 20 engines
   each, randomly assigned by engine\_id (not by time). For each fold, train on the 80 engines
   not in the fold, predict on the 20 held-out engines' last-cycle feature vectors, compute
   RMSE. Average across folds gives the CV estimate.
   Why not walk-forward: see requirements_spec.md Section 2 and the design decisions section
   below. The key point: you are generalizing to new engines, not future cycles of known engines.
2. **Training.** After CV, retrain on all 100 training engines for the final model:
   ```python
   import lightgbm as lgb
   model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31)
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test).clip(min=0)
   ```
 
3. **Evaluate.** Compute RMSE and NASA score on the 100 test engines.
4. **Feature importance.** Plot the top 20 features by gain importance. The expectation: rolling
   statistics of the informative sensors (sensor\_11, sensor\_12, sensor\_4, sensor\_7 are
   typically the strongest in FD001) should dominate. If near-constant sensors or the raw cycle
   number dominate, something went wrong in Phase 3.
5. **Light hyperparameter search.** Try a small grid over `num_leaves` ∈ {15, 31, 63} and
   `min_child_samples` ∈ {10, 20, 50}. Select based on CV RMSE, not test RMSE (do not touch the
   test set until the final evaluation).
**Deliverable.** Trained LightGBM model. CV RMSE (honest estimate). Test RMSE and NASA score.
Feature importance chart. Comparison table updated.
 
---
 
### Phase 6 — Evaluation & comparison
**Estimated effort: 4 hours**
 
**Objective.** Understand not just how well the model performs, but where it fails and why. A
model whose RMSE number looks good but fails catastrophically on a specific subset of engines is
not a model you want making maintenance decisions.
 
**Activities.**
 
1. **Comparison table.** Finalize the table started in Phase 4:
   | Model | RMSE | NASA score | Notes |
   |---|---|---|---|
   | Naive | | | |
   | Linear degradation | | | |
   | LightGBM | | | |
   | 1D-CNN (stretch) | | | |
2. **Predicted vs. true scatter plot.** Plot `y_pred` vs. `rul_true` for the 100 test engines.
   A perfect model would give a diagonal line at y = x. Identify:
   - Systematic over-prediction (model is too optimistic — dangerous)
   - Systematic under-prediction (model is conservative — costly but safer)
   - High-variance engines (the model is confident but wrong)
3. **Error by true RUL.** Plot prediction error (d = pred − true) vs. true RUL. This reveals
   whether the model is better calibrated near failure (low true RUL, critical region) vs. far
   from failure (high true RUL, less critical). A good model should be most accurate when RUL
   is small.
4. **Failure case diagnosis.** Identify the 5–10 test engines with the largest prediction error.
   Plot their full observable sensor trajectories. Ask: is there something visually unusual about
   these engines that explains why the model was fooled?
5. **Written summary.** Document in 1–2 paragraphs: what the model learned, where it fails, and
   what the most likely cause of failures is.
**Deliverable.** Finalized comparison table. Scatter plot + error analysis plots. Written
diagnostic summary.
 
**Checkpoint.** This is the end of the core project. If phases 0–6 have consumed noticeably more
than ~34 hours by this point, drop Phase 7 for this iteration rather than compressing it — see
the schedule note below.
 
---
 
### Phase 7 — 1D-CNN (stretch, optional)
**Estimated effort: 8 hours**
 
**Objective.** Test whether the raw temporal sequence of sensor readings over the last W cycles
contains information beyond what the hand-engineered rolling statistics in Phase 3 already
capture. This is the question that justifies adding a deep learning model alongside LightGBM.
 
**Activities.**
 
1. **Data preparation.** Normalize each sensor channel first — z-score using mean/std computed
   on the training set only, then apply the same transform to test data (unlike LightGBM, tree
   splits are scale-invariant but Conv1D is not; skipping this step will visibly hurt training).
   For each (engine, cycle) training example, extract the last W = 30 cycles of the normalized
   sensor readings as a matrix of shape (W, n\_sensors). This is the model's input sequence. For
   test engines, use the last 30 available cycles.
2. **Architecture.** A small 1D-CNN is CPU-feasible at this dataset size:
   ```
   Input: (batch, 30, n_sensors)
   Conv1D(32, kernel=5, padding='same') → ReLU
   Conv1D(64, kernel=3, padding='same') → ReLU
   AdaptiveAvgPool1d(1) → flatten
   Linear(64 → 1)
   ```
 
3. **Training loop.** Standard PyTorch training loop. MSE loss. Adam optimizer. ~50 epochs.
   Training should complete in under 10 minutes on CPU.
4. **Evaluation.** Same metrics: RMSE and NASA score on the 100 test engines.
5. **Comparison.** The key question: does the CNN's RMSE beat LightGBM? If yes, the raw temporal
   structure adds signal beyond rolling statistics. If no, the Phase 3 features already captured
   most of what the sequence knows.
**Deliverable.** Trained 1D-CNN. Test RMSE and NASA score. Comparison table completed.
One-paragraph discussion of what the comparison implies about feature engineering vs. end-to-end
sequence learning for this problem.
 
---
 
## Schedule
 
### Working hours
 
| Day | Hours/day |
|---|---|
| Monday–Thursday | 3h |
| Friday | 2h |
| Saturday–Sunday | 4h |
| **Total per week** | **22h** |
 
### Day-by-day timeline
 
Start date: **Thursday 6 August 2026**.
 
| Date | Day | Daily hours | Cumulative hours | Phase activity |
|---|---|---|---|---|
| Thu 6 Aug | Thu | 3 | 3 | Phase 0 complete (2h) · Phase 1 start (1h) |
| Fri 7 Aug | Fri | 2 | 5 | Phase 1 continue (3h done of 7h) |
| Sat 8 Aug | Sat | 4 | 9 | Phase 1 complete (4h) |
| Sun 9 Aug | Sun | 4 | 13 | Phase 2 complete (3h) · Phase 3 start (1h) |
| Mon 10 Aug | Mon | 3 | 16 | Phase 3 continue (4h done of 7h) |
| Tue 11 Aug | Tue | 3 | 19 | Phase 3 complete (3h) |
| Wed 12 Aug | Wed | 3 | 22 | Phase 4 continue (3h done of 4h) |
| Thu 13 Aug | Thu | 3 | 25 | Phase 4 complete (1h) · Phase 5 start (2h) |
| Fri 14 Aug | Fri | 2 | 27 | Phase 5 continue (4h done of 7h) |
| Sat 15 Aug | Sat | 4 | 31 | Phase 5 complete (3h) · Phase 6 start (1h) |
| **Sun 16 Aug** | Sun | 4 | 35 | **Phase 6 complete (4h) ← core done** · Phase 7 start (1h) |
| Mon 17 Aug | Mon | 3 | 38 | Phase 7 continue (4h done of 8h) |
| Tue 18 Aug | Tue | 3 | 41 | Phase 7 continue (7h done of 8h) |
| **Wed 19 Aug** | Wed | 3 | 44 | **Phase 7 complete (1h) ← stretch done** · Buffer (2h) |
 
**Core project complete: Sunday 16 August** (phases 0–6, 34 hours) — unchanged from the original
schedule; the extra hour added to Phase 1 is absorbed without moving this date.
 
**Stretch complete: Wednesday 19 August** (all phases, 42 hours, with only **2 hours** of buffer
remaining, down from a full buffer day). Because the added Phase 1 rigor pushes total effort from
41h to 42h, margin is now thin. Treat Sunday 16 Aug as a hard checkpoint: if phases 0–6 ran over
~34 hours, skip Phase 7 for this iteration rather than compressing it into less time than
estimated.
 
---
 
## Critical design decisions
 
These six decisions must be resolved correctly upfront. Getting any of them wrong propagates
silently through the entire project and produces results that look plausible but are invalid.
 
### 1. CV strategy: leave-engines-out, not walk-forward
 
Walk-forward CV tests whether the model can continue an engine's trajectory after seeing its
early life. That is not the test-time task. At test time, each engine is entirely new — the
model has never seen any of its cycles during training. The CV strategy must match this: hold
out complete engines, train on the rest, validate on the held-out engines in their entirety.
 
Walk-forward gives an optimistically biased CV estimate because the model trains on the early
cycles of the same engines it is then validated on, giving it a structural head start that
disappears at actual test time.
 
### 2. RUL label cap at 125 cycles
 
Raw linear RUL labels in the healthy phase (more than ~125 cycles from failure) ask the model
to distinguish RUL = 300 from RUL = 350 based on sensor readings that are statistically
identical. The model cannot learn this — it learns noise. The cap converts all healthy-phase
labels to a constant (125), removing this false signal and allowing the model to focus on the
degradation phase where the sensor readings are actually informative.
 
The threshold of 125 is validated empirically in Phase 2 by plotting mean sensor values binned
by raw RUL: the degradation signal onset should visually align with the chosen cap value.
 
### 3. Test features use only each engine's own observable history
 
Rolling statistics for a test engine must be computed from its own cycles only — not from any
other engine's data, and not from cycles beyond the cut-off point. Cross-engine contamination
in feature computation is the look-ahead bias problem in a different form: it leaks information
from engines the model is not supposed to know about at prediction time.
 
Practical check: compute the feature matrix for each test engine independently, using only
cycles 1 through T\_last for that engine. Then take the feature vector at T\_last as the
model input.
 
### 4. NASA score must be evaluated separately from RMSE
 
A model that systematically over-predicts RUL — telling the scheduler the engine has more life
than it does — can achieve a low RMSE by being accurate on average while still creating
disproportionate risk through its largest positive errors. The NASA scoring function penalizes
these positive errors exponentially harder, making the score a more faithful proxy for the
maintenance scheduler's actual cost function. Always report both.
 
### 5. ACF/PACF must be computed on a stationarity-corrected series
 
Raw sensor trajectories trend toward failure by construction, so they are non-stationary — the
same condition the theory curriculum already covers, and the same rule applies here: ACF/PACF
are only informative once that trend is removed. Differencing each sensor within-engine before
computing ACF/PACF (Phase 1) turns "how far back does this trajectory's trend echo" — not a
useful question — into "what does this sensor's cycle-to-cycle noise look like" — a useful one.
Skipping this step doesn't just produce a slightly-off plot; it produces a plot that *looks*
informative (slow decay, non-zero at every lag) while actually reflecting nothing but the trend
that's already known to exist.
 
### 6. `engine_id` is excluded from the feature set
 
Training and test engine IDs both happen to run 1–100, but they refer to different physical
units — there is no real-world relationship between "engine 37 in training" and "engine 37 in
test." Any predictive weight the model puts on raw ID would be spurious pattern-matching on an
artifact of file numbering, not a source of legitimate generalization, so it's excluded outright
rather than left to the model (or a feature-importance check after the fact) to discover.
 
---
 
*Document version: 1.1 — revised after expert review (stationarity-corrected ACF/PACF, feature
count reconciliation, engine_id exclusion, CNN normalization, updated schedule/buffer).*
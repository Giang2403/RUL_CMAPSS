# Requirements specification — turbofan RUL prediction
## NASA CMAPSS predictive maintenance project
 
---
 
## 1. Business context
 
An operator runs a fleet of turbofan engines. Each engine accumulates operational cycles, and
sensor readings are collected at every cycle. The downstream decision this system feeds is
maintenance scheduling: sending an engine for inspection too early wastes downtime and incurs
unnecessary maintenance cost; sending it too late risks catastrophic in-flight failure.
 
The asymmetry between those two failure modes is not incidental. It governs the choice of
evaluation metric (see Section 4), and any model that ignores it — for example, one optimized
purely on symmetric RMSE — may produce predictions that look good on paper but fail the
scheduler's actual needs.
 
---
 
## 2. Formal problem statement
 
**Input.** The full sensor history of engine *i* from cycle 1 to the current cycle *t*:
 
```
{ (s₁, ..., s₂₁, op₁, op₂, op₃) }  for cycles 1 to t
```
 
where s₁...s₂₁ are 21 sensor channels and op₁...op₃ are 3 operational settings.
 
**Output.** A single non-negative real number: the estimated remaining useful life RUL(*i*, *t*),
measured in operational cycles.
 
**Problem type.** Supervised regression. This is not a classification problem (not "will it fail"),
not a time series forecasting problem (not "predict future sensor values"), but a health-state
regression: given the observable history up to now, estimate time remaining.
 
**Target construction (training data).** For training engines where the complete run-to-failure
trajectory is known:
 
```
RUL(i, t) = max_cycle(i) − t
```
 
then capped at 125 cycles. The cap is not a hack — it encodes the fact that sensor readings in the
healthy phase (more than 125 cycles from failure) are statistically indistinguishable across
engines. Asking the model to differentiate RUL = 300 from RUL = 350 in the training data asks it
to learn from noise; the cap removes this false signal. See Section 6 of the implementation plan
for a full justification.
 
---
 
## 3. Data specification
 
**Dataset.** NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation).
**Primary sub-dataset.** FD001 — single operating condition, single fault mode, cleanest signal.
**Stretch sub-datasets.** FD002 (6 operating conditions) and FD004 (6 conditions, 2 fault modes).
 
### 3.1 Raw file format
 
Space-delimited flat text, no headers. Each row is one (engine, cycle) observation.
 
| Column(s) | Content |
|---|---|
| 1 | engine\_id (integer, 1-indexed) |
| 2 | cycle (integer, starts at 1 for each engine) |
| 3–5 | operational settings: op\_1, op\_2, op\_3 |
| 6–26 | sensor readings: sensor\_1 through sensor\_21 |
 
### 3.2 Training data
 
- 100 engines, each observed from cycle 1 until failure (end of useful life).
- Approximately 20,000 rows total.
- RUL label is computable for every row: `max_cycle(engine_id) − cycle`.
### 3.3 Test data
 
- 100 engines, each truncated at some unknown point before failure.
- One prediction is required per engine: the estimated RUL at the final observed cycle.
- Ground truth: a separate file `RUL_FD001.txt` containing the true RUL for each of the
  100 test engines (one value per line, in engine order).
### 3.4 Scale
 
Entire FD001 dataset (train + test) fits comfortably in memory on a standard laptop. No data
pipeline or streaming infrastructure is required.
 
---
 
## 4. Success criteria
 
### 4.1 Primary metric — RMSE
 
Root mean squared error on the 100 test engines' RUL predictions.
 
```
RMSE = sqrt( (1/N) * Σᵢ (RUL_predicted(i) − RUL_true(i))² )
```
 
**Floor to beat.** A linear degradation baseline on FD001 typically achieves RMSE ≈ 35 cycles.
Every model submitted must beat this floor before being considered.
 
**Competitive target.** Published LightGBM-class results on FD001 fall in the range RMSE 12–18
cycles. This is the stretch target for Phase 5.
 
### 4.2 Secondary metric — NASA asymmetric scoring function
 
```
S = Σᵢ f(dᵢ)
 
where dᵢ = RUL_predicted(i) − RUL_true(i)
 
f(d) = exp(−d / 13) − 1    if d < 0   (early prediction: model is conservative)
f(d) = exp( d / 10) − 1    if d ≥ 0   (late prediction: model is optimistic)
```
 
Lower S is better. The denominator difference (13 vs 10) makes late predictions penalized
exponentially harder than early predictions of the same magnitude. A model that systematically
over-predicts RUL — optimistically telling the scheduler the engine has more life left than it
does — will score poorly here even if its RMSE is acceptable.
 
Both metrics must be reported for every model.
 
### 4.3 Sanity checks
 
- Predicted RUL values must be clipped to ≥ 0 before evaluation (negative RUL is not meaningful).
- Feature importance (Phase 5) must show that the degrading sensors — not the near-constant
  sensors identified in EDA — dominate the model's predictions.
- `engine_id` must not be used as a model input feature. Training and test engine IDs both run
  1–100, but they refer to different physical units — any predictive signal the model finds
  through raw ID would be spurious and would not generalize to a real fleet.
---
 
## 5. Constraints
 
### 5.1 Hardware
 
- CPU only. No GPU access.
- Standard laptop RAM (≤ 16 GB). Given the FD001 dataset size (~20K rows), this is not a
  binding constraint.
### 5.2 Software
 
| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data loading, feature engineering |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | Preprocessing, cross-validation utilities |
| `lightgbm` | Primary model (Phase 5) |
| `statsmodels` | ACF / PACF computation (Phase 1) |
| `torch` | Stretch CNN model (Phase 7, optional) |
 
**Note on Phase 7 (CNN).** Unlike the LightGBM path, where tree splits are scale-invariant, the
CNN's input is not — sensor channels vary widely in raw magnitude. If Phase 7 is attempted,
normalize each channel (e.g. z-score) using statistics computed on the training set only, then
apply the same transform to test data.
 
No external data sources beyond NASA CMAPSS. No pretrained models.
 
### 5.3 Time
 
- Monday–Thursday: 3 hours/day
- Friday: 2 hours/day
- Saturday–Sunday: 4 hours/day
- Total available per week: 22 hours
---
 
## 6. Out of scope
 
The following are explicitly excluded from this project:
 
- **Prediction intervals or uncertainty quantification.** Point estimates only.
- **Online or streaming inference.** Batch prediction on fixed test set only.
- **Real-time monitoring pipeline.** No deployment infrastructure.
- **Multi-asset fleet optimization.** Individual engine RUL only.
- **FD003.** Deferred; FD001 and FD002/FD004 are sufficient to demonstrate the key concepts.
- **Hyperparameter optimization beyond manual grid search.** Bayesian optimization or AutoML
  tools are out of scope for this version.
---
 
*Document version: 1.1 — revised after expert review (engine_id exclusion, CNN normalization note).*
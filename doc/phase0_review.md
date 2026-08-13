# Phase 0 review — setup & data loading
 
Reviews [notebook/phase0_setup_data_loading.ipynb](../notebook/phase0_setup_data_loading.ipynb)
against the Phase 0 spec in [plan.md](plan.md#phase-0--setup--data-loading) and
[requirements_spec.md §3](requirements_spec.md).
 
---
 
## 1. Defined deliverables (from plan.md)
 
**Objective.** Get clean, well-named DataFrames into memory — everything downstream depends on
this being correct.
 
**Deliverable statement.** *"Three clean Python objects — `train_df`, `test_df`, `rul_true` —
verified and ready to use for the rest of the project."*
 
Broken into the five activities specified:
 
| # | Activity |
|---|---|
| 1 | Obtain the raw FD001 files: `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` |
| 2 | Parse the space-delimited format; assign column names (`engine_id, cycle, op_1..op_3, sensor_1..sensor_21`) |
| 3 | Verify integrity: no missing values, min/max cycle counts per engine, confirm 100 train + 100 test engines |
| 4 | Load `RUL_FD001.txt` into a Series indexed by engine ID (1-indexed, one value per engine) |
| 5 | Print a summary: shape of each DataFrame, dtypes, min/max values for key columns |
 
---
 
## 2. Summary of work done
 
- **Loading.** `load_fd001()` reads both raw files with `sep=r"\s+"`, no header, and the 26-column
  name list (`engine_id, cycle, op_1..op_3, sensor_1..sensor_21`), casting `engine_id`/`cycle` to
  `int`. Produces `train_df` (20,631 rows) and `test_df` (13,096 rows).
- **RUL labels.** `RUL_FD001.txt` loaded into `rul_true`, a Series indexed 1–100 (`engine_id`),
  named `rul`.
- **Missing values.** `isna().sum()` checked and asserted zero across `train_df`, `test_df`, and
  `rul_true`.
- **Cycle counts.** `groupby('engine_id')['cycle'].max()` computed for **training** engines only —
  min 128 (engine 39), max 362 (engine 69), mean 206.3.
- **Engine counts.** Asserted `nunique()` == 100 for both `train_df` and `test_df` engine IDs, and
  100 rows in `rul_true`; also asserted the ID sets equal `{1..100}` exactly for all three.
- **Summary output.** Shapes printed for all three objects; dtypes printed for `train_df`; a
  min/max table printed for a subset of key columns (`cycle, op_1, op_2, op_3, sensor_1,
  sensor_4, sensor_11`) on `train_df` and `test_df`, plus the min/max of `rul_true`.
---
 
## 3. Mapping to deliverables
 
| Activity | Status | Evidence |
|---|---|---|
| 1. Obtain raw files | ✅ Done (prerequisite) | Files present in `data/`; loaded successfully by the notebook |
| 2. Parse + assign column names | ✅ Done | `COLUMNS` list + `load_fd001()`, verified against `train_df.head()` |
| 3. Verify integrity — missing values | ✅ Done | Section 2, all three objects asserted zero-missing |
| 3. Verify integrity — min/max cycles per engine | ⚠️ Partial | Computed for **train** engines only (Section 3); not computed for **test** engines |
| 3. Verify integrity — 100/100 engine confirmation | ✅ Done | Section 4, exact ID-set equality asserted, not just counts |
| 4. `rul_true` as indexed Series | ✅ Done | 1-indexed by `engine_id`, one value/engine |
| 5. Summary print (shapes, dtypes, key-column ranges) | ✅ Mostly done | Shapes for all 3; dtypes for `train_df` only (not `test_df`, though same schema); key-column min/max for both `train_df` and `test_df` |
| **Deliverable: three clean, verified objects ready for reuse** | ✅ **Met** | `train_df`, `test_df`, `rul_true` all constructed, validated, and error-free |
 
---
 
## 4. Gaps / optional follow-ups
 
None of these block moving to Phase 1 — the core deliverable is met — but worth a quick look if
convenient:
 
1. **Per-engine cycle range for `test_df` was not computed.** Section 3 only covers training
   engines. Test engines are truncated before failure, so their per-engine max-cycle distribution
   is a different (and equally relevant) quantity — e.g. useful later for sanity-checking that no
   test engine's truncation point looks implausible. A one-line addition:
   `test_df.groupby('engine_id')['cycle'].max().describe()`.
2. **`test_df.dtypes` wasn't printed**, only `train_df.dtypes`. Since both come from the same
   `load_fd001()` path the schemas should match, but printing both costs nothing and removes the
   assumption.
Neither gap affects correctness of the three deliverable objects — both are cheap additions if you
want the verification section fully symmetric between train and test before moving on.
 
---
 
## 5. Verdict
 
**Phase 0 deliverable is met.** `train_df`, `test_df`, and `rul_true` are correctly parsed,
schema-verified, free of missing values, confirmed at exactly 100 engines each, and summarized —
ready for Phase 1 (EDA & sensor analysis).
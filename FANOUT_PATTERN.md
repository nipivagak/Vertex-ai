# MLForecast Pipeline: Fan-Out Pattern

## Overview

The pipeline now includes two implementations:

### 1. **Fan-Out Pipeline** (Recommended)
- **File**: [pipeline_mlforecast_fanout.py](pipeline_mlforecast_fanout.py)
- **YAML Output**: `nixtla_mlforecast_fanout_pipeline.yaml`
- **Key Feature**: Dynamically trains multiple models specified in `model_types` parameter
- **Use**: Run with `python3 run_pipeline.py fanout`

### 2. **Original Pipeline** (Legacy)
- **File**: [pipeline_mlforecast.py](pipeline_mlforecast.py)
- **YAML Output**: `nixtla_mlforecast_parallel_pipeline.yaml`  
- **Key Feature**: Hardcoded 3 models (lgbm, rf, et)
- **Use**: Run with `python3 run_pipeline.py original`

---

## Fan-Out Architecture

The fan-out pattern implements **parallelization at scale**:

```
┌─────────────────────────────────────────────────────────────┐
│ Pipeline Input: model_types = ["lgbm", "rf", "et", "xgb"] │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  FAN-OUT: Create    │
                    │  Training Tasks     │
                    │  for each type      │
                    └─────────────────────┘
            ┌───────────────┬───────────────┬───────────────┬───────────────┐
            │               │               │               │               │
            ▼               ▼               ▼               ▼               ▼
        train-lgbm      train-rf       train-et       train-xgb    (any models)
          Task 1         Task 2         Task 3         Task 4       (parallel)
            │               │               │               │
            │               └───────┬───────┘               │
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │  FAN-IN: Collect All Results  │
                    │  Select Champion              │
                    └───────────────────────────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
                ▼                   ▼                   ▼
        register-deploy-     batch-forecast      (other post-processing)
          champion                 │
                                    ▼
                        Write batch forecasts
                        to BigQuery
```

### Key Differences

| Aspect | Fan-Out | Original |
|--------|---------|----------|
| **Model Count** | Dynamic (1-N) | Fixed (3) |
| **Parameter** | `model_types: ["lgbm", "rf", "et", ...]` | None |
| **Task Creation** | Python loop | Hardcoded |
| **Flexibility** | Add new models by changing one parameter | Modify code |
| **Scale** | Can train 10+ models efficiently | Limited to 3 |

---

## Usage

### 1. **Compile Pipeline** (Choose One)

**Fan-Out (Recommended)**:
```bash
python3 pipeline_mlforecast_fanout.py
# Outputs: nixtla_mlforecast_fanout_pipeline.yaml
```

**Original**:
```bash
python3 pipeline_mlforecast.py
# Outputs: nixtla_mlforecast_parallel_pipeline.yaml
```

### 2. **Provision BigQuery Tables**

```bash
python3 setup_bq_tables.py
```

### 3. **Submit to Vertex AI**

**Fan-Out Mode** (default):
```bash
python3 run_pipeline.py fanout
```

**Original Mode**:
```bash
python3 run_pipeline.py original
```

### 4. **Customize Model Types in Fan-Out**

Edit [run_pipeline.py](run_pipeline.py#L27) and change the `model_types` list:

```python
PARAMETER_VALUES = {
    ...
    "model_types": ["lgbm", "rf", "et", "xgb", "gb", "ridge", "svr"],  # Mix and match!
    ...
}
```

**Available Models**:

| Model Code | Model Name | Best For | Hyperparameters |
|-----------|-----------|----------|-----------------|
| `lgbm` | LightGBM | Fast gradient boosting | n_est=300, lr=0.05, leaves=64 |
| `rf` | Random Forest | Baseline ensemble | n_est=300, depth=10 |
| `et` | Extra Trees | Fast ensemble | n_est=300, depth=10 |
| `xgb` | XGBoost | High-performance boosting | n_est=300, lr=0.05, depth=6 |
| `gb` | Gradient Boosting | Stable boosting | n_est=300, lr=0.05, depth=5 |
| `ridge` | Ridge Regression | Linear + L2 regularization | alpha=1.0 |
| `svr` | Support Vector Regressor | Non-linear patterns | kernel=rbf, C=100 |

**Example Combinations**:

Lightweight (fast training):
```python
"model_types": ["rf", "ridge", "lgbm"]
```

Comprehensive comparison (all models):
```python
"model_types": ["lgbm", "rf", "et", "xgb", "gb", "ridge", "svr"]
```

Production (proven winners):
```python
"model_types": ["lgbm", "xgb", "rf"]
```

---

## DAG Execution Order

1. **Fan-Out Phase** (parallel, all run simultaneously):
   - `train-lgbm`
   - `train-rf`
   - `train-et`
   - Any additional `train-{model_type}` tasks

2. **Aggregation Phase**:
   - `select-champion` (waits for all training to complete)

3. **Deployment & Scoring Phase** (parallel):
   - `register-deploy-champion`
   - `batch-forecast`

---

## BigQuery Tables

All of these are automatically created by [setup_bq_tables.py](setup_bq_tables.py#L42):

| Table | Purpose | Rows Per Run |
|-------|---------|--------------|
| `sales_predictions` | Raw predictions from training validation | N_models × N_series × horizon |
| `sales_champion` | Champion model metadata | 1 row |
| `sales_batch_forecasts` | Future forecasts from batch_forecast_component | N_series × horizon |

---

## Benefits of Fan-Out

✅ **Scalability**: Train 1-10+ models in parallel without code changes  
✅ **Flexibility**: Add models by updating a parameter list  
✅ **Maintainability**: Single component logic works for all models  
✅ **Efficiency**: All models train simultaneously (faster than sequential)  
✅ **Auditability**: Champion table tracks which model won across runs  

---

## Real-World Example

Train 5 models including XGBoost:

```python
PARAMETER_VALUES = {
    ...
    "model_types": ["lgbm", "rf", "et", "xgb", "sklearn_rf"],
    ...
}
```

Then add conditional logic in [train_single_model_component](pipeline_mlforecast_fanout.py#L73) to handle the new types, and the pipeline will automatically:
- Create 5 parallel training tasks
- Train all in parallel
- Select the best champion
- Deploy and forecast with it

---

## Monitoring in Vertex AI UI

After submission:
1. Open [Vertex AI Pipelines](https://console.cloud.google.com/vertex-ai/pipelines)
2. Click your run
3. See the fan-out DAG:
   - **Parallel execution** of all `train-*` tasks
   - **Sequential execution** of champion selection → deployment → batch forecast

---

## Next Steps

### To Add More Custom Models

If you want to add additional models beyond the 7 supported ones:

1. **Install the package** in [pipeline_mlforecast_fanout.py](pipeline_mlforecast_fanout.py#L12):
   ```python
   packages_to_install=[
       ...
       "catboost>=1.2.0",  # Example: CatBoost
   ],
   ```

2. **Import the model** in [train_single_model_component](pipeline_mlforecast_fanout.py#L51):
   ```python
   from catboost import CatBoostRegressor
   ```

3. **Add model logic** in the `if/elif` chain ([pipeline_mlforecast_fanout.py](pipeline_mlforecast_fanout.py#L118)):
   ```python
   elif model_type == "catboost":
       model = CatBoostRegressor(
           iterations=300,
           learning_rate=0.05,
           depth=5,
           random_state=42,
           verbose=False,
       )
   ```

4. **Update the error message** to include the new model code

5. **Run**:
   ```bash
   python3 pipeline_mlforecast_fanout.py
   python3 run_pipeline.py fanout
   ```

6. **Use in your run**:
   ```python
   PARAMETER_VALUES = {
       ...
       "model_types": ["lgbm", "xgb", "catboost"],  # Your new model!
       ...
   }
   ```

---

## Existing Model Support

To add existing scikit-learn or tree-based models (CatBoost, NGBoost, Hist Gradient Boosting, etc.):

1. Install the package
2. Import it
3. Add the conditional branch
4. Done!

All metrics (RMSE, MAE, WMAPE) are computed the same way for any model type.

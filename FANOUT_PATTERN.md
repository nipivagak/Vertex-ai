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
    "model_types": ["lgbm", "xgb", "arima", "ets", "croston_optimized"],  # Mix any!
    ...
}
```

## Available Models: 30+ Options

### Tree-Based (6 models)

| Code | Model | Library | Best For |
|------|-------|---------|----------|
| `lgbm` | LightGBM | LightGBM | Fast gradient boosting |
| `rf` | Random Forest | scikit-learn | Baseline ensemble |
| `et` | Extra Trees | scikit-learn | Fast ensemble |
| `xgb` | XGBoost | XGBoost | High-performance boosting |
| `gb` | Gradient Boosting | scikit-learn | Stable boosting |
| `ridge` | Ridge Regression | scikit-learn | Linear + L2 |

### Support Vector & Linear (2 models)

| Code | Model | Library | Best For |
|------|-------|---------|----------|
| `svr` | SVR | scikit-learn | Non-linear patterns |
| `ridge` | Ridge | scikit-learn | Linear baseline |

### Statistical & Forecasting (10 models)

| Code | Model | Library | Best For | Notes |
|------|-------|---------|----------|-------|
| `arima` | AutoARIMA | statsforecast | Classical ARIMA | Auto selects (p,d,q) |
| `ets` | AutoETS | statsforecast | Exponential Smoothing | Auto-selects components |
| `theta` | AutoTheta | statsforecast | Theta method | Seasonal patterns |
| `imapa` | IMAPA | statsforecast | Intermittent demand | Sparse/irregular data |
| `adida` | ADIDA | statsforecast | Intermittent demand | Sparse/irregular data |
| `croston` | Croston | statsforecast | Intermittent demand | Classic Croston method |
| `croston_optimized` | Croston Optimized | statsforecast | Intermittent demand | Optimized Croston |
| `croston_sba` | Croston SBA | statsforecast | Intermittent demand | SBA variant |
| `mstl` | MSTL | statsforecast | Multiple seasonality | Multi-seasonal decomposition |

### GARCH Models (7 models)

Used for volatility modeling in time series with conditional heteroskedasticity.

| Code | Model | Arch | Best For |
|------|-------|------|----------|
| `garch_1_1` | GARCH(1,1) | arch library | Standard volatility |
| `garch_1_2` | GARCH(1,2) | arch library | More MA terms |
| `garch_2_1` | GARCH(2,1) | arch library | More AR terms |
| `garch_2_2` | GARCH(2,2) | arch library | Complex volatility |
| `garch_3_1` | GARCH(3,1) | arch library | Extreme volatility |
| `garch_3_2` | GARCH(3,2) | arch library | Very complex volatility |
| `garch_3_3` | GARCH(3,3) | arch library | Maximum flexibility |

### ARCH Models (2 models)

Used for modeling heteroskedasticity in time series.

| Code | Model | Library | Best For |
|------|-------|---------|----------|
| `arch_2` | ARCH(p=2) | arch library | Simple architecture |
| `arch_3` | ARCH(p=3) | arch library | Complex architecture |

---

## Example Combinations

### Quick Comparison (4 models)
```python
"model_types": ["lgbm", "xgb", "arima", "ets"]
```

### Comprehensive Battle (12 models)
```python
"model_types": ["lgbm", "rf", "xgb", "gb", "arima", "ets", "theta", "croston_optimized", "imapa", "garch_1_1", "garch_2_2", "svr"]
```

### Intermittent Demand Focus
```python
"model_types": ["imapa", "adida", "croston", "croston_optimized", "croston_sba"]
```

### Multi-Seasonal Data
```python
"model_types": ["mstl", "arima", "ets", "theta"]
```

### Volatility Modeling
```python
"model_types": ["garch_1_1", "garch_2_2", "garch_3_3", "arch_2", "arch_3"]
```

### Production (Proven Winners)
```python
"model_types": ["lgbm", "xgb", "arima", "ets"]
```

---

## Model Categories by Use Case

**If your data is:**
- **Regular/high frequency**: Use tree models or ARIMA
- **Intermittent/sparse**: Use IMAPA, ADIDA, Croston variants
- **Multi-seasonal**: Use MSTL, ETS with multiple seasonalities
- **Volatile**: Use GARCH/ARCH models
- **Linear pattern**: Use Ridge, ARIMA
- **Non-linear**: Use XGBoost, SVR, LightGBM

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

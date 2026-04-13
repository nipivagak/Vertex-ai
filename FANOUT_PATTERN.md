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
    "model_types": ["lgbm", "rf", "et", "xgb"],  # Add more models!
    ...
}
```

Supported models (defined in [train_single_model_component](pipeline_mlforecast_fanout.py#L73)):
- `lgbm` - LightGBM Regressor
- `rf` - Random Forest Regressor
- `et` - Extra Trees Regressor
- (extend with more in component logic)

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

To add more model types:

1. **Extend** [train_single_model_component](pipeline_mlforecast_fanout.py#L73) with new `elif model_type == "xgb":` branches
2. **Update** [setup_bq_tables.py](setup_bq_tables.py) if schema changes
3. **Run**:
   ```bash
   python3 pipeline_mlforecast_fanout.py
   python3 run_pipeline.py fanout
   ```

Done! No other changes needed.

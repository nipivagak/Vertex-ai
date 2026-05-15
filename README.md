# Vertex AI Forecast Pipeline

## Setup: Upload Pipeline Scripts to Cloud Storage

Before submitting pipeline runs, upload the pipeline scripts to your Cloud Storage bucket:

```bash
gsutil -m cp run_ray_pipeline.py gs://dazzling-seat-366014-vertex-pipelines/nixtla-pipeline-scripts/
gsutil -m cp ray_job/train_nixtla_ray_cluster.py gs://dazzling-seat-366014-vertex-pipelines/nixtla-pipeline-scripts/ray_job/
```

Update your `pipeline_config.yaml` to point to your GCS bucket:

```yaml
gcs_bucket: "dazzling-seat-366014-vertex-pipelines"
```

## Submit Pipeline Run

Use the following command to submit a Vertex AI Pipeline run:

```bash
gcloud ai pipelines runs submit \
  --project=dazzling-seat-366014 \
  --region=us-central1 \
  --pipeline-root=gs://dazzling-seat-366014-vertex-pipelines \
  --display-name=my-forecast-$(date +%s) \
  --template-path=nixtla_vertex_ray_pipeline.yaml \
  --parameters=@pipeline_config.yaml
```

## Run From Notebook (ipykernel)

Use the following cells in Vertex AI Workbench/Jupyter.

### Cell 1: Install dependencies in the active kernel

```python
%pip install -q google-cloud-aiplatform pyyaml kfp
```

### Cell 2: Compile the pipeline YAML

```python
!python nixtla_vertex_ray_pipeline.py --output nixtla_vertex_ray_pipeline.yaml
```

### Cell 3: Submit pipeline job with the Python SDK

```python
from google.cloud import aiplatform
import yaml
import time

PROJECT_ID = "dazzling-seat-366014"
REGION = "us-central1"

with open("pipeline_config.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

aiplatform.init(project=PROJECT_ID, location=REGION)

job = aiplatform.PipelineJob(
    display_name=f"my-forecast-{int(time.time())}",
    template_path="nixtla_vertex_ray_pipeline.yaml",
    pipeline_root="gs://dazzling-seat-366014-vertex-pipelines",
    parameter_values=params,
    enable_caching=False,
)
job.submit()
print(job.resource_name)
```

### Cell 4: Optional wait for completion

```python
job.wait()
print(job.state)
```

### Alternative: one command from notebook

```python
!python submit_pipeline_workbench.py
```

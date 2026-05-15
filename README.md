# Vertex AI Forecast Pipeline

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

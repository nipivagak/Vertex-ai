import argparse
import time

import yaml
from google.cloud import aiplatform


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit Vertex AI Pipeline job from Workbench using Python SDK.",
    )
    parser.add_argument("--project_id", default="dazzling-seat-366014")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--template_path", default="nixtla_vertex_ray_pipeline.yaml")
    parser.add_argument("--pipeline_root", default="gs://dazzling-seat-366014-vertex-pipelines")
    parser.add_argument("--parameter_values_file", default="pipeline_config.yaml")
    parser.add_argument("--display_name", default=None)
    parser.add_argument(
        "--enable_caching",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.parameter_values_file, "r", encoding="utf-8") as f:
        parameter_values = yaml.safe_load(f)

    aiplatform.init(project=args.project_id, location=args.region)

    display_name = args.display_name or f"my-forecast-{int(time.time())}"

    job = aiplatform.PipelineJob(
        display_name=display_name,
        template_path=args.template_path,
        pipeline_root=args.pipeline_root,
        parameter_values=parameter_values,
        enable_caching=args.enable_caching,
    )
    job.submit()

    print(f"Submitted pipeline job: {job.resource_name}")
    print("Open Vertex AI Pipelines in Console to monitor progress.")


if __name__ == "__main__":
    main()

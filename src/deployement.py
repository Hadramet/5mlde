import config
from training import train_model
from training import batch_inference

from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import (
    CronSchedule,
    IntervalSchedule,
)

model_deployement_every_friday = Deployment.build_from_flow(
    name="Model Training - Deployment",
    flow=train_model,
    version='1.0.0',
    tags=['model'],
    schedule=CronSchedule(cron="0 0 * * FRI"),
    parameters={
        "data_path": config.DATA_PATH,
    }
)

inference_deployement_every_10_minutes = Deployment.build_from_flow(
    name="Batch Inference - Deployment",
    flow=batch_inference,
    version='1.0.0',
    tags=['inference'],
    schedule=IntervalSchedule(interval= 10 * 60),
    parameters={
        "input_path": config.DATA_PATH,
    }
)

if __name__ == '__main__':
    model_deployement_every_friday.apply()
    inference_deployement_every_10_minutes.apply()
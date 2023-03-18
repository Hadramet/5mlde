import config
from training import train_model
from training import batch_inference
from training import data_validation

from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import (
    CronSchedule,
    IntervalSchedule,
)

data_validation_every_10_minutes = Deployment.build_from_flow(
    name="Data validation - Deployment",
    flow=data_validation,
    version='1.0.0',
    tags=['data'],
    work_pool_name="5mlde",
    schedule=IntervalSchedule(interval=1 * 60),
    parameters={
        "checkpoint_name": "gx_checkpoint",
    }
)

model_deployement_every_friday = Deployment.build_from_flow(
    name="Model Training - Deployment",
    flow=train_model,
    version='1.0.0',
    tags=['model'],
    work_pool_name="5mlde",
    schedule=IntervalSchedule(interval=2 * 60),
    parameters={
        "data_path": config.DATA_PATH,
    }
)

inference_deployement_every_10_minutes = Deployment.build_from_flow(
    name="Batch Inference - Deployment",
    flow=batch_inference,
    version='1.0.0',
    tags=['inference'],
    work_pool_name="5mlde",
    schedule=IntervalSchedule(interval=3 * 60),
    parameters={
        "input_path": config.DATA_PATH,
    }
)

if __name__ == '__main__':
    data_validation_every_10_minutes.apply()
    model_deployement_every_friday.apply()
    inference_deployement_every_10_minutes.apply()


import config

from training import train_model
from training import batch_inference
from training import data_validation
from prefect.deployments import Deployment

data_validation_every_10_minutes = Deployment.build_from_flow(
    name="Data validation - Deployment",
    flow=data_validation,
    version='1.0.0',
    tags=['data'],
    parameters={
        "checkpoint_name": "email_checkpoint",
    }
)

model_deployement_every_friday = Deployment.build_from_flow(
    name="Model Training - Deployment",
    flow=train_model,
    version='1.0.0',
    tags=['model'],
    parameters={
        "data_path": config.DATA_PATH,
    }
)

inference_deployement_every_10_minutes = Deployment.build_from_flow(
    name="Batch Inference - Deployment",
    flow=batch_inference,
    version='1.0.0',
    tags=['inference'],
    parameters={
        "input_path": config.DATA_PATH,
    }
)

if __name__ == '__main__':
    config.init_folders()
    data_validation_every_10_minutes.apply()
    model_deployement_every_friday.apply()
    inference_deployement_every_10_minutes.apply()


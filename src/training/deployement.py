import config

from training import train_model
from training import data_validation, stage_latest_best_model
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

best_model_deployement = Deployment.build_from_flow(
    name="Best Model Training - Deployment",
    flow=stage_latest_best_model,
    version='1.0.0',
    tags=['model'],
    parameters={
        "version": 0,
        "model_name": "email_spam_model",
        "stage": "Production",
    }
)


if __name__ == '__main__':
    config.init_folders()
    data_validation_every_10_minutes.apply()
    model_deployement_every_friday.apply()
    best_model_deployement.apply()


import os
import config
import gdown
from prefect import task

@task(name='load_data', retries=2, retry_delay_seconds=60)
def download_data():
    if not os.path.exists(config.DATA_FOLDER):
        os.makedirs(config.DATA_FOLDER, exist_ok=True)
    gdown.download(config.DATA_URL, config.DATA_PATH, quiet=False , fuzzy=True)

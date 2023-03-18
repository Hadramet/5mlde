import requests
import time
import subprocess

ORION_SERVER_URL = "http://localhost:4200"
CUSTOM_WORK_POOL = "5mlde"

session = requests.Session()

def check_server_health(url, timeout=60):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health")
            if response.status_code == 200 and response:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)

    return False

def start_orion_server():
    """Start Orion server."""

    set_api_url_cmd = f"prefect config set PREFECT_API_URL={ORION_SERVER_URL}/api"
    subprocess.run(set_api_url_cmd, shell=True, check=True)

    start_server_cmd = "prefect server start --host 0.0.0.0"
    subprocess.Popen(start_server_cmd, shell=True)

    if not check_server_health(f"{ORION_SERVER_URL}/api"):
        raise RuntimeError("Server health check failed")

    create_pool_cmd = f"prefect work-pool create {CUSTOM_WORK_POOL}"
    subprocess.run(create_pool_cmd, shell=True, check=True)

    start_agent_cmd = f"prefect agent start -p {CUSTOM_WORK_POOL}"
    subprocess.run(start_agent_cmd, shell=True, check=True)

    

if __name__ == "__main__":
    start_orion_server()
import requests
import time
import subprocess

ORION_SERVER_URL = "http://localhost:4200"

session = requests.Session()

def start_orion_server():
    """Start Orion server."""
    set_api_url_cmd = f"prefect config set PREFECT_API_URL={ORION_SERVER_URL}/api"
    subprocess.run(set_api_url_cmd, shell=True, check=True)

    start_server_cmd = "prefect orion start --host 0.0.0.0"
    subprocess.run(start_server_cmd, shell=True, check=True)
    



def wait_for_server_ready(time_out: int = 60):
    """Wait for Orion server to be ready."""
    up = False
    deadline = time.time() + time_out
    while time.time() < deadline :
        print("Waiting for Orion server to be ready...")
        try:
            response = session.get(f"{ORION_SERVER_URL}/api/health")
            response.raise_for_status()
            up = response.json()
            if up:
                print("Orion server is ready.")
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    if not up:
        raise RuntimeError("Orion server is not ready.")

if __name__ == "__main__":
    start_orion_server()
    wait_for_server_ready()
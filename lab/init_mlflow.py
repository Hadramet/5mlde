import subprocess
cmd = f"mlflow ui --port 5000 --host 0.0.0.0"
subprocess.run(cmd, shell=True, check=True)

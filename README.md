# 5mlde

This repo contains the group work done as part of the 5MLDE course at supinfo school. It is about deploying machine learning models on different platforms.

## Getting Started

These instructions will help you set up the development environment for Jupyter and MLflow using Docker Compose.

### Prerequisites

- Docker: Install Docker on your machine by following the instructions on the [Docker website](https://www.docker.com).
- Docker Compose: Install Docker Compose by following the instructions on the [Docker Compose website](https://docs.docker.com/compose/install/).
- Visual Studio Code: Install Visual Studio Code by following the instructions on the [Visual Studio Code website](https://code.visualstudio.com).
- Remote - Containers extension: Install the "Remote - Containers" extension for Visual Studio Code from the [Visual Studio Code marketplace](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

### Setting up the development environment

1. Clone this repository: git clone https://github.com/hadramet/5mlde.git
2. Open the project folder in Visual Studio Code.
3. Click the "Reopen in Container" option in the bottom-left corner of the window. If you don't see the option, press `F1` and type "Remote-Containers: Reopen in Container" in the command palette.This will launch the Jupyter and MLflow containers using the Docker Compose configuration.
4. Once the containers are up and running, you can access the Jupyter notebook at `http://localhost:8888` and the MLflow server at `http://localhost:5000`.

### Running prefect server

1. Set the API URL for the local server
```bash
prefect config set PREFECT_API_URL=http://0.0.0.0:4200/api
```
2. Start the local Prefect server
```bash
prefect orion start --host 0.0.0.0 
```

## License

This project is licensed under the [MIT License](LICENSE).
#!/bin/bash

function check_dependencies() {
  command -v docker >/dev/null 2>&1 || { echo >&2 "Docker is not installed. Please install Docker and try again."; exit 1; }
  command -v docker-compose >/dev/null 2>&1 || { echo >&2 "Docker Compose is not installed. Please install Docker Compose and try again."; exit 1; }
}

function start_services() {
  check_dependencies

  read -p "Enter the number of workers (default: ${NUM_WORKERS:-5}): " workers
  if [ -z "$workers" ]; then
    workers="${NUM_WORKERS:-5}"
  fi

  echo "Starting services with $workers worker(s)..."
  cd src && docker-compose up --scale worker="$workers" -d --build

  echo "Waiting for services to start..."
  sleep 30
	echo "Services started successfully."
}

function stop_services() {
  check_dependencies

  echo "Stopping services and removing volumes..."
  cd src && docker-compose down --volumes
}

function build {
  check_dependencies

  echo "Building services..."
  cd src && docker-compose build
}

if [ "$1" == "start-services" ]; then
  start_services
elif [ "$1" == "stop-services" ]; then
  stop_services
elif [ "$1" == "build" ]; then
  build
else
  echo "Usage: $0 <start-services|stop-services>"
fi

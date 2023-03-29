.DEFAULT_GOAL := help
.PHONY: help start-services stop-services check-dependencies

help:
	@echo "Available commands:"
	@echo "    start-services\tStart services with the user-specified or default number of workers"
	@echo "    stop-services\tStop services and remove volumes"

NUM_WORKERS ?= 5

check-dependencies:
	@command -v docker >/dev/null 2>&1 || { echo >&2 "Docker is not installed. Please install Docker and try again."; exit 1; }
	@command -v docker-compose >/dev/null 2>&1 || { echo >&2 "Docker Compose is not installed. Please install Docker Compose and try again."; exit 1; }

start-services: check-dependencies
	@read -p "Enter the number of workers (default: $(NUM_WORKERS)): " workers; \
	if [ -z "$$workers" ]; then \
		workers=$(NUM_WORKERS); \
	fi; \
	echo "Starting services with $$workers worker(s)..."; \
	cd src && docker-compose up --scale worker=$$workers -d

stop-services: check-dependencies
	@echo "Stopping services and removing volumes..."
	@cd src && docker-compose down --volumes

#!/bin/bash

# Set the API URL for the local server
prefect config set PREFECT_API_URL=http://0.0.0.0:4200/api

# Start the local Prefect server
prefect orion start --host 0.0.0.0 &

# Wait for a few seconds to ensure the server starts
sleep 5

echo "Prefect server started. Database is stored at: ~/.prefect/orion.db"
echo "To reset the database, run: prefect orion database reset"

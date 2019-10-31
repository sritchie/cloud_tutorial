#!/bin/bash

# Submit a remote job
python3 submit_with_api.py --config_file=experiment_config.yaml --docker remote

# Run a local job via Docker
#python3 submit_with_api.py --config_file=experiment_config.yaml --docker local


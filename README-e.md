# 2025QC-Reservoir
Validation Support for Evaluating the Effectiveness of Reservoir Computing

## Overview
As part of NEDO's (New Energy and Industrial Technology Development Organization) "Technical Survey Project on Validating the Effectiveness of Reservoir Computing Technology," we collect field data and perform trial analysis using our proprietary library to evaluate its effectiveness.

## Scope of Work
We examine methods for applying reservoir computing in the household service robot domain, create validation data, and evaluate the technology's effectiveness using our proprietary library.

Specifically, the following tasks are performed:
- Development of path prediction program for moving persons using reservoir computing
- Comparative study with conventional methods in home environments
- Integration study of QuantumCore's library

## Deliverables
- Validation results report
- Validation data
- Validation program and manual (complete set)

## TID -> QC
[ROS 2 Leg Finder + Path Prediction (Dockerized)](https://gitlab.com/tidbots/path_prediction)

```
git clone --recursive git@github.com:tidbots/2025QC-Reservoir.git
```

## Documentation

See [docs/](docs/) for detailed documentation.

- [path_prediction Overview](docs/path_prediction-e.md) - System overview
- [leg_finder Details](docs/path_prediction_leg_finder-e.md) - Leg detection algorithm
- [ESN Path Prediction Details](docs/path_prediction_esn-e.md) - ESN algorithm
- [Deployment](docs/path_prediction_deployment-e.md) - Docker configuration

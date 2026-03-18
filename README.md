# BPPO Project 2

This repository contains a discrete-event business process simulation built around a BPMN/Petri net process model. The project combines process mining, stochastic simulation, routing logic, and resource-allocation strategies to evaluate operational scenarios and optimization approaches.

## Overview

The simulation pipeline includes:

- BPMN-to-Petri-net execution with `pm4py`
- Arrival generation based on historical event-log data
- Basic and advanced activity duration models
- Resource planning strategies, including heuristic and optimization-based assignment
- Scenario evaluation and metric aggregation

The main entry point is [`simulation/main.py`](simulation/main.py).

## Repository Structure

```text
simulation/
  engine/             Core discrete-event simulation engine and logging
  resource_manager/   Resource allocation logic, planners, metrics, and tests
  routing/            Basic and advanced branching logic
  spawner/            Case arrival generation
  timing/             Processing-time modelling and evaluation utilities
  Evaluation/         Scenario evaluation scripts and analysis helpers
  process_model.bpmn  Process model used by the simulation

data/
  bpi-chall.xes       Source event log used for training and simulation inputs

optimization_artifacts/
  Generated summaries, rankings, appendices, and plots

eval_logs_scenario_15runs_1month/
  Scenario evaluation outputs
```

## Requirements

Install the Python dependencies listed in [`requirements.txt`](requirements.txt):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project expects the following input assets to be available:

- [`data/bpi-chall.xes`](data/bpi-chall.xes)
- [`simulation/process_model.bpmn`](simulation/process_model.bpmn)
- Precomputed model artifacts such as [`quantile_models.pkl`](quantile_models.pkl) and [`fitted_distributions.pkl`](fitted_distributions.pkl)

## Running the Simulation

Run the main simulation from the repository root:

```bash
python -m simulation.main
```

This will:

- load the BPMN model and convert it to a Petri net
- read the historical event log from `data/bpi-chall.xes`
- mine the organizational model
- generate arrivals for the configured simulation window
- execute the simulation
- write metrics to [`simulation_metrics.csv`](simulation_metrics.csv)
- write the generated event log to `simulation_log.csv`

## Evaluation

Scenario-based evaluation scripts are located in [`simulation/Evaluation`](simulation/Evaluation). A representative entry point is:

```bash
python -m simulation.Evaluation.scenario_evaluation
```

This script runs repeated scenario simulations, stores raw and cleaned logs, and aggregates impact metrics in `eval_logs_scenario_15runs_1month/`.

## Testing

The repository includes test modules under [`simulation/resource_manager`](simulation/resource_manager) and [`simulation/testing`](simulation/testing). Where applicable, run them with:

```bash
pytest
```

## Outputs

The repository already contains generated outputs and artifacts, including:

- simulation logs such as [`simulation_log_batch.csv`](simulation_log_batch.csv)
- summary metrics such as [`simulation_metrics.csv`](simulation_metrics.csv)
- optimization reports in [`optimization_artifacts`](optimization_artifacts)

## Notes

- Paths in the code assume the repository is run from its root directory.
- Several scripts depend on pre-generated model files that are already included in the repository.
- The project is organized around experimentation and evaluation, so some folders contain result artifacts in addition to source code.

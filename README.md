# Sengupta Research Project

## Purpose
This project implements data generation and loading utilities for anomaly detection research, specifically:
- Synthetic sensor data generation with various attack scenarios
- HAI (Hardware-in-the-Loop) dataset loading and preprocessing
- Evaluation metrics for detection performance

## Setup

### Prerequisites
- Python 3.11.15+
- Virtual environment (.venv) is already set up

### Installation
The virtual environment is already set up with all required dependencies:
```bash
# Activate the virtual environment
.venv\Scripts\activate

# All dependencies are already installed
pip list
```

## Running Tests

Run all specification tests from the project root:
```bash
.venv\Scripts\python.exe -m pytest tests/spec -v
```

### Smoke Tests
```bash
# Test synthetic data generator
.venv\Scripts\python.exe -c "from src.data.synthetic import SensorDataGenerator; sg = SensorDataGenerator(n_sensors=5); data, meta = sg.generate(n_timesteps=10000, attack_type='step')"

# Test HAI data loader
.venv\Scripts\python.exe -c "from src.data.hai_loader import HAILoader; hl = HAILoader(); train, val, test = hl.load()"
```

## Directory Structure

- `src/data/synthetic.py` - SensorDataGenerator class
- `src/data/hai_loader.py` - HAILoader class
- `src/detection/base.py` - BaseDetector abstract class
- `src/evaluation/metrics.py` - Evaluation metrics
- `src/utils/` - Utility modules (to be implemented)
- `tests/spec/` - Specification tests (locked, immutable)
- `tests/dev/` - Integration tests (to be implemented)
- `results/` - Output files (plots, reports)
- `data/` - Raw data files

## Data Generation

The SensorDataGenerator creates synthetic sensor data with:
- Low-frequency sine wave baselines
- Gaussian noise
- Various attack types (step, ramp, periodic, coordinated)
- Configurable correlation between sensors

## HAI Dataset

The HAI dataset contains:
- 59 sensor measurements
- Attack labels
- Timestamps in epoch seconds
- Four CSV files: train1.csv, train2.csv, test1.csv, test2.csv

## Results

Generated outputs are saved to the `results/` directory:
- `synthetic_delta_x_plot.png` - Plots showing attack injections
- `hai_loader_report.md` - Data split statistics and analysis

## License
MIT
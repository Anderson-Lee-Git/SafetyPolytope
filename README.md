# Learning Safety Constraints for LLMs

This repository contains the implementation of the paper Learning Safety Constraints for LLMs.

## Installation

### Prerequisites
- Conda (Miniconda or Anaconda)
- Git

### Setup Instructions

1. Clone the repository:
```bash
git clone git@github.com:lasgroup/SafetyPolytope.git
cd SafetyPolytope
```

2. Create and activate a new conda environment:
```bash
conda create -n sap python=3.10 -y
conda activate sap
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

To run the BeaverTails pipeline with default settings:

```bash
python src/crlhf/polytope/run_beaver_pipeline.py \
    --model_path=Qwen/Qwen2-1.5B-Instruct \
    --mode=local \
    --reduced_data
```

The `--reduced_data` flag will run the pipeline with reduced data. Remove this flag if you want to train on the full dataset.

## TODOs
- [ ] Add Harmbench pipeline code

## License
MIT License.

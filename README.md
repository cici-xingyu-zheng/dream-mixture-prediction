# Team Tamarin: olfectory mixture prediction

This repo is used to reproduce the submitted result of team Tamarin for:
- [DREAM Olfactory Mixtures Prediction Challenge](https://www.synapse.org/Synapse:syn53470621/wiki/627282)

Detailed model description see:
- [Project wiki](https://www.synapse.org/Synapse:syn61846935/wiki/629226)

## Setup Instructions

### Using Conda

**Create and activate the conda environment**:

    ```sh
    conda env create -f environment.yml
    conda activate dreamdf
    ```

### Using pip

1. **Create and activate a virtual environment**:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```


2. **Install dependencies**:

    ```sh
    pip install -r requirements.txt
    ```

## Reproduce the submitted model

Run the notebook: `submitted_model.ipynb`

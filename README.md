# CT Classification - Luna Dataset

## Introduction
This repository contains a CT classification project, which uses the Luna dataset for lung nodule detection training. The data can be independently downloaded from Luna for competition use.

## Structure
The repository consists of the following key folders and files:

1. **code**: Contains the source code for the project.

2. **data-unversioned/cache**: Contains cached data files for faster loading.

3. **evaluationScript**: Contains scripts for evaluating the performance of the model.

4. **luna**: Contains scripts related to the Luna dataset.

5. **seg-lungs-LUNA16**: Contains scripts for lung segmentation.

6. **util**: Contains utility scripts used across the project.

7. **123.py**: A Python script (the purpose of this script should be described in more detail).

8. **data.py, dsets.py, model.py, prepcache.py, screencts.py, training.py, vis.py**: Python scripts for various stages of the machine learning pipeline.

9. **annotations.csv, candidates.csv, candidates_V2.csv, sampleSubmission.csv**: CSV files related to the Luna dataset.

## Quick Start
1. First, you need to install Python and pip. If you haven't installed them yet, please refer to the following links:
   - [Install Python](https://www.python.org/downloads/)
   - [Install pip](https://pip.pypa.io/en/stable/installation/)

2. Clone this repository to your local machine:
    ```
    git clone https://github.com/your-repository/ct-classification.git
    ```

3. Switch to the directory of the repository you just cloned:
    ```
    cd ct-classification
    ```

4. Install the required Python libraries (if a `requirements.txt` file is provided):
    ```
    pip install -r requirements.txt
    ```

5. Run the main script (replace `main_script.py` with the actual main script):
    ```
    python main_script.py
    ```

## Features
- **Lung Nodule Detection**: The project uses the Luna dataset to train a model for lung nodule detection.

## Note
Please ensure you have the necessary Python libraries installed before running the scripts. If a `requirements.txt` file is provided, you can install the requirements via pip:
```
pip install -r requirements.txt
```
You also need to download the Luna dataset independently for use in this project. Please refer to the Luna dataset documentation for more details.

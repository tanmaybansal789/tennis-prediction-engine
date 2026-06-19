# Tennis prediction engine

This project is a machine learning pipeline for predicting the outcome of professional tennis matches. It uses historical ATP match data to train a model that can predict the winner of a match based on various player and match features.

## Structure

The project is organized into the following directories:

- `data_processing/`: Contains the Python scripts for the data processing pipeline.
- `tennis_atp/`: Contains the raw ATP match data in CSV format, with one file per year.
- `processed_data/`: The output directory for the processed features and labels.
- `models/`: Stores the trained machine learning models.
- `notebooks/`: Jupyter notebooks for exploratory data analysis.
- `documents/`: Supplementary documents - initial proposal and a documentation of the process.
- `tennis_prediction_engine/`: Contains the core RandomForest prediction engine, along with the command-line model training script.

## Data processing pipeline

The data processing pipeline is orchestrated by `data_processing/main.py` and is defined in `data_processing/tennis_pipeline.py`. It consists of the following steps:

1.  **Data loading**: The pipeline loads raw ATP match data from the `tennis_atp/` directory for the years specified in `data_processing/config.py`.

2.  **Feature engineering**: New features are created from the raw data in `data_processing/feature_engineer.py`, including
    *   Calculating Elo ratings for each player.
    *   Creating rolling statistics for various performance metrics (e.g., serve win ratio, breakpoint conversion).
    *   Calculating surface-specific win ratios.
    *   Determining recent form using an exponential moving average.
    *   Calculating win streaks.

3.  **Data encoding**: Categorical features are converted into numerical format in `data_processing/data_encoder.py`. This includes encoding features like player hand, entry type, surface, and tournament level.

4.  **Output**: The final processed features and labels are saved to the `processed_data/` directory as `tennis_model_features.csv` and `tennis_model_labels.csv`.

## How to run

To run the data processing pipeline, execute the `main.py` script in the `data_processing` directory:

```bash
python data_processing/main.py
```

## Configuration

The data processing pipeline can be configured by editing `data_processing/config.py`. This file allows you to set:

-   Input and output directories.
-   The range of years to use for training.
-   Parameters for Elo calculation.
-   Configuration for rolling statistics and other feature engineering parameters.
-   Mappings for categorical feature encoding.

## Model training and prediction
Trained models are stored in the `models/` directory. The model training script is located in `tennis_prediction_engine/model.py`. You can train a new model by running:
```bash
python3 tennis_prediction_engine/model.py --mode train --features processed_data/tennis_model_features.csv --labels processed_data/tennis_model_labels.csv
```


The following options are availble
```
options:
  -h, --help            show this help message and exit
  --mode {train,evaluate,predict}
                        Operation mode
  --features FEATURES   Path to features CSV file
  --labels LABELS       Path to labels CSV
  --model MODEL         Path to saved model if evaluating/predicting
  --test-size TEST_SIZE
                        How much of the data to use as the test set
  --random-state RANDOM_STATE
                        Pseudorandom seed
```
# Tennis prediction engine

This project is a machine learning pipeline for predicting the outcome of professional tennis matches. It uses historical ATP match data to train a model that can predict the winner of a match based on various player and match features.

## Structure

The project is organized into the following directories:

- `data_processing/`: Contains the Python scripts for the data processing pipeline.
- `tennis_atp/`: Contains the raw ATP match data in CSV format, with one file per year.
- `processed_data/`: The output directory for the processed features and labels.
- `models/`: Stores the trained machine learning models.
- `notebooks/`: Jupyter notebooks for data exploration and preparation.
- `documents/`: For any supplementary documents.
- `tennis_prediction_engine/`: Likely contains the core prediction engine logic (though not explored in this README).

## Data processing pipeline

The data processing pipeline is orchestrated by `data_processing/main.py` and is defined in `data_processing/tennis_pipeline.py`. It consists of the following steps:

1.  **Data Loading**: The pipeline loads raw ATP match data from the `tennis_atp/` directory for the years specified in `data_processing/config.py`.

2.  **Feature Engineering**: New features are created from the raw data in `data_processing/feature_engineer.py`. This includes:
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

The processed data can then be used to train a machine learning model to predict match outcomes. The training and prediction logic is expected to be in the `tennis_prediction_engine/` directory, which is not detailed in this README. The trained models are saved in the `models/` directory.

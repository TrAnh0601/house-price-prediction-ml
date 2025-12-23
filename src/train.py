import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import config
from preprocessing import get_processor

# v1.2 Remove outliers
def remove_outliers(df):
    outliers = df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index
    return df.drop(outliers)

def run_training():
    # Load data
    df = pd.read_csv(config.TRAIN_PATH)

    # v1.2 Remove outliers
    df = remove_outliers(df)

    X = df.drop([config.TARGET, 'Id'], axis=1)
    y = np.log1p(df[config.TARGET])

    # Initialize the complete model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', get_processor()),
        ('model', Ridge())
    ])

    # Define the parameter grid
    param_grid = {'model__alpha': [17.928, 17.929, 17.930, 17.931, 17.932]}

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=1
    )

    # Run the search
    grid_search.fit(X, y)

    # Result
    best_alpha = grid_search.best_params_['model__alpha']
    best_rmse = np.sqrt(-grid_search.best_score_)

    print(f"Best Alpha found: {best_alpha}")
    print(f"Best CV RMSE: {best_rmse:.5f}")

    # Return best model found
    return grid_search.best_estimator_

def make_submission(model_pipeline):
    test_df = pd.read_csv(config.TEST_PATH)
    test_X = test_df.drop(['Id'], axis=1)

    # Predict and transform back log-scale to original price
    log_predictions = model_pipeline.predict(test_X)
    predictions = np.expm1(log_predictions)

    # Format according to Kaggle requirement
    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'SalePrice': predictions
    })

    submission.to_csv(config.SUBMISSION_PATH, index=False)

if __name__ == "__main__":
    # Run the process
    best_pipeline = run_training()
    make_submission(best_pipeline)

# House Prices Prediction: Ridge Regression
Predicting residential home prices in Ames, Iowa, using a robust Scikit-Learn Pipeline.

## Performance
CV RMSE: 0.13243  
Optimal Alpha: 17.929  
Rank: Top 20% on Kaggle (approx)

## Key Implementation
1. Outlier Removal: Dropped instances with GrLivArea > 4000 and SalePrice < 300,000.
2. Target Transformation: Used np.log1p for training and np.expm1 for inference to stabilize variance.
3. Automated Pipeline:
   - Numerical: Median Imputation + Standard Scaling.
   - Categorical: Constant Imputation + One-Hot Encoding.
4. Tuning: Systematic grid search for Ridge regularization strength ($\alpha$).

## Structure
preprocessing.py: Automated ColumnTransformer pipeline.  
train.py: Model training, CV, and submission logic.  
config.py: File paths and global constants.

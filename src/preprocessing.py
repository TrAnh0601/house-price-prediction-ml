from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_processor():
    # Pipeline for numerical features:
    # 1. Fill missing values with median
    # 2. Scale features to mean=0 and std=1
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical features:
    # 1. Fill mising values with a constant 'NA'
    # 2. Convert text labels to one-hot binary vectors
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Use make_column_selector to automatically detect data types at runtime.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, make_column_selector(dtype_include=['number'])),
            ('cat', cat_transformer, make_column_selector(dtype_include=['object', 'category']))
        ]
    )

    return preprocessor

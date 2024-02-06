from flask import Flask, request, jsonify
import pandas as pd
from joblib import load
import json
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
# Assuming the rest of your script is as provided, including the import statements,
# definition of DummiesTransformer, add_hold_days_category, the pipeline setup, and the add_missing_columns function.
import os
import sys



app = Flask(__name__)

# Load the model
model_l = load('lightgbm_model.pkl')
expected_columns = ['PART_NUMBER', 'TRANSACTION_QTY', 'ROUTE', 'HOLD_DAYS', 'PKG_CODE', 'TESTER', 'DIETYPE', 'PACKAGETYPE', 'DOWNTIME_DAYS', 'WASHOLD_0', 'WASHOLD_1', 'TESTERTYPE_ASL1K', 'TESTERTYPE_ASL3K', 'TESTERTYPE_ASL4K', 'TESTERTYPE_DOT400', 'TESTERTYPE_DUO', 'TESTERTYPE_EAGLE88', 'TESTERTYPE_HP', 'TESTERTYPE_KTS', 'TESTERTYPE_KVDM2', 'TESTERTYPE_LTX', 'TESTERTYPE_LTXMX', 'TESTERTYPE_MAV1', 'TESTERTYPE_MAV2', 'TESTERTYPE_MICROFLEX', 'TESTERTYPE_NOISE', 'TESTERTYPE_QUARTET', 'TESTERTYPE_RFX', 'TESTERTYPE_SC212', 'TESTERTYPE_STS5020', 'TESTERTYPE_SZ', 'TESTERTYPE_TERA360Z', 'TESTERTYPE_TERCAT', 'TESTERTYPE_TERFLEX', 'TESTERTYPE_TERJ750', 'TESTERTYPE_TERMAG2', 'HOLD_DAYS_CATEGORY']


# Create a function to add missing columns with 0s
def add_missing_columns(df, expected_columns):
    missing_cols = set(expected_columns) - set(df.columns)
    for c in missing_cols:
        df[c] = 0
    # Ensure the order of columns matches the model's expectations
    df = df[expected_columns]
    return df

def pipeline(sample_data):
    
    # Custom transformer for applying pd.get_dummies
    class DummiesTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, columns, dtype=float):
            self.columns = columns
            self.dtype = dtype
            
        def fit(self, X, y=None):
            return self  # Nothing to fit
        
        def transform(self, X):
            # Applying pd.get_dummies to the specified columns
            return pd.get_dummies(X, columns=self.columns, dtype=self.dtype)

    # Custom function for transforming HOLD_DAYS
    def add_hold_days_category(X):
        X_new = X.copy()  # Copy to avoid changing the original data
        X_new['HOLD_DAYS_CATEGORY'] = (X_new['HOLD_DAYS'] != 0).astype(int)
        return X_new


    # Instantiate the custom transformers
    dummies_transformer = DummiesTransformer(columns=['WASHOLD', 'TESTERTYPE'])
    hold_days_transformer = FunctionTransformer(add_hold_days_category)
    # Create the pipeline
    _pipeline = Pipeline(steps=[
        ('dummies', dummies_transformer),
        ('hold_days_category', hold_days_transformer)
    ])
    
    df = _pipeline.fit_transform(sample_data)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    return df


@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    input_data = request.get_json(force=True)
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Select and reorder the relevant columns as per training
    data = input_df[["PART_NUMBER", "TRANSACTION_QTY", "ROUTE", "HOLD_DAYS", "PKG_CODE", "TESTER", 
                     "TESTERTYPE", "WASHOLD", "DIETYPE", "PACKAGETYPE", "DOWNTIME_DAYS"]]
    
    # Transform the data using the pipeline
    transformed_df = pipeline(data)
    
    # Add missing columns to match the training dataframe
    final_df = add_missing_columns(transformed_df, expected_columns)
    
    # Make a prediction
    prediction = model_l.predict(final_df)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction.tolist()[0]})

# if __name__ == '__main__':
#     app.run(debug=False, host='0.0.0.0', port=5000)

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
sample_data = {'PART_NUMBER': '89-32KHZ#S08', 'TRANSACTION_QTY': 10804, 'ACT_CT': 23.2013, 'ROUTE': 'P41421_WIP', 'HOLD_DAYS': 0.0, 'PLAN_CT': 0.692132082, 'PKG_CODE': 'W20#H2', 'TESTER': 'MAV1064', 'BU': "nan", 'TESTERTYPE': 'MAV1', 'EXCLUDE_FLAG': 'NONTESTFLOW', 'PLAN_CT_100PERC': 0.408357928, 'TESTSTEPCNT': 3.0, 'PLAN_CT_100PERC2H1D': 1.658357928, 'WASHOLD': 0, 'HOLDREASONS': "nan", 'PLAN_CT_100PERC1D': 1.408357928, 'OEEFACTOR': 0.59, 'OEEFACTORSRC': 'CPT_HIST', 'DIETYPE': 'MO78A-0A', 'PACKAGETYPE': 'SOIC (W)', 'DOWNTIME_DAYS': 0.877430556, 'QUEUE_DAYS': "nan", 'HOLIDAY_HRS': "nan"}
input_data = pd.DataFrame([sample_data])

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
    pipeline = Pipeline(steps=[
        ('dummies', dummies_transformer),
        ('hold_days_category', hold_days_transformer)
    ])
    
    df = pipeline.fit_transform(data)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    return df


data = input_data[["PART_NUMBER","TRANSACTION_QTY",
            "ROUTE","HOLD_DAYS","PKG_CODE","TESTER",
            "TESTERTYPE","WASHOLD","DIETYPE","PACKAGETYPE",
            "DOWNTIME_DAYS"]]



df = pipeline.fit_transform(data)

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category')

from joblib import load

expected_columns = ['PART_NUMBER', 'TRANSACTION_QTY', 'ROUTE', 'HOLD_DAYS', 'PKG_CODE', 'TESTER', 'DIETYPE', 'PACKAGETYPE', 'DOWNTIME_DAYS', 'WASHOLD_0', 'WASHOLD_1', 'TESTERTYPE_ASL1K', 'TESTERTYPE_ASL3K', 'TESTERTYPE_ASL4K', 'TESTERTYPE_DOT400', 'TESTERTYPE_DUO', 'TESTERTYPE_EAGLE88', 'TESTERTYPE_HP', 'TESTERTYPE_KTS', 'TESTERTYPE_KVDM2', 'TESTERTYPE_LTX', 'TESTERTYPE_LTXMX', 'TESTERTYPE_MAV1', 'TESTERTYPE_MAV2', 'TESTERTYPE_MICROFLEX', 'TESTERTYPE_NOISE', 'TESTERTYPE_QUARTET', 'TESTERTYPE_RFX', 'TESTERTYPE_SC212', 'TESTERTYPE_STS5020', 'TESTERTYPE_SZ', 'TESTERTYPE_TERA360Z', 'TESTERTYPE_TERCAT', 'TESTERTYPE_TERFLEX', 'TESTERTYPE_TERJ750', 'TESTERTYPE_TERMAG2', 'HOLD_DAYS_CATEGORY']

# Create a function to add missing columns with 0s
def add_missing_columns(df, expected_columns):
    missing_cols = set(expected_columns) - set(df.columns)
    for c in missing_cols:
        df[c] = 0
    # Ensure the order of columns matches the model's expectations
    df = df[expected_columns]
    return df

# Apply the function to your DataFrame before prediction
df = add_missing_columns(df, expected_columns)
# Correct way to access th
model_l = load('xgboost_model.pkl')
y_pred = model_l.predict(df)
# y_pred = model_l.predict(sample_row)

print(f"Predicted Output: {y_pred[0]}")


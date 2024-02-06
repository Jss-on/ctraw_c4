from joblib import load
import shap
import matplotlib.pyplot as plt
# model_xgb = xgb.XGBRegressor(**best_params_xgb)
# model_xgb.fit(X,y)
# Save the model to a file
# dump(model_xgb, 'xgboost_model.pkl')

model_xgb = load('xgboost_model.pkl')
# Initialize the JS visualization in the notebook (only for Jupyter environments; skip otherwise)
# shap.initjs()

# Create a SHAP explainer object
explainer = shap.Explainer(model_xgb)
shap_values = explainer(X)

# Summarize the effects of all the features
shap.summary_plot(shap_values, X)
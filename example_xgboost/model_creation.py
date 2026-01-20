import xgboost as xgb

model = xgb.XGBRegressor()

# Fit the model

model.save_model('checkpoint.json')  # Save the model
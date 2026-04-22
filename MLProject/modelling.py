import pandas as pd
import mlflow
import mlflow.sklearn
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="housepricessdataset_preprocessing")
args = parser.parse_args()

data_path = args.data_path

if "MLFLOW_TRACKING_URI" not in os.environ:
    mlflow.set_tracking_uri("file:./mlruns")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

X_train = pd.read_csv(f'{data_path}/X_train.csv')
X_test = pd.read_csv(f'{data_path}/X_test.csv')
y_train = pd.read_csv(f'{data_path}/y_train.csv')
y_test = pd.read_csv(f'{data_path}/y_test.csv')

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

mlflow.sklearn.autolog()

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

try:
    from sklearn.metrics import root_mean_squared_error
    rmse = root_mean_squared_error(y_test, y_pred)
except ImportError:
    from sklearn.metrics import mean_squared_error
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)

mlflow.log_metric("rmse_manual", rmse)
mlflow.log_metric("r2_manual", r2)

print("Training selesai!")

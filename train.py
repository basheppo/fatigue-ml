import pandas as pd 
import valohai
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import os
import json
import joblib
from funcs import get_scores

    

lr_model= LinearRegression()
train_dataset_path = valohai.inputs('train_dataset').path()
data = pd.read_csv(train_dataset_path)
y_train = data['Fatigue']
X_train = data.drop(columns=['Sl. No.', 'Fatigue'])
print(X_train)
lr_model.fit(X_train, y_train)
r2_s_train , mae_train , y_pred = get_scores(lr_model,X_train,y_train)
print(r2_s_train)
metrics = {
    "r2_score": r2_s_train,
    "mae": mae_train
}
output_dir = os.getenv('VH_OUTPUTS_DIR', '.')
output_path = os.path.join(output_dir, 'metrics_train.json')
with open(output_path, 'w') as f:
    json.dump(metrics, f)

print(f"Metrics saved to {output_path}")

model_path = os.path.join(output_dir, 'model')
joblib.dump(lr_model, model_path)
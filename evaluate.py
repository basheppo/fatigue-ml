import pandas as pd 
import valohai
from sklearn.linear_model import LinearRegression
import os
import json
import joblib
from funcs import get_scores
model_path = valohai.inputs("model").path()
lr_model = joblib.load(model_path)
test_dataset_path = valohai.inputs("test_dataset").path()
data = pd.read_csv(test_dataset_path)
y_train = data['Fatigue']
X_train = data.drop(columns=['Sl. No.', 'Fatigue'])
r2_s_test , mae_test , y_pred = get_scores(lr_model,X_train,y_train)
metrics = {
    "r2_score": r2_s_test,
    "mae": mae_test
}

output_dir = os.getenv('VH_OUTPUTS_DIR', '.')
output_path = os.path.join(output_dir, 'metrics_test.json')
with open(output_path, 'w') as f:
    json.dump(metrics, f)

print(f"Metrics saved to {output_path}")
import pandas as pd 
import os
VH_INPUTS_DIR = os.getenv('VH_INPUTS_DIR')
VH_OUTPUTS_DIR = os.getenv('VH_OUTPUTS_DIR') 
if not VH_INPUTS_DIR:
    raise FileNotFoundError("Dataset not foundd")


input_path = os.path.join(VH_INPUTS_DIR, 'dataset/fatigue_dataset.csv')
data = pd.read_csv(input_path)
shuffled = data.sample(frac=1, random_state=42)

train_size = 0.8
train_index = int(len(data) * train_size)

train_df = shuffled[:train_index]
test_df = shuffled[train_index:]

print(train_df.shape, test_df.shape)
os.makedirs(VH_OUTPUTS_DIR, exist_ok=True)
train_csv = os.path.join(VH_OUTPUTS_DIR, 'train.csv')
test_csv = os.path.join(VH_OUTPUTS_DIR, 'test.csv')

train_df.to_csv(train_csv, index=False)
test_df.to_csv(test_csv, index=False)


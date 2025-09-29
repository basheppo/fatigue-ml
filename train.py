import pandas as pd 
import valohai
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

def get_scores(model, X, y):

    y_pred = model.predict(X)
    r_sq = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    print(type(model).__name__)
    print("R2-score: {:.3f}".format(r_sq))
    print("MAE: {:.2f}".format(mae))
    
    return r_sq,mae,y_pred
    

lr_model= LinearRegression()
train_dataset_path = valohai.inputs('train_dataset').path()
data = pd.read_csv(train_dataset_path)
y_train = data['Fatigue']
X_train = data.drop(columns=['Sl. No.', 'Fatigue'])
print(X_train)
lr_model.fit(X_train, y_train)
r2_s_train , mae_train , y_pred = get_scores(lr_model,X_train,y_train)
print(r2_s_train)

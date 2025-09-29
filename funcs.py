from sklearn.metrics import r2_score, mean_absolute_error

def get_scores(model, X, y):

    y_pred = model.predict(X)
    r_sq = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    print(type(model).__name__)
    print("R2-score: {:.3f}".format(r_sq))
    print("MAE: {:.2f}".format(mae))
    
    return r_sq,mae,y_pred
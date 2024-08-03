from sklearn.metrics import confusion_matrix, accuracy_score

def predict_model(model, Xtest):
    ypred = model.predict(Xtest)
    return ypred

def evaluate_model(ytest, ypred):
    conf_matrix = confusion_matrix(ytest, ypred)
    accuracy = accuracy_score(ytest, ypred)
    return conf_matrix, accuracy

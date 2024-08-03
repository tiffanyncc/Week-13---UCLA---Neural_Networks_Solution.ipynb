from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def split_and_scale_data(x, y):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=123)
    
    scaler = MinMaxScaler()
    scaler.fit(xtrain)
    
    Xtrain = scaler.transform(xtrain)
    Xtest = scaler.transform(xtest)
    
    return Xtrain, Xtest, ytrain, ytest, scaler

def train_model(Xtrain, ytrain):
    MLP = MLPClassifier(batch_size=50, max_iter=100, random_state=123, verbose=True)
    MLP.fit(Xtrain, ytrain)
    return MLP

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

def perform_grid_search(x, y):
    MLP = MLPClassifier(random_state=123)
    params = {'batch_size': [20, 30, 40, 50],
              'hidden_layer_sizes': [(2,), (3,), (3, 2)],
              'learning_rate_init': [50, 70, 100]}
    grid = GridSearchCV(MLP, params, cv=10, scoring='accuracy')
    grid.fit(x, y)
    return grid.best_params_, grid.best_score_

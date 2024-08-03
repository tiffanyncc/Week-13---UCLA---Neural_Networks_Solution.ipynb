import os

# Set the environment variable to avoid the memory leak warning
os.environ["OMP_NUM_THREADS"] = "1"

import logging
import matplotlib.pyplot as plt
from src.data.make_dataset import load_data
from src.features.build_features import preprocess_data, prepare_data
from src.models.train_model import split_and_scale_data, train_model
from src.models.predict_model import predict_model, evaluate_model
from src.models.grid_search import perform_grid_search
from src.visualization.visualize import plot_scatter, plot_distributions, plot_loss_curve
from sklearn.model_selection import cross_val_score

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def main():
    try:
        df = load_data('src/data/raw/Admission.csv')
        logging.info('Data loaded successfully.')
    except Exception as e:
        logging.error(f'Error loading data: {e}')
        return

    try:
        df = preprocess_data(df)
        logging.info('Data preprocessed successfully.')
    except Exception as e:
        logging.error(f'Error preprocessing data: {e}')
        return

    try:
        plot_scatter(df)
        logging.info('Scatterplot displayed and saved.')
    except Exception as e:
        logging.error(f'Error displaying scatterplot: {e}')

    try:
        x, y = prepare_data(df)
        Xtrain, Xtest, ytrain, ytest, scaler = split_and_scale_data(x, y)
        logging.info('Data split and scaled successfully.')
    except Exception as e:
        logging.error(f'Error splitting and scaling data: {e}')
        return

    try:
        plot_distributions(df, Xtrain)
        logging.info('Distributions plot displayed and saved.')
    except Exception as e:
        logging.error(f'Error displaying distributions plot: {e}')

    try:
        model = train_model(Xtrain, ytrain)
        logging.info('Model trained successfully.')
    except Exception as e:
        logging.error(f'Error training model: {e}')
        return
    
    try:
        # Cross-validation using cross_val_score
        cv_scores = cross_val_score(model, x, y, cv=10, scoring='accuracy')
        logging.info(f'Cross-validation scores: {cv_scores}')
        logging.info(f'Mean cross-validation score: {cv_scores.mean()}')
    except Exception as e:
        logging.error(f'Error performing cross-validation: {e}')
        return

    try:
        ypred = predict_model(model, Xtest)
        conf_matrix, accuracy = evaluate_model(ytest, ypred)
        logging.info(f'Model evaluated successfully. Accuracy: {accuracy}')
    except Exception as e:
        logging.error(f'Error predicting or evaluating model: {e}')
        return

    try:
        plot_loss_curve(model.loss_curve_)
        logging.info('Loss curve plot displayed and saved.')
    except Exception as e:
        logging.error(f'Error displaying loss curve plot: {e}')

    try:
        best_params, best_score = perform_grid_search(x, y)
        logging.info(f'Grid search completed. Best params: {best_params}, Best score: {best_score}')
    except Exception as e:
        logging.error(f'Error performing grid search: {e}')

if __name__ == '__main__':
    main()
import pandas as pd
import numpy as np
import itertools
import joblib
import json
import os
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

class ModelTrainer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.model = SGDRegressor()
        self.scaler = StandardScaler()

    def load_and_preprocess_data(self, test_size=0.2, val_size=0.25):
        all_data = pd.read_csv(self.filepath)
        columns_to_keep = ['beds', 'bathrooms', 'Cleanliness_rating', 
                           'Accuracy_rating', 'Communication_rating', 'Location_rating', 
                           'Check-in_rating', 'Value_rating', 'Price_Night']
        data = all_data[columns_to_keep]
        X = data.drop('Price_Night', axis=1)
        y = data['Price_Night']

        X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_adjusted, random_state=42)

        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def train_model(self):
        self.model.fit(self.X_train_scaled, self.y_train)

    def evaluate_model(self, X, y):
        predictions = self.model.predict(X)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        return {'RMSE': rmse, 'R2': r2}

    def predict(self, new_data):
        new_data_scaled = self.scaler.transform(new_data)
        return self.model.predict(new_data_scaled)

def custom_tune_regression_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, hyperparameters):
    best_model = None
    best_hyperparams = {}
    best_rmse = float('inf')
    performance_metrics = {}

    hyperparams_combinations = [dict(zip(hyperparameters, v)) for v in itertools.product(*hyperparameters.values())]

    for hyperparams in hyperparams_combinations:
        model = model_class(**hyperparams)
        model.fit(X_train, y_train)
        val_predictions = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_predictions))

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_hyperparams = hyperparams

    performance_metrics['validation_RMSE'] = best_rmse
    return best_model, best_hyperparams, performance_metrics

def tune_regression_model_hyperparameters(model, param_grid, X_train, y_train, cv=5):
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X_train, y_train)
    return grid_search

def save_model(model, hyperparameters, metrics, folder='models/regression/linear_regression'):
    if not os.path.exists(folder):
        os.makedirs(folder)

    model_path = os.path.join(folder, 'model.joblib')
    joblib.dump(model, model_path)

    hyperparameters_path = os.path.join(folder, 'hyperparameters.json')
    with open(hyperparameters_path, 'w') as file:
        json.dump(hyperparameters, file)

    metrics_path = os.path.join(folder, 'metrics.json')
    with open(metrics_path, 'w') as file:
        json.dump(metrics, file)

    print(f'Model, hyperparameters, and metrics saved in {folder}')

def evaluate_all_models(X_train, y_train, X_test, y_test, folder_prefix='models/regression'):
    models = {
        'decision_tree': DecisionTreeRegressor(),
        'random_forest': RandomForestRegressor(),
        'gradient_boosting': GradientBoostingRegressor()
    }

    hyperparameters = {
        'decision_tree': {'max_depth': [3, 5, 10, None]},
        'random_forest': {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 10, None]},
        'gradient_boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
    }

    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        grid_search_results = tune_regression_model_hyperparameters(model, hyperparameters[model_name], X_train, y_train)
        best_params = grid_search_results.best_params_
        best_model = grid_search_results.best_estimator_
        test_score = best_model.score(X_test, y_test)

        model_folder = os.path.join(folder_prefix, model_name)
        save_model(best_model, best_params, {'Test Score': test_score}, folder=model_folder)

def find_best_model(folder_prefix='models/regression'):
    best_model = None
    best_hyperparameters = None
    best_metrics = None
    best_score = float('-inf')

    for model_name in os.listdir(folder_prefix):
        model_folder = os.path.join(folder_prefix, model_name)
        metrics_path = os.path.join(model_folder, 'metrics.json')
        with open(metrics_path, 'r') as file:
            metrics = json.load(file)

        if metrics.get('Test Score', float('-inf')) > best_score:
            best_score = metrics['Test Score']
            model_path = os.path.join(model_folder, 'model.joblib')
            best_model = joblib.load(model_path)

            hyperparameters_path = os.path.join(model_folder, 'hyperparameters.json')
            with open(hyperparameters_path, 'r') as file:
                best_hyperparameters = json.load(file)

            best_metrics = metrics

    return best_model, best_hyperparameters, best_metrics

if __name__ == "__main__":
    filepath = 'clean_tabular_data.csv'
    trainer = ModelTrainer(filepath)
    trainer.load_and_preprocess_data()

    evaluate_all_models(trainer.X_train_scaled, trainer.y_train, trainer.X_test_scaled, trainer.y_test)

    best_model, best_hyperparameters, best_metrics = find_best_model()
    print("Best Model:", best_model)
    print("Best Model Hyperparameters:", best_hyperparameters)
    print("Best Model Performance Metrics:", best_metrics)

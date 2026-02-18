import pandas as pd
import numpy as np
import AnalyticsFunctions as AF
from AnalyticsFunctions import custom_train_test_split
from torch import conv1d
from sklearn.svm import SVR
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import *
import matplotlib

matplotlib.use('TkAgg')  # Use the TkAgg backend for plotting

# load data
data = pd.read_excel('Blood_glucose_spectral_datasets_multi_participents.xlsx', header=0, index_col=0)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = custom_train_test_split(X.values.astype(np.float32), y.values.astype(np.float32),
                                                           test_size=0.3, method='SPXY')
X_train, X_valid, y_train, y_valid = custom_train_test_split(X_train, y_train, test_size=0.15, method='SPXY')
print(X_train.shape, X_valid.shape, X_test.shape)


def objective(trial):
    # Define hyperparameters to optimize
    print(f"Trial {trial.number} started")
    params = {
        'preprocess': {
            'window_size': 2 * trial.suggest_int('window_size', 100, 200) + 1
        },
        'feature_selection': {
            'remove_feat_ratio': trial.suggest_float('remove_feat_ratio', 0.15, 0.2)
        },
        'dimensionality_reduction': {
            'n_components': trial.suggest_int('n_components', 5, 100)
        },

        'model': {'kernel': trial.suggest_categorical('SVR_kernel', ['linear', 'rbf']),
                  'C': trial.suggest_float("SVR_c", 300, 400, log=True),
                  'epsilon': trial.suggest_float('SVR_epsilon', 0.04, 0.05, log=True),
                  'degree': trial.suggest_int('SVR_degree', 1, 4),
                  'gamma': trial.suggest_float("SVR_gamma", 5, 10, log=True)}
    }

    # Preprocessing
    X_train_proc, X_valid_proc = X_train.copy(), X_valid.copy()
    X_train_proc, X_valid_proc, y_train_proc, y_valid_proc = AF.move_avg(
        X_train_proc, X_valid_proc, y_train, y_valid,
        window_size=params['preprocess']['window_size']
    )

    # Feature selection
    X_train_proc, X_valid_proc, y_train_proc, y_valid_proc = AF.remove_high_variance_and_normalize(
        X_train_proc, X_valid_proc, y_train_proc, y_valid_proc,
        remove_feat_ratio=params['feature_selection']['remove_feat_ratio']
    )

    # Dimensionality reduction
    X_train_proc, X_valid_proc, y_train_proc, y_valid_proc = AF.pca(
        X_train_proc, X_valid_proc, y_train_proc, y_valid_proc,
        n_components=params['dimensionality_reduction']['n_components']
    )

    # Train model
    model_params = params['model'].copy()
    if model_params['kernel'] != 'poly':
        model_params.pop('degree')

    model = SVR(**model_params, max_iter=10000)
    model.fit(X_train_proc, y_train_proc)

    # Predict and calculate evaluation metrics
    y_pred = model.predict(X_valid_proc)
    rmse = np.sqrt(mean_squared_error(y_valid_proc, y_pred))
    r2 = 1 - np.sum((y_valid_proc - y_pred) ** 2) / np.sum((y_valid_proc - np.mean(y_valid_proc)) ** 2)

    # Optimization objective (minimize RMSE and maximize R2)
    # Use a weighted sum instead of division since R2 can be negative
    return rmse


# Create study object and optimize
study = optuna.create_study(direction='minimize')

import argparse

parser = argparse.ArgumentParser(description='Optimize SVR model')
parser.add_argument('--n_trials', type=int, default=100, help='Number of optimization trials')
args = parser.parse_args()

study.optimize(objective, n_trials=args.n_trials)

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Get best parameters
best_params = {
    'preprocess': {
        'window_size': 2 * trial.params['window_size'] + 1
    },
    'feature_selection': {
        'remove_feat_ratio': trial.params['remove_feat_ratio']
    },
    'dimensionality_reduction': {
        'n_components': trial.params['n_components']
    },
    'model': {
        'kernel': trial.params['SVR_kernel'],
        'C': trial.params['SVR_c'],
        'epsilon': trial.params['SVR_epsilon'],
        'gamma': trial.params['SVR_gamma'],
        'degree': trial.params['SVR_degree']
    }

}

# Make final predictions using best parameters
X_train_final, X_test_final = X_train.copy(), X_test.copy()

# Apply preprocessing with best parameters
X_train_final, X_test_final, y_train, y_test = AF.move_avg(
    X_train_final, X_test_final, y_train, y_test,
    window_size=best_params['preprocess']['window_size']
)

# Apply feature selection with best parameters
X_train_final, X_test_final, y_train, y_test = AF.remove_high_variance_and_normalize(
    X_train_final, X_test_final, y_train, y_test,
    remove_feat_ratio=best_params['feature_selection']['remove_feat_ratio']
)

# Apply dimensionality reduction with best parameters
X_train_final, X_test_final, y_train, y_test = AF.pca(
    X_train_final, X_test_final, y_train, y_test,
    n_components=best_params['dimensionality_reduction']['n_components']
)

# Train final model
final_model_params = best_params['model'].copy()
if final_model_params['kernel'] != 'poly':
    final_model_params.pop('degree')
final_model = SVR(**final_model_params)
final_model.fit(X_train_final, y_train)

# Final predictions
y_train_pred = final_model.predict(X_train_final)
y_pred = final_model.predict(X_test_final)

results = AF.get_regression_metrics(y_test, y_pred)
print(results)

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Create scatter plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, label='Training Data', alpha=0.5)
plt.scatter(y_test, y_pred, label='Test Data', alpha=0.5)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()

# Create RMSE distribution plot
plt.subplot(1, 2, 2)
# Calculate absolute errors for each point
train_errors = np.abs(y_train - y_train_pred)
test_errors = np.abs(y_test - y_pred)

# Plot training data
plt.scatter(y_train, train_errors, alpha=0.6, label='Training Data')
# Plot test data 
plt.scatter(y_test, test_errors, alpha=0.6, label='Test Data')

# Calculate and plot mean RMSE lines
train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
test_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
train_std = np.std(train_errors)
test_std = np.std(test_errors)

plt.axhline(y=train_rmse, color='r', linestyle='--', label=f'Train RMSE: {train_rmse:.5f}')
plt.axhline(y=train_rmse + train_std, color='g', linestyle=':', label=f'Train Mean ± Std')
plt.axhline(y=train_rmse - train_std, color='g', linestyle=':')

# Find and annotate worst points
worst_train_idx = np.argsort(train_errors)[-3:]
for idx in worst_train_idx:
    plt.annotate(f'({y_train[idx]:.2f}, {train_errors[idx]:.5f})',
                 (y_train[idx], train_errors[idx]),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.xlabel('Measured Values')
plt.ylabel('RMSE Errors')
plt.title('RMSE Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

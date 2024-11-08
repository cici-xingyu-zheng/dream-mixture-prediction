import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.utils import *

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline, Pipeline

from scipy import stats
from scipy.optimize import minimize
from scipy.stats import pearsonr
from scipy.stats import ks_2samp

import xgboost as xgb


best_rf_dense = {'n_estimators': 500, 'min_samples_split': 2, 'max_features': 'sqrt', 'max_depth': 20, 'bootstrap': True}
best_rf_sparse = {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 0.5, 'max_depth': 30, 'bootstrap': True}


sns.set_style('ticks')

def evaluate_fold(y_true, y_pred):
    '''
    Evaluate the performance of a model (RMSE and Correlation) on a fold
    '''
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    corr, _ = pearsonr(y_true, y_pred)
    return rmse, corr

def stacking_ensemble_cv(X_dense, X_sparse, y, base_model_dense, base_model_sparse, meta_models, n_folds=10, seed=314159):
    '''
    Perform stacking ensemble cross-validation on the base and meta models
    '''
    original_indices = np.arange(X_dense.shape[0] // 2)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    dense_preds = np.zeros(len(y))
    sparse_preds = np.zeros(len(y))
    meta_preds = {name: np.zeros(len(y)) for name in meta_models.keys()}
    
    fold_results = {
        'dense_model': {'RMSE': [], 'Correlation': [], 'RMSE_avg': [], 'Correlation_avg': []},
        'sparse_model': {'RMSE': [], 'Correlation': [], 'RMSE_avg': [], 'Correlation_avg': []}
    }
    for name in meta_models.keys():
        fold_results[f'meta_model_{name}'] = {'RMSE': [], 'Correlation': [], 'RMSE_avg': [], 'Correlation_avg': []}
    
    for train_index, val_index in kf.split(original_indices):
        train_index_coupled = np.sort(np.column_stack((2*train_index, 2*train_index+1)).flatten())
        val_index_coupled = np.sort(np.column_stack((2*val_index, 2*val_index+1)).flatten())
        
        X_dense_train, X_dense_val = X_dense[train_index_coupled], X_dense[val_index_coupled]
        X_sparse_train, X_sparse_val = X_sparse[train_index_coupled], X_sparse[val_index_coupled]
        y_train, y_val = y[train_index_coupled], y[val_index_coupled]
        
        # Train and predict with base models
        base_model_dense.fit(X_dense_train, y_train)
        base_model_sparse.fit(X_sparse_train, y_train)
        
        dense_preds[val_index_coupled] = base_model_dense.predict(X_dense_val)
        sparse_preds[val_index_coupled] = base_model_sparse.predict(X_sparse_val)
        
        # Evaluate base models for this fold (non-averaged)
        rmse, corr = evaluate_fold(y_val, dense_preds[val_index_coupled])
        fold_results['dense_model']['RMSE'].append(rmse)
        fold_results['dense_model']['Correlation'].append(corr)
        
        rmse, corr = evaluate_fold(y_val, sparse_preds[val_index_coupled])
        fold_results['sparse_model']['RMSE'].append(rmse)
        fold_results['sparse_model']['Correlation'].append(corr)
        
        # Evaluate base models for this fold (averaged)
        dense_preds_avg = (dense_preds[val_index_coupled][0::2] + dense_preds[val_index_coupled][1::2]) / 2
        sparse_preds_avg = (sparse_preds[val_index_coupled][0::2] + sparse_preds[val_index_coupled][1::2]) / 2
        y_val_avg = (y_val[0::2] + y_val[1::2]) / 2
        
        rmse_avg, corr_avg = evaluate_fold(y_val_avg, dense_preds_avg)
        fold_results['dense_model']['RMSE_avg'].append(rmse_avg)
        fold_results['dense_model']['Correlation_avg'].append(corr_avg)
        
        rmse_avg, corr_avg = evaluate_fold(y_val_avg, sparse_preds_avg)
        fold_results['sparse_model']['RMSE_avg'].append(rmse_avg)
        fold_results['sparse_model']['Correlation_avg'].append(corr_avg)
        
        # Train and predict with meta models, prediction sample size is still 2x
        meta_features_train = np.column_stack((
            base_model_dense.predict(X_dense_train),
            base_model_sparse.predict(X_sparse_train)
        ))
        meta_features_val = np.column_stack((dense_preds[val_index_coupled], sparse_preds[val_index_coupled]))
        
        for name, meta_model in meta_models.items():
            meta_model.fit(meta_features_train, y_train)
            meta_preds[name][val_index_coupled] = meta_model.predict(meta_features_val)
            
            # Evaluate meta model for this fold (non-averaged)
            rmse, corr = evaluate_fold(y_val, meta_preds[name][val_index_coupled])
            fold_results[f'meta_model_{name}']['RMSE'].append(rmse)
            fold_results[f'meta_model_{name}']['Correlation'].append(corr)
            
            # Evaluate meta model for this fold (averaged)
            meta_preds_avg = (meta_preds[name][val_index_coupled][0::2] + meta_preds[name][val_index_coupled][1::2]) / 2
            rmse_avg, corr_avg = evaluate_fold(y_val_avg, meta_preds_avg)
            fold_results[f'meta_model_{name}']['RMSE_avg'].append(rmse_avg)
            fold_results[f'meta_model_{name}']['Correlation_avg'].append(corr_avg)
    
    # Calculate overall performance (non-averaged)
    overall_results = {
        'dense_model': {'RMSE': np.sqrt(mean_squared_error(y, dense_preds)),
                        'Correlation': pearsonr(y, dense_preds)[0]},
        'sparse_model': {'RMSE': np.sqrt(mean_squared_error(y, sparse_preds)),
                         'Correlation': pearsonr(y, sparse_preds)[0]}
    }
    
    for name in meta_models.keys():
        overall_results[f'meta_model_{name}'] = {
            'RMSE': np.sqrt(mean_squared_error(y, meta_preds[name])),
            'Correlation': pearsonr(y, meta_preds[name])[0]
        }
    
    # Calculate overall performance (averaged)
    y_avg = (y[0::2] + y[1::2]) / 2
    dense_preds_avg = (dense_preds[0::2] + dense_preds[1::2]) / 2
    sparse_preds_avg = (sparse_preds[0::2] + sparse_preds[1::2]) / 2
    
    overall_results['dense_model']['RMSE_avg'] = np.sqrt(mean_squared_error(y_avg, dense_preds_avg))
    overall_results['dense_model']['Correlation_avg'] = pearsonr(y_avg, dense_preds_avg)[0]
    overall_results['sparse_model']['RMSE_avg'] = np.sqrt(mean_squared_error(y_avg, sparse_preds_avg))
    overall_results['sparse_model']['Correlation_avg'] = pearsonr(y_avg, sparse_preds_avg)[0]
    
    for name in meta_models.keys():
        meta_preds_avg = (meta_preds[name][0::2] + meta_preds[name][1::2]) / 2
        overall_results[f'meta_model_{name}']['RMSE_avg'] = np.sqrt(mean_squared_error(y_avg, meta_preds_avg))
        overall_results[f'meta_model_{name}']['Correlation_avg'] = pearsonr(y_avg, meta_preds_avg)[0]
    
    return {'performance': overall_results, 'fold_results': fold_results}

def visualize_fold_results(fold_results):
    '''
    Visualize the performance of models by fold
    '''
    num_models = len(fold_results)
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    metrics = ['RMSE', 'Correlation', 'RMSE_avg', 'Correlation_avg']
    titles = ['RMSE (Non-averaged)', 'Correlation (Non-averaged)', 
              'RMSE (Averaged)', 'Correlation (Averaged)']
    
    for ax, metric, title in zip(axes.flatten(), metrics, titles):
        for model_name, results in fold_results.items():
            values = results[metric]
            ax.plot(range(1, len(values) + 1), values, marker='o', label=model_name)
        
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()


def stacking_ensemble_cv_averaged(X_dense, X_sparse, y, base_model_dense, base_model_sparse, meta_models, n_folds=10, seed=314159):
    '''
    Perform stacking the cross-validation results on the base and meta models, with averaging of predictions
    '''
    original_indices = np.arange(X_dense.shape[0] // 2)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    dense_preds_avg = np.zeros(len(y) // 2)
    sparse_preds_avg = np.zeros(len(y) // 2)
    meta_preds_avg = {name: np.zeros(len(y) // 2) for name in meta_models.keys()}
    
    fold_results = {
        'dense_model': {'RMSE': [], 'Correlation': []},
        'sparse_model': {'RMSE': [], 'Correlation': []}
    }
    for name in meta_models.keys():
        fold_results[f'meta_model_{name}'] = {'RMSE': [], 'Correlation': []}
    
    for train_index, val_index in kf.split(original_indices):
        train_index_coupled = np.sort(np.column_stack((2*train_index, 2*train_index+1)).flatten())
        val_index_coupled = np.sort(np.column_stack((2*val_index, 2*val_index+1)).flatten())
        
        X_dense_train, X_dense_val = X_dense[train_index_coupled], X_dense[val_index_coupled]
        X_sparse_train, X_sparse_val = X_sparse[train_index_coupled], X_sparse[val_index_coupled]
        y_train, y_val = y[train_index_coupled], y[val_index_coupled]
        
        # Train and predict with base models
        base_model_dense.fit(X_dense_train, y_train)
        base_model_sparse.fit(X_sparse_train, y_train)
        
        dense_preds = base_model_dense.predict(X_dense_val)
        sparse_preds = base_model_sparse.predict(X_sparse_val)
        
        # Average the predictions
        dense_preds_avg[val_index] = (dense_preds[0::2] + dense_preds[1::2]) / 2
        sparse_preds_avg[val_index] = (sparse_preds[0::2] + sparse_preds[1::2]) / 2
        y_val_avg = (y_val[0::2] + y_val[1::2]) / 2

        # Evaluate base models for this fold (averaged)
        rmse, corr = evaluate_fold(y_val_avg, dense_preds_avg[val_index])
        fold_results['dense_model']['RMSE'].append(rmse)
        fold_results['dense_model']['Correlation'].append(corr)
        
        rmse, corr = evaluate_fold(y_val_avg, sparse_preds_avg[val_index])
        fold_results['sparse_model']['RMSE'].append(rmse)
        fold_results['sparse_model']['Correlation'].append(corr)
        
        # Train and predict with meta models using averaged predictions, now trainig size is just original 
        meta_features_train = np.column_stack((
            (base_model_dense.predict(X_dense_train)[0::2] + base_model_dense.predict(X_dense_train)[1::2]) / 2,
            (base_model_sparse.predict(X_sparse_train)[0::2] + base_model_sparse.predict(X_sparse_train)[1::2]) / 2
        ))
        meta_features_val = np.column_stack((dense_preds_avg[val_index], sparse_preds_avg[val_index]))
        y_train_avg = (y_train[0::2] + y_train[1::2]) / 2
        
        for name, meta_model in meta_models.items():
            meta_model.fit(meta_features_train, y_train_avg)
            meta_preds_avg[name][val_index] = meta_model.predict(meta_features_val)
            
            # Evaluate meta model for this fold
            rmse, corr = evaluate_fold(y_val_avg, meta_preds_avg[name][val_index])
            fold_results[f'meta_model_{name}']['RMSE'].append(rmse)
            fold_results[f'meta_model_{name}']['Correlation'].append(corr)
    
    # Calculate overall performance
    y_avg = (y[0::2] + y[1::2]) / 2
    overall_results = {
        'dense_model': {'RMSE': np.sqrt(mean_squared_error(y_avg, dense_preds_avg)),
                        'Correlation': pearsonr(y_avg, dense_preds_avg)[0]},
        'sparse_model': {'RMSE': np.sqrt(mean_squared_error(y_avg, sparse_preds_avg)),
                         'Correlation': pearsonr(y_avg, sparse_preds_avg)[0]}
    }
    
    for name in meta_models.keys():
        overall_results[f'meta_model_{name}'] = {
            'RMSE': np.sqrt(mean_squared_error(y_avg, meta_preds_avg[name])),
            'Correlation': pearsonr(y_avg, meta_preds_avg[name])[0]
        }
    
    return {'performance': overall_results, 'fold_results': fold_results}

def visualize_fold_results_averaged(fold_results):
    '''
    Visualize the performance of models by fold, with averaging of predictions
    '''
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    metrics = ['RMSE', 'Correlation']
    titles = ['RMSE', 'Correlation']
    
    for ax, metric, title in zip(axes, metrics, titles):
        for model_name, results in fold_results.items():
            values = results[metric]
            ax.plot(range(1, len(values) + 1), values, marker='o', label=model_name)
        
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def print_results(results):
    '''
    Print the overall performance and fold-wise results
    '''
    overall_results = results['performance']
    fold_results = results['fold_results']

    print("Overall Performance:")
    print("=" * 50)
    for model, metrics in overall_results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    print("\n\nFold Results Summary:")
    print("=" * 50)
    for model, metrics in fold_results.items():
        print(f"\n{model}:")
        for metric, values in metrics.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            min_value = np.min(values)
            max_value = np.max(values)
            print(f"  {metric}:")
            print(f"    Mean ± Std: {mean_value:.4f} ± {std_value:.4f}")
            print(f"    Min: {min_value:.4f}")
            print(f"    Max: {max_value:.4f}")


def predict_base_models(X_dense_new, X_sparse_new, final_models):
    '''
    Predict with base models and average the predictions for each symmetric pair
    '''
    base_predictions = {'dense': [], 'sparse': []}
    
    for _, models in final_models.items():
        for dense_model, sparse_model, _ in models:
            dense_pred = dense_model.predict(X_dense_new)
            sparse_pred = sparse_model.predict(X_sparse_new)
            dense_pred = (dense_pred[0::2] + dense_pred[1::2])/2
            sparse_pred = (sparse_pred[0::2] + sparse_pred[1::2])/2          
            base_predictions['dense'].append(dense_pred)
            base_predictions['sparse'].append(sparse_pred)

    # Calculate mean predictions for base models
    base_predictions_mean = {
        'dense': np.mean(base_predictions['dense'], axis=0),
        'sparse': np.mean(base_predictions['sparse'], axis=0)
    }
    
    return base_predictions, base_predictions_mean

def evaluate_performance(y_true, y_pred):
    '''
    Evaluate the performance of the final model
    '''
    y_true = (y_true[0::2] + y_true[1::2])/2
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    corr, _ = pearsonr(y_true, y_pred)
    return rmse, corr

def train_final_models(X_dense, X_sparse, y, base_model_dense_class, base_model_sparse_class, meta_models, n_models=10, seed = 0):
    '''
    Train the final model (base and meta models); again prediciton will be averaged over symmetric pairs.
    '''
    final_models = {name: [] for name in meta_models.keys()}
    
    for seed in range(seed, n_models+ seed):
        base_model_dense = base_model_dense_class(**best_rf_dense, random_state=seed)
        base_model_sparse = base_model_sparse_class(**best_rf_sparse, random_state=seed)
        
        # Train base models
        final_base_model_dense = base_model_dense.fit(X_dense, y)
        final_base_model_sparse = base_model_sparse.fit(X_sparse, y)
        
        # Generate averaged predictions for meta models
        dense_preds = final_base_model_dense.predict(X_dense)
        sparse_preds = final_base_model_sparse.predict(X_sparse)
        
        dense_preds_avg = (dense_preds[0::2] + dense_preds[1::2]) / 2
        sparse_preds_avg = (sparse_preds[0::2] + sparse_preds[1::2]) / 2
        
        final_meta_features = np.column_stack((dense_preds_avg, sparse_preds_avg))
        y_avg = (y[0::2] + y[1::2]) / 2
        for name, meta_model_class in meta_models.items():
            if name == 'Poly_Ridge':
                meta_model = meta_model_class.fit(final_meta_features, y_avg)
            elif name == 'KNN':
                meta_model = meta_model_class().fit(final_meta_features, y_avg)
            else:
                meta_model = meta_model_class(random_state=seed).fit(final_meta_features, y_avg)
            final_models[name].append((final_base_model_dense, final_base_model_sparse, meta_model))

    return final_models


def predict_stacked_ensemble(X_dense_new, X_sparse_new, final_models):
    '''
    Predict with the final model (meta model)
    '''
    predictions = {name: [] for name in final_models.keys()}
    
    for name, models in final_models.items():
        for dense_model, sparse_model, meta_model in models:
            dense_pred = dense_model.predict(X_dense_new)
            sparse_pred = sparse_model.predict(X_sparse_new)
            
            dense_pred_avg = (dense_pred[0::2] + dense_pred[1::2]) / 2
            sparse_pred_avg = (sparse_pred[0::2] + sparse_pred[1::2]) / 2
            
            meta_features = np.column_stack((dense_pred_avg, sparse_pred_avg))
            meta_pred = meta_model.predict(meta_features)
            predictions[name].append(meta_pred)
    
    return {name: np.mean(preds, axis=0) for name, preds in predictions.items()}

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.base import BaseEstimator, MetaEstimatorMixin
import numpy as np
import xgboost as xgb
import pandas as pd
# from joblib import parallel_backend

import sys
import os

# Dynamically set python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the Python path
sys.path.append(parent_dir)

from src.utils import *
from src.utils import *


class MatchedKFold(BaseEstimator, MetaEstimatorMixin):
    '''class inplace of KFold for random search'''
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        # Couple 2i and 2i+1 th samples:
        n_samples = X.shape[0] // 2
        # Create split based on original number of samples:
        base_kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        for train_index, test_index in base_kf.split(np.arange(n_samples)):
            train_index_matched = np.concatenate([2*train_index, 2*train_index+1])
            test_index_matched = np.concatenate([2*test_index, 2*test_index+1])
            yield train_index_matched, test_index_matched

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

def para_search(seed, X, y_true):

    # Define the search space 
    rf_param_dist = {
        'n_estimators': [200, 300, 400, 500, 600, 700, 800],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True]
    }

    xgb_param_dist = {
        'n_estimators': [200, 300, 400, 500, 600, 700, 800],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.5, 0.7, 1.0],
        'colsample_bytree': [0.5, 0.7, 1.0]
    }


    rf = RandomForestRegressor(random_state=seed)
    xgb_model = xgb.XGBRegressor(random_state=seed)

    # Create the custom MatchedKFold splitter
    matched_cv = MatchedKFold(n_splits=10, shuffle=True, random_state=seed)

    # Perform Random Search with cross-validation for Random Forest
    # with parallel_backend('threading', n_jobs=8):

    #     rf_random = RandomizedSearchCV(estimator=rf, 
    #                                 param_distributions=rf_param_dist, 
    #                                 n_iter=50, 
    #                                 cv=matched_cv, 
    #                                 random_state=seed,) 
    #                                 # n_jobs=8)
    #     rf_random.fit(X, y_true)

    rf_random = RandomizedSearchCV(estimator=rf, 
                                param_distributions=rf_param_dist, 
                                n_iter=50, 
                                cv=matched_cv, 
                                random_state=seed, 
                                n_jobs=-1)
    rf_random.fit(X, y_true)

    best_rf = rf_random.best_estimator_

    # Perform Random Search with cross-validation for XGBoost
    # with parallel_backend('threading', n_jobs=8):

    #     xgb_random = RandomizedSearchCV(estimator=xgb_model, 
    #                                     param_distributions=xgb_param_dist, 
    #                                     n_iter=100, 
    #                                     cv=matched_cv, 
    #                                     random_state=seed,) 
    #                                     #n_jobs=8)
    #     xgb_random.fit(X, y_true)

    xgb_random = RandomizedSearchCV(estimator=xgb_model, 
                                    param_distributions=xgb_param_dist, 
                                    n_iter=100, 
                                    cv=matched_cv, 
                                    random_state=seed,
                                    n_jobs=-1)
    xgb_random.fit(X, y_true)

    best_xgb = xgb_random.best_estimator_

    # Evaluate 
    rf_pred = best_rf.predict(X)
    rf_corr = np.corrcoef(rf_pred, y_true)[0, 1]
    rf_rmse = np.sqrt(mean_squared_error(y_true, rf_pred))

    xgb_pred = best_xgb.predict(X)
    xgb_corr = np.corrcoef(xgb_pred, y_true)[0, 1]
    xgb_rmse = np.sqrt(mean_squared_error(y_true, xgb_pred))

    print("Best Random Forest model:")
    print("Hyperparameters:", rf_random.best_params_)
    print("Correlation:", rf_corr)
    print("RMSE:", rf_rmse)
    print()
    print("Best XGBoost model:")
    print("Hyperparameters:", xgb_random.best_params_)
    print("Correlation:", xgb_corr)
    print("RMSE:", xgb_rmse)

    return rf_random.best_params_, xgb_random.best_params_

def avg_rf_best(rf_best, X_features, y_true, random_seeds = None):

    if random_seeds == None:
        random_seeds = [42, 123, 456, 789, 1011]

    n_fold = 10

    seed_results = []

    for seed in random_seeds:
        np.random.seed(seed)
        rf_model = RandomForestRegressor(**rf_best, random_state=seed)
        original_indices = np.arange(X_features.shape[0] // 2)
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
        
        rf_corr_fold, rf_rmse_fold = [], []
        rf_corr_avg_fold, rf_rmse_avg_fold = [], []
        all_rf_pred, all_y_true = [], []
        
        for train_index, test_index in kf.split(original_indices):
            train_index_coupled = np.sort(np.column_stack((2*train_index, 2*train_index+1)).flatten())
            test_index_coupled = np.sort(np.column_stack((2*test_index, 2*test_index+1)).flatten())
            
            X_train, X_test = X_features[train_index_coupled], X_features[test_index_coupled]
            y_train, y_test = y_true[train_index_coupled], y_true[test_index_coupled]
            
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            
            all_rf_pred.extend(rf_pred)
            all_y_true.extend(y_test)
            
            # Non-averaged metrics, per fold
            rf_corr = np.corrcoef(rf_pred, y_test)[0, 1]
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
            rf_corr_fold.append(rf_corr)
            rf_rmse_fold.append(rf_rmse)
            
            # Averaged metrics, per fold
            rf_pred_avg = (rf_pred[0::2] + rf_pred[1::2]) / 2
            y_test_avg = (y_test[0::2] + y_test[1::2]) / 2
            rf_corr_avg = np.corrcoef(rf_pred_avg, y_test_avg)[0, 1]
            rf_rmse_avg = np.sqrt(mean_squared_error(y_test_avg, rf_pred_avg))
            rf_corr_avg_fold.append(rf_corr_avg)
            rf_rmse_avg_fold.append(rf_rmse_avg)
        
        # Calculate overall metrics for this seed
        all_rf_pred, all_y_true = np.array(all_rf_pred), np.array(all_y_true)
        overall_corr = np.corrcoef(all_rf_pred, all_y_true)[0, 1]
        overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_rf_pred))
        
        all_rf_pred_avg = (all_rf_pred[0::2] + all_rf_pred[1::2]) / 2
        all_y_true_avg = (all_y_true[0::2] + all_y_true[1::2]) / 2
        overall_corr_avg = np.corrcoef(all_rf_pred_avg, all_y_true_avg)[0, 1]
        overall_rmse_avg = np.sqrt(mean_squared_error(all_y_true_avg, all_rf_pred_avg))
        
        # Store results for this seed
        seed_results.append({
            'per_fold_r': np.mean(rf_corr_fold),
            'per_fold_rmse': np.mean(rf_rmse_fold),
            'per_fold_r_avg': np.mean(rf_corr_avg_fold),
            'per_fold_rmse_avg': np.mean(rf_rmse_avg_fold),
            'overall_r': overall_corr,
            'overall_rmse': overall_rmse,
            'overall_r_avg': overall_corr_avg,
            'overall_rmse_avg': overall_rmse_avg
        })

    # Calculate mean and std across seeds
    results = pd.DataFrame(seed_results)
    mean_results = results.mean()
    std_results = results.std()

    print("RandomForest Performance (averaged over seeds):")
    print("\nPer-fold Performance (Non-averaged):")
    print(f"R mean: {mean_results['per_fold_r']:.4f} ± {std_results['per_fold_r']:.4f}")
    print(f"RMSE mean: {mean_results['per_fold_rmse']:.4f} ± {std_results['per_fold_rmse']:.4f}")
    print("\nPer-fold Performance (Averaged):")
    print(f"R mean: {mean_results['per_fold_r_avg']:.4f} ± {std_results['per_fold_r_avg']:.4f}")
    print(f"RMSE mean: {mean_results['per_fold_rmse_avg']:.4f} ± {std_results['per_fold_rmse_avg']:.4f}")
    print("\nOverall Performance (Non-averaged):")
    print(f"R: {mean_results['overall_r']:.4f} ± {std_results['overall_r']:.4f}")
    print(f"RMSE: {mean_results['overall_rmse']:.4f} ± {std_results['overall_rmse']:.4f}")
    print("\nOverall Performance (Averaged):")
    print(f"R: {mean_results['overall_r_avg']:.4f} ± {std_results['overall_r_avg']:.4f}")
    print(f"RMSE: {mean_results['overall_rmse_avg']:.4f} ± {std_results['overall_rmse_avg']:.4f}")

    return mean_results.to_dict(), std_results.to_dict()




def avg_xgb_best(xgb_best, X_features, y_true, random_seeds = None):

    if random_seeds == None:
        random_seeds = [42, 123, 456, 789, 1011]

    n_fold = 10

    seed_results = []

    for seed in random_seeds:
        np.random.seed(seed)
        xgb_model = xgb.XGBRegressor(**xgb_best, random_state=seed)
        original_indices = np.arange(X_features.shape[0] // 2)
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
        
        xgb_corr_fold, xgb_rmse_fold = [], []
        xgb_corr_avg_fold, xgb_rmse_avg_fold = [], []
        all_xgb_pred, all_y_true = [], []
        
        for train_index, test_index in kf.split(original_indices):
            train_index_coupled = np.sort(np.column_stack((2*train_index, 2*train_index+1)).flatten())
            test_index_coupled = np.sort(np.column_stack((2*test_index, 2*test_index+1)).flatten())
            
            X_train, X_test = X_features[train_index_coupled], X_features[test_index_coupled]
            y_train, y_test = y_true[train_index_coupled], y_true[test_index_coupled]
            
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
            
            all_xgb_pred.extend(xgb_pred)
            all_y_true.extend(y_test)
            
            # Non-averaged metrics, per fold
            xgb_corr = np.corrcoef(xgb_pred, y_test)[0, 1]
            xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
            xgb_corr_fold.append(xgb_corr)
            xgb_rmse_fold.append(xgb_rmse)
            
            # Averaged metrics, per fold
            xgb_pred_avg = (xgb_pred[0::2] + xgb_pred[1::2]) / 2
            y_test_avg = (y_test[0::2] + y_test[1::2]) / 2
            xgb_corr_avg = np.corrcoef(xgb_pred_avg, y_test_avg)[0, 1]
            xgb_rmse_avg = np.sqrt(mean_squared_error(y_test_avg, xgb_pred_avg))
            xgb_corr_avg_fold.append(xgb_corr_avg)
            xgb_rmse_avg_fold.append(xgb_rmse_avg)
        
        # Calculate overall metrics for this seed
        all_xgb_pred, all_y_true = np.array(all_xgb_pred), np.array(all_y_true)
        overall_corr = np.corrcoef(all_xgb_pred, all_y_true)[0, 1]
        overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_xgb_pred))
        
        all_xgb_pred_avg = (all_xgb_pred[0::2] + all_xgb_pred[1::2]) / 2
        all_y_true_avg = (all_y_true[0::2] + all_y_true[1::2]) / 2
        overall_corr_avg = np.corrcoef(all_xgb_pred_avg, all_y_true_avg)[0, 1]
        overall_rmse_avg = np.sqrt(mean_squared_error(all_y_true_avg, all_xgb_pred_avg))
        
        # Store results for this seed
        seed_results.append({
            'per_fold_r': np.mean(xgb_corr_fold),
            'per_fold_rmse': np.mean(xgb_rmse_fold),
            'per_fold_r_avg': np.mean(xgb_corr_avg_fold),
            'per_fold_rmse_avg': np.mean(xgb_rmse_avg_fold),
            'overall_r': overall_corr,
            'overall_rmse': overall_rmse,
            'overall_r_avg': overall_corr_avg,
            'overall_rmse_avg': overall_rmse_avg
        })

    # Calculate mean and std across seeds
    results = pd.DataFrame(seed_results)
    mean_results = results.mean()
    std_results = results.std()

    print("XGBoost Performance (averaged over seeds):")
    print("\nPer-fold Performance (Non-averaged):")
    print(f"R mean: {mean_results['per_fold_r']:.4f} ± {std_results['per_fold_r']:.4f}")
    print(f"RMSE mean: {mean_results['per_fold_rmse']:.4f} ± {std_results['per_fold_rmse']:.4f}")
    print("\nPer-fold Performance (Averaged):")
    print(f"R mean: {mean_results['per_fold_r_avg']:.4f} ± {std_results['per_fold_r_avg']:.4f}")
    print(f"RMSE mean: {mean_results['per_fold_rmse_avg']:.4f} ± {std_results['per_fold_rmse_avg']:.4f}")
    print("\nOverall Performance (Non-averaged):")
    print(f"R: {mean_results['overall_r']:.4f} ± {std_results['overall_r']:.4f}")
    print(f"RMSE: {mean_results['overall_rmse']:.4f} ± {std_results['overall_rmse']:.4f}")
    print("\nOverall Performance (Averaged):")
    print(f"R: {mean_results['overall_r_avg']:.4f} ± {std_results['overall_r_avg']:.4f}")
    print(f"RMSE: {mean_results['overall_rmse_avg']:.4f} ± {std_results['overall_rmse_avg']:.4f}")

    return mean_results.to_dict(), std_results.to_dict()
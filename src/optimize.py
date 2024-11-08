
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
import xgboost as xgb

import sys
import os

# sys.path.append("/Users/xinzheng/Desktop/Desktop/DreamRF")

# Dynamically set python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the Python path
sys.path.append(parent_dir)

from src.utils import *



# n_iter = 100
# cv = 10

def para_search(seed, X, y_true):
    # Define the search space 
    rf_param_dist = {
        # 'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500],
        'n_estimators': [150, 200, 250, 300, 400, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True]
    }

    xgb_param_dist = {
        # 'n_estimators': [50, 100, 200, 250, 300, 400, 500],
        'n_estimators': [150, 200, 250, 300, 400, 500],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.5, 0.7, 1.0],
        'colsample_bytree': [0.5, 0.7, 1.0]
    }

    # Specifically for sparse input:
    # rf_param_dist = {
    # 'n_estimators': [100, 200, 300, 400, 500],
    # 'max_depth': [None, 10, 20, 30],
    # 'min_samples_split': [2, 5, 10],
    # 'min_samples_leaf': [1, 2, 4],
    # 'max_features': ['sqrt', 'log2', 0.5],
    # 'bootstrap': [True, False]
    # }

    # xgb_param_dist = {
    #     'n_estimators': [100, 200, 300, 400, 500],
    #     'max_depth': [3, 6, 9],
    #     'learning_rate': [0.01, 0.1, 0.3],
    #     'subsample': [0.5, 0.7, 1.0],
    #     'colsample_bytree': [0.5, 0.7, 1.0],
    #     'min_child_weight': [1, 3, 5],
    #     'reg_alpha': [0, 0.1, 1],
    #     'reg_lambda': [0, 0.1, 1]
    # }
    
    # Create models
    rf = RandomForestRegressor(random_state=seed)
    xgb_model = xgb.XGBRegressor(random_state=seed)

    # Perform Random Search with cross-validation for Random Forest
    rf_random = RandomizedSearchCV(estimator=rf, 
                                param_distributions=rf_param_dist, 
                                n_iter=50, 
                                cv=10, 
                                random_state=seed, 
                                n_jobs=-1)
    rf_random.fit(X, y_true)

    best_rf = rf_random.best_estimator_

    # Perform Random Search with cross-validation for XGBoost
    xgb_random = RandomizedSearchCV(estimator=xgb_model, 
                                    param_distributions=xgb_param_dist, 
                                    n_iter=100, 
                                    cv=10, 
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

    return  rf_random.best_params_, xgb_random.best_params_

def avg_rf_best(rf_best, X_features, y_true):
    # Random seeds to get average performance
    random_seeds = [42, 123, 456, 789, 1011]
    n_fold = 10  # Number of folds for cross-validation

    rf_corr_list = []
    rf_rmse_list = []

    # Evaluate the models with different random seeds
    for seed in random_seeds:
        np.random.seed(seed)
        
        # Create the XGBoost model with the best hyperparameters
        rf_model =  RandomForestRegressor(**rf_best, random_state=seed)
        
        # Create the KFold object for cross-validation
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
        
        rf_corr_fold = []
        rf_rmse_fold = []
        
        # Perform cross-validation
        for train_index, test_index in kf.split(X_features):
            X_train, X_test = X_features[train_index], X_features[test_index]
            y_train, y_test = y_true[train_index], y_true[test_index]
            
            # Train the model
            rf_model.fit(X_train, y_train)
            
            # Evaluate the model on the testing fold
            rf_pred = rf_model.predict(X_test)
            rf_corr = np.corrcoef(rf_pred, y_test)[0, 1]
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
            
            rf_corr_fold.append(rf_corr)
            rf_rmse_fold.append(rf_rmse)
        
        # Calculate the average performance across all folds
        rf_corr_avg = np.mean(rf_corr_fold)
        rf_rmse_avg = np.mean(rf_rmse_fold)
        
        rf_corr_list.append(rf_corr_avg)
        rf_rmse_list.append(rf_rmse_avg)

    print("RandomForest Average Performance:")
    print("R mean:", np.mean(rf_corr_list))
    print("R std:", np.std(rf_corr_list))
    print("RMSE mean:", np.mean(rf_rmse_list))
    print("RMSE std:", np.std(rf_rmse_list))

    return np.mean(rf_corr_list), np.mean(rf_rmse_list)
    

def avg_rgb_best(rbg_best, X_features, y_true):
    # Random seeds to get average performance
    random_seeds = [42, 123, 456, 789, 1011]
    n_fold = 10  # Number of folds for cross-validation

    xgb_corr_list = []
    xgb_rmse_list = []

    # Evaluate the models with different random seeds
    for seed in random_seeds:
        np.random.seed(seed)
        
        # Create the XGBoost model with the best hyperparameters
        xgb_model = xgb.XGBRegressor(**rbg_best, random_state=seed)
        
        # Create the KFold object for cross-validation
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
        
        xgb_corr_fold = []
        xgb_rmse_fold = []
        
        # Perform cross-validation
        for train_index, test_index in kf.split(X_features):
            X_train, X_test = X_features[train_index], X_features[test_index]
            y_train, y_test = y_true[train_index], y_true[test_index]
            
            # Train the model
            xgb_model.fit(X_train, y_train)
            
            # Evaluate the model on the testing fold
            xgb_pred = xgb_model.predict(X_test)
            xgb_corr = np.corrcoef(xgb_pred, y_test)[0, 1]
            xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
            
            xgb_corr_fold.append(xgb_corr)
            xgb_rmse_fold.append(xgb_rmse)
        
        # Calculate the average performance across all folds
        xgb_corr_avg = np.mean(xgb_corr_fold)
        xgb_rmse_avg = np.mean(xgb_rmse_fold)
        
        xgb_corr_list.append(xgb_corr_avg)
        xgb_rmse_list.append(xgb_rmse_avg)

    print("XGBoost Average Performance:")
    print("R mean:", np.mean(xgb_corr_list))
    print("R std:", np.std(xgb_corr_list))
    print("RMSE mean:", np.mean(xgb_rmse_list))
    print("RMSE std:", np.std(xgb_rmse_list))

    return np.mean(xgb_corr_list), np.mean(xgb_rmse_list)

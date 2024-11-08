from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.utils import *

input_path = os.path.join(parent_dir, 'Data')

CID_file = 'molecules_train_cid.npy'
mixture_file = 'Mixure_Definitions_Training_set.csv'

training_task_file = 'TrainingData_mixturedist.csv'
all_task_file = os.path.join(parent_dir, 'Test', 'Data', 'AllData_mixturedist.csv')

leaderboard_task_file = os.path.join(parent_dir, 'Test', 'Data', 'LeaderboardData_mixturedist.csv')
test_task_file = os.path.join(parent_dir, 'Test', 'Data', 'TestData_mixsturedist.csv')

features_CIDs = np.load(os.path.join(input_path, CID_file))
# Mapping helper files
mixtures_IDs = pd.read_csv(os.path.join(input_path, mixture_file))

# Training dataframe
training_set = pd.read_csv(os.path.join(input_path, training_task_file))
all_training_set = pd.read_csv(all_task_file)
# Test dataframe
leaderboard_set = pd.read_csv(leaderboard_task_file)
test_set = pd.read_csv(test_task_file)



def stacking_symmetric_X_features(CID2features_list, method, data = 'training'):

    # added for testing in different scenarios:
    if data == 'training':
        training_dataset = training_set
    elif data == 'all':
        training_dataset = all_training_set
    
    def create_copy(order):
        'create symmetric copy of samples'
        training_size = len(training_dataset)
        stacks = []
        
        for CID2features in CID2features_list:
            
            X, y, num_mixtures, all_pairs_CIDs = format_Xy(training_dataset,  mixtures_IDs, CID2features, method = method)

            # Added to swap:
            num_mols = np.array(num_mixtures).reshape(training_size,2)

            if order == 0:
                X_pairs = np.array([(np.concatenate((x1, x2))) for x1, x2 in X])
            elif order == 1:
                X_pairs = np.array([(np.concatenate((x2, x1))) for x1, x2 in X])
                # Added to swap:
                num_mols[:, [0, 1]] = num_mols[:, [1, 0]]

            distances= [get_euclidean_distance(m[0], m[1]) for m in X]
            similarities = [get_cosine_similarity(m[0], m[1]) for m in X]
            angles = [get_cosine_angle(m[0], m[1]) for m in X] 
            
            stack = np.hstack( (X_pairs,
                            np.array(distances).reshape(training_size, 1), 
                            np.array(similarities).reshape(training_size, 1), 
                            np.array(angles).reshape(training_size, 1)))
            stacks.append(stack)
        

        shared_monos = [ len( set(pair[0]).intersection(set(pair[1]))) for pair in all_pairs_CIDs]
        diff_monos = [ len( set(pair[0]).difference(set(pair[1]))) for pair in all_pairs_CIDs]
        
        datasets = training_dataset['Dataset'].to_numpy()
        # Returns the uniques in the order of appearance
        desired_order = training_dataset['Dataset'].unique().tolist() 
        encoder = OneHotEncoder(categories=[desired_order])
        data_arr = encoder.fit_transform(datasets.reshape(-1, 1))
        data_arr = data_arr.toarray()

        engineered_stack = np.hstack(
                            (np.array(shared_monos).reshape(training_size, 1), 
                            np.array(diff_monos).reshape(training_size, 1), 
                            num_mols,
                            # np.array(num_mixtures).reshape(training_size,2), 
                            data_arr))
        
        stacks.append(engineered_stack)

        concatenated_array = np.hstack(stacks)
        X_features_copy = np.stack(concatenated_array)
    
        return X_features_copy, y
    
    X_features_1, y = create_copy(0)
    X_features_2, _ = create_copy(1)

    X_features = np.empty((X_features_1.shape[0]*2, X_features_1.shape[1]), dtype=X_features_1.dtype)
    X_features[0::2] = X_features_1
    X_features[1::2] = X_features_2

    y_true= np.repeat(y, 2)

    return (X_features_1, X_features_2), X_features, y_true

# This is the same
def ensemble_models(X_features, y_true, param_best, type = 'rf', num_models = 10):
    models = []
    for i in range(num_models):
        if type == 'rf': 
            model = RandomForestRegressor(**param_best, random_state=i)
            model.fit(X_features, y_true)
        elif type == 'xgb':
            model = xgb.XGBRegressor(**param_best, random_state=i)
            model.fit(X_features, y_true)
        models.append(model)
    return models

def stacking_X_test_features(CID2features_list, X_train_1, X_train_2, method, data = 'leaderboard'):

    # Added for testing in different scenarios:
    if data == 'leaderboard':
        test_dataset = leaderboard_set
        
    elif data == 'test':
        test_dataset = test_set

    def create_test_copy(order):

        test_size = len(test_dataset)
        stacks = []
        
        for CID2features in CID2features_list:

            X, y, num_mixtures, all_pairs_CIDs = format_Xy(test_dataset,  mixtures_IDs, CID2features, method = method)

            # Added to swap:
            num_mols = np.array(num_mixtures).reshape(test_size,2)

            if order == 0:
                X_pairs = np.array([(np.concatenate((x1, x2))) for x1, x2 in X])
            elif order == 1:
                X_pairs = np.array([(np.concatenate((x2, x1))) for x1, x2 in X])
                # Added to swap:
                num_mols[:, [0, 1]] = num_mols[:, [1, 0]]

            distances= [get_euclidean_distance(m[0], m[1]) for m in X]
            similarities = [get_cosine_similarity(m[0], m[1]) for m in X]
            angles = [get_cosine_angle(m[0], m[1]) for m in X] 
            
            stack = np.hstack( (X_pairs,
                            np.array(distances).reshape(test_size, 1), 
                            np.array(similarities).reshape(test_size, 1), 
                            np.array(angles).reshape(test_size, 1)))
            stacks.append(stack)

        shared_monos = [ len( set(pair[0]).intersection(set(pair[1]))) for pair in all_pairs_CIDs]
        diff_monos = [ len( set(pair[0]).difference(set(pair[1]))) for pair in all_pairs_CIDs]
        
        data_arr = np.full((len(test_dataset), 4), np.nan) 

        engineered_stack = np.hstack(
                            (np.array(shared_monos).reshape(test_size, 1), 
                            np.array(diff_monos).reshape(test_size, 1), 
                            num_mols,
                            # np.array(num_mixtures).reshape(test_size,2), 
                            data_arr))
        
        stacks.append(engineered_stack)

        concatenated_array = np.hstack(stacks)
        X_features = np.stack(concatenated_array)
        
        # Create a KNNImputer object
        imputer = KNNImputer(n_neighbors=5)

        # Fit the imputer on the training data
        if order == 0:
            imputer.fit(X_train_1)
        elif order == 1:
            imputer.fit(X_train_2)

        # Transform the test data
        X_test_copy = imputer.transform(X_features)

        return X_test_copy, y

    X_test_1, y = create_test_copy(0)
    X_test_2, _ = create_test_copy(1)

    X_test = np.empty((X_test_1.shape[0]*2, X_test_1.shape[1]), dtype=X_test_1.dtype)
    X_test[0::2] = X_test_1
    X_test[1::2] = X_test_2

    y_true= np.repeat(y, 2)

    return X_test, np.array(y_true)



def pred_mean(models, X_test, return_original=False):
    y_pred_list = []
    y_pred_avg_list = []
    for model in models: 
        y_pred = model.predict(X_test)
        y_pred_list.append(y_pred)
        # Average the predictions for coupled samples
        y_pred_avg = (y_pred[0::2] + y_pred[1::2]) / 2
        y_pred_avg_list.append(y_pred_avg)

    # Average predictions across all models
    y_pred_final = np.mean(y_pred_list, axis=0)
    y_pred_avg_final = np.mean(y_pred_avg_list, axis=0)

    if return_original:
        return y_pred_final, y_pred_avg_final
    else:
        return y_pred_avg_final


def bootstrap_metrics_small_sample(y_true, y_pred, n_iterations=1000):
    n_samples = len(y_true)
    results = {'corr': [], 'rmse': []}
    
    for _ in range(n_iterations):
        # Resample with replacement
        indices = resample(range(n_samples), n_samples=n_samples)
        
        # Find samples that were not selected (out-of-bag samples)
        oob_indices = list(set(range(n_samples)) - set(indices))
        
        # If we have out-of-bag samples, use them for evaluation
        if oob_indices:
            y_true_oob = y_true[oob_indices]
            y_pred_oob = y_pred[oob_indices]
            
            # Calculate metrics
            corr, _ = pearsonr(y_true_oob, y_pred_oob)
            rmse = np.sqrt(mean_squared_error(y_true_oob, y_pred_oob))
            
            results['corr'].append(corr)
            results['rmse'].append(rmse)
    
    # Calculate confidence intervals
    ci_lower, ci_upper = 2.5, 97.5  # For 95% CI
    
    corr_ci = np.percentile(results['corr'], [ci_lower, ci_upper])
    rmse_ci = np.percentile(results['rmse'], [ci_lower, ci_upper])
    
    return {
        'corr': {
            'mean': np.mean(results['corr']),
            'ci': corr_ci
        },
        'rmse': {
            'mean': np.mean(results['rmse']),
            'ci': rmse_ci
        }
    }
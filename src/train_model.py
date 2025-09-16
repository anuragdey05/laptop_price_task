"""
Laptop Price Prediction - Model Training
Clean, concise implementation of multiple regression models
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
from itertools import combinations


class BaseRegression:
    """Base class for all regression models"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, verbose=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def _compute_gradient(self, X, y):
        raise NotImplementedError

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0

        for i in range(self.n_iterations):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Compute gradients and update parameters
            dw, db = self._compute_gradient(X_shuffled, y_shuffled)
            self.weights -= self.learning_rate * dw.astype(np.float64)
            self.bias -= self.learning_rate * float(db)
            
            # Print progress
            if self.verbose and i % 200 == 0:
                pred = self.predict(X)
                cost = np.mean((pred - y)**2) / 2
                print(f"Iteration {i}: Cost = {cost:.4f}")
        
        return self


class LinearRegression(BaseRegression):
    def _compute_gradient(self, X, y):
        n_samples = len(y)
        predictions = self.predict(X)
        error = predictions - y
        
        dw = (1/n_samples) * np.dot(X.T, error)
        db = (1/n_samples) * np.sum(error)
        return dw, db


class LassoRegression(BaseRegression):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def _compute_gradient(self, X, y):
        n_samples = len(y)
        predictions = self.predict(X)
        error = predictions - y
        
        dw = (1/n_samples) * np.dot(X.T, error) + (self.alpha/n_samples) * np.sign(self.weights)
        db = (1/n_samples) * np.sum(error)
        return dw, db


class PolynomialFeatures:
    def __init__(self, degree=2):
        self.degree = degree

    def fit_transform(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Start with linear terms
        features = [X[:, i:i+1] for i in range(n_features)]
        
        # Add polynomial terms
        if self.degree > 1:
            for d in range(2, self.degree + 1):
                # Single feature powers
                for i in range(n_features):
                    features.append(X[:, i:i+1]**d)
                # Interaction terms (only for degree 2)
                if d == 2:
                    for i, j in combinations(range(n_features), 2):
                        features.append((X[:, i] * X[:, j]).reshape(-1, 1))
        
        return np.hstack(features)


def load_data():
    """Load and prepare training data"""
    print("Loading preprocessed data...")
    X_train = pd.read_csv('data/X_train.csv').to_numpy().astype(np.float64)
    X_test = pd.read_csv('data/X_test.csv').to_numpy().astype(np.float64)
    y_train = pd.read_csv('data/y_train.csv').values.ravel().astype(np.float64)
    y_test = pd.read_csv('data/y_test.csv').values.ravel().astype(np.float64)
    
    print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}


def inverse_log_transform(y_log):
    """Convert log predictions back to original scale"""
    return np.expm1(y_log)


def cross_validate_lasso(X, y):
    """Find best alpha for Lasso regression"""
    print("Running cross-validation for Lasso...")
    
    alphas = np.logspace(-4, 2, 8)
    best_alpha = alphas[0]
    best_rmse = float('inf')
    
    # Simple 5-fold cross-validation
    n_samples = len(X)
    fold_size = n_samples // 5
    
    for alpha in alphas:
        fold_rmses = []
        
        for fold in range(5):
            # Split data
            val_start = fold * fold_size
            val_end = val_start + fold_size
            val_indices = slice(val_start, val_end)
            train_indices = np.concatenate([np.arange(0, val_start), np.arange(val_end, n_samples)])
            
            X_fold_train, y_fold_train = X[train_indices], y[train_indices]
            X_fold_val, y_fold_val = X[val_indices], y[val_indices]
            
            # Train and evaluate
            model = LassoRegression(alpha=alpha, n_iterations=500, verbose=False)
            model.fit(X_fold_train, y_fold_train)
            
            pred = model.predict(X_fold_val)
            rmse = np.sqrt(np.mean((pred - y_fold_val)**2))
            fold_rmses.append(rmse)
        
        avg_rmse = np.mean(fold_rmses)
        print(f"  Alpha {alpha:.4f}: RMSE = {avg_rmse:.4f}")
        
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_alpha = alpha
    
    print(f"Best alpha: {best_alpha:.4f} with RMSE: {best_rmse:.4f}")
    return best_alpha


def train_single_model(model_class, model_name, X_train, X_test, y_train, y_test, **model_kwargs):
    """Train a single model and return results"""
    print(f"\nTraining {model_name}...")
    
    # Train model
    model = model_class(**model_kwargs)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Convert to original scale
    y_train_orig = inverse_log_transform(y_train)
    y_test_orig = inverse_log_transform(y_test)
    y_train_pred_orig = inverse_log_transform(y_train_pred)
    y_test_pred_orig = inverse_log_transform(y_test_pred)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train_orig, y_train_pred_orig)
    test_metrics = calculate_metrics(y_test_orig, y_test_pred_orig)
    
    print(f"{model_name} - Test RMSE: {test_metrics['RMSE']:.2f}, R²: {test_metrics['R2']:.4f}")
    
    return {
        'model': model,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'predictions': {
            'y_test_true': y_test_orig,
            'y_test_pred': y_test_pred_orig
        }
    }


def train_polynomial_models(X_train, X_test, y_train, y_test):
    """Train polynomial regression models with different degrees"""
    print("\nTraining Polynomial Regression...")
    
    degrees = [2, 3, 4]
    best_result = None
    best_rmse = float('inf')
    
    for degree in degrees:
        print(f"\n  Testing degree {degree}...")
        
        # Create polynomial features
        poly_transformer = PolynomialFeatures(degree=degree)
        X_train_poly = poly_transformer.fit_transform(X_train)
        X_test_poly = poly_transformer.fit_transform(X_test)
        
        print(f"    Features: {X_train.shape[1]} → {X_train_poly.shape[1]}")
        
        # Skip if too many features
        if X_train_poly.shape[1] > X_train.shape[0]:
            print(f"    Skipped (too many features)")
            continue
        
        # Train model with adaptive learning rate
        lr = 0.001 / np.sqrt(degree)
        result = train_single_model(
            LinearRegression, f"Polynomial (degree {degree})",
            X_train_poly, X_test_poly, y_train, y_test,
            learning_rate=lr, verbose=False
        )
        
        # Store transformer with result
        result['transformer'] = poly_transformer
        result['degree'] = degree
        
        # Track best model
        if result['test_metrics']['RMSE'] < best_rmse:
            best_rmse = result['test_metrics']['RMSE']
            best_result = result
    
    if best_result:
        print(f"\nBest polynomial: degree {best_result['degree']} (RMSE: {best_rmse:.2f})")
    
    return best_result


def save_results(results, best_model_name):
    """Save training results to files"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Save individual models
    model_files = {
        'Linear Regression': 'models/regression_model1.pkl',
        'Polynomial Regression': 'models/regression_model2.pkl', 
        'Lasso Regression': 'models/regression_model3.pkl'
    }
    
    for model_name, result in results.items():
        if model_name in model_files:
            with open(model_files[model_name], 'wb') as f:
                if 'transformer' in result:
                    # Save polynomial model with transformer
                    pickle.dump({
                        'model': result['model'],
                        'transformer': result['transformer'],
                        'degree': result['degree']
                    }, f)
                else:
                    pickle.dump(result['model'], f)
    
    # Save best model
    with open('models/regression_model_final.pkl', 'wb') as f:
        pickle.dump(results[best_model_name]['model'], f)
    
    # Save metrics
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('results/train_metrics.txt', 'w') as f:
        f.write(f"LAPTOP PRICE PREDICTION - TRAINING RESULTS\n")
        f.write(f"Training Date: {timestamp}\n\n")
        
        for model_name, result in results.items():
            f.write(f"{model_name.upper()}\n")
            f.write("-" * 30 + "\n")
            
            test_metrics = result['test_metrics']
            f.write(f"Test RMSE: {test_metrics['RMSE']:.2f}\n")
            f.write(f"Test MAE:  {test_metrics['MAE']:.2f}\n")
            f.write(f"Test R²:   {test_metrics['R2']:.4f}\n\n")
        
        f.write(f"BEST MODEL: {best_model_name}\n")
        f.write(f"Best RMSE: {results[best_model_name]['test_metrics']['RMSE']:.2f}\n")
    
    print(f"\nResults saved to 'models/' and 'results/' directories")


def main():
    """Main training pipeline"""
    print("="*60)
    print("LAPTOP PRICE PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Set random seed
    np.random.seed(42)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train models
    results = {}
    
    # 1. Linear Regression
    results['Linear Regression'] = train_single_model(
        LinearRegression, 'Linear Regression',
        X_train, X_test, y_train, y_test
    )
    
    # 2. Polynomial Regression
    poly_result = train_polynomial_models(X_train, X_test, y_train, y_test)
    if poly_result:
        results['Polynomial Regression'] = poly_result
    
    # 3. Lasso Regression
    best_alpha = cross_validate_lasso(X_train, y_train)
    results['Lasso Regression'] = train_single_model(
        LassoRegression, 'Lasso Regression',
        X_train, X_test, y_train, y_test,
        alpha=best_alpha
    )
    
    # Find best model
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    best_model_name = None
    best_rmse = float('inf')
    
    for model_name, result in results.items():
        rmse = result['test_metrics']['RMSE']
        r2 = result['test_metrics']['R2']
        print(f"{model_name}: RMSE = {rmse:.2f}, R² = {r2:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = model_name
    
    print(f"\nBest Model: {best_model_name} (RMSE: {best_rmse:.2f})")
    
    # Save results
    save_results(results, best_model_name)
    print("Training completed successfully!")


if __name__ == '__main__':
    main()
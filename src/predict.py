import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_model import LinearRegression, LassoRegression, PolynomialFeatures


def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def inverse_log_transform(y_log):
    return np.expm1(np.array(y_log))


def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}


def make_predictions(model_path, data_path, output_path):
    print(f"Loading model from {model_path}...")
    
    model_data = load_model(model_path)
    X = pd.read_csv(data_path).to_numpy().astype(np.float64)
    
    if isinstance(model_data, dict) and 'transformer' in model_data:
        transformer = model_data['transformer']
        model = model_data['model']
        degree = model_data.get('degree', 2)
        X_transformed = transformer.fit_transform(X)
        y_pred_log = model.predict(X_transformed)
        print(f"Using polynomial regression (degree {degree})")
        print(f"Features: {X.shape[1]} → {X_transformed.shape[1]}")
    else:
        model = model_data
        y_pred_log = model.predict(X)
    
    y_pred_original = inverse_log_transform(y_pred_log)
    
    results = pd.DataFrame({
        'Predicted_Price_Log': y_pred_log,
        'Predicted_Price': y_pred_original
    })
    
    if 'test' in data_path.lower():
        try:
            y_true = pd.read_csv('data/y_test.csv').values.ravel()
            y_true_original = inverse_log_transform(y_true)
            
            results['True_Price_Log'] = y_true
            results['True_Price'] = y_true_original
            results['Error'] = abs(y_true_original - y_pred_original)
            results['Error_Percentage'] = (results['Error'] / y_true_original) * 100
            
            metrics = calculate_metrics(y_true_original, y_pred_original)
            print(f"\nPrediction Metrics:")
            print(f"RMSE: ₹{metrics['RMSE']:,.0f}")
            print(f"MAE:  ₹{metrics['MAE']:,.0f}")
            print(f"R²:   {metrics['R2']:.4f}")
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('results/prediction_metrics.txt', 'w') as f:
                f.write(f"PREDICTION METRICS\n")
                f.write(f"Date: {timestamp}\n")
                f.write(f"Model: {model_path}\n")
                f.write(f"Data: {data_path}\n\n")
                f.write(f"RMSE: ₹{metrics['RMSE']:,.0f}\n")
                f.write(f"MAE:  ₹{metrics['MAE']:,.0f}\n")
                f.write(f"R²:   {metrics['R2']:.4f}\n")
                
        except FileNotFoundError:
            print("No test labels found - predictions only")
    
    os.makedirs('results', exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    print(f"\nSample Predictions:")
    print(results.head(10).to_string(index=False, float_format='%.0f'))
    
    return results


def compare_all_models():
    print("="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    models = {
        'Linear Regression': 'models/regression_model1.pkl',
        'Polynomial Regression': 'models/regression_model2.pkl',
        'Lasso Regression': 'models/regression_model3.pkl',
        'Best Model': 'models/regression_model_final.pkl'
    }
    
    X_test = pd.read_csv('data/X_test.csv').to_numpy().astype(np.float64)
    y_test = pd.read_csv('data/y_test.csv').values.ravel()
    y_test_original = inverse_log_transform(y_test)
    
    comparison_results = []
    
    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"Model {model_name} not found at {model_path}")
            continue
            
        print(f"\nTesting {model_name}...")
        
        model_data = load_model(model_path)
        
        if isinstance(model_data, dict) and 'transformer' in model_data:
            transformer = model_data['transformer']
            model = model_data['model']
            degree = model_data.get('degree', 2)
            X_transformed = transformer.fit_transform(X_test)
            y_pred_log = model.predict(X_transformed)
            print(f"  Polynomial degree: {degree}")
        else:
            model = model_data
            y_pred_log = model.predict(X_test)
        
        y_pred_original = inverse_log_transform(y_pred_log)
        
        metrics = calculate_metrics(y_test_original, y_pred_original)
        
        comparison_results.append({
            'Model': model_name,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'R2': metrics['R2']
        })
        
        print(f"  RMSE: ₹{metrics['RMSE']:,.0f}")
        print(f"  MAE:  ₹{metrics['MAE']:,.0f}")
        print(f"  R²:   {metrics['R2']:.4f}")
    
    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df.to_csv('results/model_comparison.csv', index=False)
        print(f"\nComparison saved to 'results/model_comparison.csv'")
        
        best_model = comparison_df.loc[comparison_df['RMSE'].idxmin()]
        print(f"\nBest Model: {best_model['Model']}")
        print(f"Best RMSE: ₹{best_model['RMSE']:,.0f}")


def main():
    print("="*60)
    print("LAPTOP PRICE PREDICTION")
    print("="*60)
    
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'compare':
            compare_all_models()
        elif sys.argv[1] == 'test':
            make_predictions(
                'models/regression_model_final.pkl',
                'data/X_test.csv',
                'results/test_predictions.csv'
            )
        else:
            print("Usage: python predict.py [compare|test]")
    else:
        make_predictions(
            'models/regression_model_final.pkl',
            'data/X_test.csv',
            'results/predictions.csv'
        )


if __name__ == '__main__':
    main()

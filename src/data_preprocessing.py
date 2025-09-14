"""
Data Preprocessing Pipeline for Laptop Price Prediction
Based on EDA findings - prepares data for linear regression models
"""

import pandas as pd
import numpy as np
import re
from scipy import stats

def load_and_clean_data(file_path):
    """Load data and perform basic cleaning"""
    print("Loading data...")
    data = pd.read_csv(file_path)
    
    # Remove duplicates (25 found in EDA)
    print(f"Removing {data.duplicated().sum()} duplicate records...")
    data = data.drop_duplicates()
    
    # Convert Weight to numerical
    print("Converting Weight to numerical...")
    data['Weight'] = data['Weight'].str.extract('(\d+\.?\d*)').astype(float)
    
    print(f"Cleaned data shape: {data.shape}")
    return data

def extract_numerical_features(data):
    """Extract numerical values from text fields"""
    print("Extracting numerical features...")
    
    # Extract RAM in GB
    data['Ram_GB'] = data['Ram'].str.extract('(\d+)').astype(float)
    
    # Extract storage capacity in GB
    storage_numbers = data['Memory'].str.findall(r'(\d+(?:\.\d+)?)\s*(?:GB|TB)', flags=re.IGNORECASE)
    data['Storage_Capacity_GB'] = storage_numbers.apply(
        lambda x: sum([float(num) * (1000 if 'TB' in str(x).upper() else 1) for num in x]) if x else 0
    )
    
    return data

def create_categorical_features(data):
    """Create categorical features from complex text fields"""
    print("Creating categorical features...")
    
    # CPU Brand extraction
    data['CPU_Brand'] = data['Cpu'].str.extract('(Intel|AMD)', flags=re.IGNORECASE)
    data['CPU_Brand'] = data['CPU_Brand'].fillna('Other')
    
    # CPU Series extraction
    data['CPU_Series'] = data['Cpu'].str.extract('(i[3579]|Ryzen [3579]|Core [mi][3579]|Celeron|Pentium|Atom)', flags=re.IGNORECASE)
    
    # GPU Brand extraction
    data['GPU_Brand'] = 'Unknown'
    data.loc[data['Gpu'].str.contains('Intel', case=False, na=False), 'GPU_Brand'] = 'Intel'
    data.loc[data['Gpu'].str.contains('Nvidia|GeForce|GTX|RTX', case=False, na=False), 'GPU_Brand'] = 'Nvidia'
    data.loc[data['Gpu'].str.contains('AMD|Radeon', case=False, na=False), 'GPU_Brand'] = 'AMD'
    
    # Storage Type extraction
    data['Storage_Type'] = 'Unknown'
    data.loc[data['Memory'].str.contains('SSD', case=False, na=False), 'Storage_Type'] = 'SSD'
    data.loc[data['Memory'].str.contains('HDD', case=False, na=False), 'Storage_Type'] = 'HDD'
    data.loc[data['Memory'].str.contains('Flash', case=False, na=False), 'Storage_Type'] = 'Flash'
    data.loc[data['Memory'].str.contains('SSD.*HDD|HDD.*SSD', case=False, na=False), 'Storage_Type'] = 'Hybrid'
    
    # Resolution Category extraction
    data['Resolution_Category'] = 'Unknown'
    data.loc[data['ScreenResolution'].str.contains('1366x768', case=False, na=False), 'Resolution_Category'] = 'HD'
    data.loc[data['ScreenResolution'].str.contains('1920x1080|Full HD', case=False, na=False), 'Resolution_Category'] = 'Full HD'
    data.loc[data['ScreenResolution'].str.contains('2560x1440', case=False, na=False), 'Resolution_Category'] = 'QHD'
    data.loc[data['ScreenResolution'].str.contains('3840x2160|4K', case=False, na=False), 'Resolution_Category'] = '4K'
    
    # Touchscreen binary feature
    data['Touchscreen'] = data['ScreenResolution'].str.contains('Touchscreen', case=False, na=False)
    
    return data

def encode_categorical_features(data):
    """Encode categorical variables for linear regression"""
    print("Encoding categorical features...")
    
    # One-hot encoding for low-cardinality features
    one_hot_features = ['TypeName', 'OpSys', 'Storage_Type', 'Resolution_Category', 'Touchscreen']
    data_encoded = pd.get_dummies(data, columns=one_hot_features, prefix=one_hot_features)
    
    # Label encoding for high-cardinality features
    label_encode_features = ['Company', 'CPU_Brand', 'CPU_Series', 'GPU_Brand']
    
    for col in label_encode_features:
        # Create mapping for consistent encoding
        unique_values = data[col].unique()
        mapping = {val: idx for idx, val in enumerate(unique_values)}
        data_encoded[col + '_encoded'] = data[col].map(mapping)
    
    return data_encoded

def scale_numerical_features(data):
    """Scale numerical features for linear regression"""
    print("Scaling numerical features...")
    
    # Features to scale
    numerical_features = ['Inches', 'Weight', 'Ram_GB', 'Storage_Capacity_GB']
    
    # Manual standardization (mean=0, std=1)
    for feature in numerical_features:
        if feature in data.columns:
            mean_val = data[feature].mean()
            std_val = data[feature].std()
            data[feature + '_scaled'] = (data[feature] - mean_val) / std_val
    
    return data

def transform_target_variable(data):
    """Apply log transformation to handle price skewness"""
    print("Transforming target variable...")
    
    # Log transformation to handle right skewness (1.56 from EDA)
    data['Price_log'] = np.log1p(data['Price'])
    
    print(f"Original price skewness: {data['Price'].skew():.3f}")
    print(f"Log-transformed skewness: {data['Price_log'].skew():.3f}")
    
    return data

def prepare_features_and_target(data):
    """Prepare final feature matrix and target variable"""
    print("Preparing final features and target...")
    
    # Select features for modeling
    feature_cols = []
    
    # Scaled numerical features
    numerical_features = ['Inches_scaled', 'Weight_scaled', 'Ram_GB_scaled', 'Storage_Capacity_GB_scaled']
    feature_cols.extend([col for col in numerical_features if col in data.columns])
    
    # One-hot encoded features
    one_hot_cols = [col for col in data.columns if any(prefix in col for prefix in ['TypeName_', 'OpSys_', 'Storage_Type_', 'Resolution_Category_', 'Touchscreen_'])]
    feature_cols.extend(one_hot_cols)
    
    # Label encoded features
    label_encoded_cols = [col for col in data.columns if col.endswith('_encoded')]
    feature_cols.extend(label_encoded_cols)
    
    # Create feature matrix
    X = data[feature_cols].fillna(0)  # Fill any remaining NaN with 0
    y = data['Price_log']  # Use log-transformed target
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Features used: {len(feature_cols)}")
    
    return X, y, feature_cols

def split_data(X, y, test_size=0.1, random_state=42):
    """Split data into train and test sets"""
    print(f"Splitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test...")
    
    # Manual train-test split
    np.random.seed(random_state)
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Random indices for test set
    test_indices = np.random.choice(n_samples, n_test, replace=False)
    train_indices = np.setdiff1d(np.arange(n_samples), test_indices)
    
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def preprocess_data(file_path):
    """Complete preprocessing pipeline"""
    print("="*60)
    print("STARTING DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Step 1: Load and clean data
    data = load_and_clean_data(file_path)
    
    # Step 2: Extract numerical features
    data = extract_numerical_features(data)
    
    # Step 3: Create categorical features
    data = create_categorical_features(data)
    
    # Step 4: Encode categorical features
    data = encode_categorical_features(data)
    
    # Step 5: Scale numerical features
    data = scale_numerical_features(data)
    
    # Step 6: Transform target variable
    data = transform_target_variable(data)
    
    # Step 7: Prepare features and target
    X, y, feature_cols = prepare_features_and_target(data)
    
    # Step 8: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("="*60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_cols,
        'original_data': data
    }

def inverse_transform_predictions(y_pred_log):
    """Convert log predictions back to original price scale"""
    return np.expm1(y_pred_log)

# Example usage
if __name__ == "__main__":
    # Run preprocessing pipeline
    file_path = "/Users/anurag/Desktop/Anurag_Dey_A1/laptop_price_task/data/Laptop_Price.csv"
    
    try:
        processed_data = preprocess_data(file_path)
        
        print("\nPreprocessing Summary:")
        print(f"Training features: {processed_data['X_train'].shape}")
        print(f"Test features: {processed_data['X_test'].shape}")
        print(f"Training target: {processed_data['y_train'].shape}")
        print(f"Test target: {processed_data['y_test'].shape}")
        
        # Save processed data for model training
        processed_data['X_train'].to_csv('data/X_train.csv', index=False)
        processed_data['X_test'].to_csv('data/X_test.csv', index=False)
        processed_data['y_train'].to_csv('data/y_train.csv', index=False)
        processed_data['y_test'].to_csv('data/y_test.csv', index=False)
        
        print("\nProcessed data saved to CSV files")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")

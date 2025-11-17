import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.config import RANDOM_SEED, SEPARATOR_WIDTH, CLEANED_DATA_FILE, TARGET_COLUMN, TEST_SIZE

MODELS_DIR = "saved_models"

def load_cleaned_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def prepare_data(df):
    numeric_df = df.select_dtypes(include=[np.number])

    if TARGET_COLUMN not in numeric_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in data")

    X = numeric_df.drop(columns=[TARGET_COLUMN])
    y = numeric_df[TARGET_COLUMN]

    print(f"\nFeatures: {X.shape[1]} columns")
    print(f"Target: {TARGET_COLUMN}")

    return X, y

def train_model(X, y, optimize_hyperparams=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    if optimize_hyperparams:
        print("\nOptimizing Random Forest hyperparameters...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None], 
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
            #TODO: tune it if time 
        }
        model = GridSearchCV(
            RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        model.fit(X_train, y_train)
        print(f"\nBest parameters:")
        for param, value in model.best_params_.items():
            print(f"  {param}: {value}")
        print(f"Best CV R² score: {model.best_score_:.4f}")
        model = model.best_estimator_
    else:
        print("\nTraining Random Forest with default parameters...")
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print("\n" + "=" * SEPARATOR_WIDTH)
    print("MODEL EVALUATION - Random Forest")
    print("=" * SEPARATOR_WIDTH)

    print("\nTrain Metrics:")
    print(f"  R²    : {train_r2:.4f}")
    print(f"  MAE   : {train_mae:.4f}")
    print(f"  RMSE  : {train_rmse:.4f}")

    print("\nTest Metrics:")
    print(f"  R²    : {test_r2:.4f}")
    print(f"  MAE   : {test_mae:.4f}")
    print(f"  RMSE  : {test_rmse:.4f}")

    # Feature importance
    print("\nFeature Importance (Top 10):")
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['Feature']:40s} : {row['Importance']:8.4f}")

    return {
        'train_r2': train_r2,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse
    }

def save_model(model):
    os.makedirs(MODELS_DIR, exist_ok=True)
    filename = f"{MODELS_DIR}/random_forest_model.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved: {filename}")

def main(optimize_hyperparams=True):
    df = load_cleaned_data(CLEANED_DATA_FILE)

    X, y = prepare_data(df)

    model, X_train, X_test, y_train, y_test = train_model(
        X, y,
        optimize_hyperparams=optimize_hyperparams
    )

    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

    save_model(model)

    print("\n" + "=" * SEPARATOR_WIDTH)
    print("TRAINING COMPLETED")
    print("=" * SEPARATOR_WIDTH)

if __name__ == "__main__":
    main()

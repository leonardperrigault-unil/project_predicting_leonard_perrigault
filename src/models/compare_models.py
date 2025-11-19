import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.config import RANDOM_SEED, SEPARATOR_WIDTH, CLEANED_DATA_FILE, TARGET_COLUMN, TEST_SIZE

MODELS_DIR = "saved_models"
RESULTS_DIR = "results"

def load_model(model_type):
    filename = f"{MODELS_DIR}/{model_type}_model.pkl"
    if not os.path.exists(filename):
        return None

    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def get_available_models():
    if not os.path.exists(MODELS_DIR):
        return []

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    available_models = []

    for f in model_files:
        model_type = f.replace('_model.pkl', '')
        available_models.append(model_type)

    return available_models

def load_and_prepare_data():
    df = pd.read_csv(CLEANED_DATA_FILE)
    numeric_df = df.select_dtypes(include=[np.number])

    X = numeric_df.drop(columns=[TARGET_COLUMN])
    y = numeric_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # Scale data for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        'scaled': (X_train_scaled, X_test_scaled),
        'unscaled': (X_train.values, X_test.values),
        'y': (y_train, y_test),
        'feature_names': X.columns
    }

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }

def plot_metrics_comparison(results):
    models = list(results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # R² comparison (higher is better)
    r2_values = [results[m]['r2'] for m in models]
    best_r2_idx = r2_values.index(max(r2_values))
    r2_colors = ['green' if i == best_r2_idx else 'steelblue' for i in range(len(models))]
    axes[0].bar(models, r2_values, color=r2_colors)
    axes[0].set_title('R² Score')
    axes[0].set_ylabel('R²')
    axes[0].tick_params(axis='x', rotation=45)

    # MAE comparison (lower is better)
    mae_values = [results[m]['mae'] for m in models]
    best_mae_idx = mae_values.index(min(mae_values))
    mae_colors = ['green' if i == best_mae_idx else 'steelblue' for i in range(len(models))]
    axes[1].bar(models, mae_values, color=mae_colors)
    axes[1].set_title('MAE')
    axes[1].set_ylabel('MAE')
    axes[1].tick_params(axis='x', rotation=45)

    # RMSE comparison (lower is better)
    rmse_values = [results[m]['rmse'] for m in models]
    best_rmse_idx = rmse_values.index(min(rmse_values))
    rmse_colors = ['green' if i == best_rmse_idx else 'steelblue' for i in range(len(models))]
    axes[2].bar(models, rmse_values, color=rmse_colors)
    axes[2].set_title('RMSE')
    axes[2].set_ylabel('RMSE')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    comparison_dir = f"{RESULTS_DIR}/comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    filename = f"{comparison_dir}/metrics_comparison.png"
    plt.savefig(filename, dpi=150)
    print(f"Saved: {filename}")
    plt.close()

def plot_feature_importance_comparison(models_dict, feature_names):
    fig, axes = plt.subplots(1, len(models_dict), figsize=(6*len(models_dict), 6))

    if len(models_dict) == 1:
        axes = [axes]

    for idx, (name, model) in enumerate(models_dict.items()):
        if hasattr(model, 'coef_'):
            importance = model.coef_
            xlabel = 'Coefficient'
        elif hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            xlabel = 'Importance'
        else:
            continue

        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance,
            'Abs_Importance': np.abs(importance)
        })
        feature_importance = feature_importance.sort_values('Abs_Importance', ascending=False).head(10)

        axes[idx].barh(feature_importance['Feature'], feature_importance['Importance'])
        axes[idx].set_xlabel(xlabel)
        axes[idx].set_title(f'{name.replace("_", " ").title()}\nTop 10 Features')
        axes[idx].invert_yaxis()
        axes[idx].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    comparison_dir = f"{RESULTS_DIR}/comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    filename = f"{comparison_dir}/feature_importance_comparison.png"
    plt.savefig(filename, dpi=150)
    print(f"Saved: {filename}")
    plt.close()

def main():
    print("=" * SEPARATOR_WIDTH)
    print("MODEL COMPARISON")
    print("=" * SEPARATOR_WIDTH)

    available_models = get_available_models()

    if not available_models:
        print("No saved models found. Please train models first.")
        return

    print(f"\nFound {len(available_models)} saved models: {', '.join(available_models)}")

    # Load data
    data = load_and_prepare_data()
    y_train, y_test = data['y']

    print(f"Test set: {len(y_test)} samples")

    # Evaluate all models
    results = {}
    models_dict = {}

    # Models that don't need scaling
    no_scale_models = ['random_forest', 'xgboost']

    print("\nEvaluating models...")

    for model_type in available_models:
        model = load_model(model_type)
        if model is None:
            continue

        models_dict[model_type] = model

        # Choose scaled or unscaled data
        if model_type in no_scale_models:
            X_test = data['unscaled'][1]
        else:
            X_test = data['scaled'][1]

        results[model_type] = evaluate_model(model, X_test, y_test)
        print(f"  {model_type}: R² = {results[model_type]['r2']:.4f}")

    # Display comparison table
    print("\n" + "=" * SEPARATOR_WIDTH)
    print("RESULTS SUMMARY")
    print("=" * SEPARATOR_WIDTH)

    print(f"\n{'Model':<20} {'R²':<10} {'MAE':<10} {'RMSE':<10}")
    print("-" * 50)

    # Sort by R² score
    sorted_models = sorted(results.keys(), key=lambda x: results[x]['r2'], reverse=True)

    for model_type in sorted_models:
        r2 = results[model_type]['r2']
        mae = results[model_type]['mae']
        rmse = results[model_type]['rmse']
        print(f"{model_type:<20} {r2:<10.4f} {mae:<10.4f} {rmse:<10.4f}")

    # Find best model
    best_model = sorted_models[0]
    print(f"\nBest model: {best_model} (R² = {results[best_model]['r2']:.4f})")

    # Generate plots
    print("\n" + "=" * SEPARATOR_WIDTH)
    print("GENERATING COMPARISON PLOTS")
    print("=" * SEPARATOR_WIDTH)

    plot_metrics_comparison(results)
    plot_feature_importance_comparison(models_dict, data['feature_names'])

    print("\n" + "=" * SEPARATOR_WIDTH)
    print("COMPARISON COMPLETED")
    print("=" * SEPARATOR_WIDTH)

if __name__ == "__main__":
    main()

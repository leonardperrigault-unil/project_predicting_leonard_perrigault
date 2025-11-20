import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import RANDOM_SEED, SEPARATOR_WIDTH, CLEANED_DATA_FILE, TARGET_COLUMN, TEST_SIZE

MODELS_DIR = "saved_models"
RESULTS_DIR = "results"

def load_model(model_type):
    filename = f"{MODELS_DIR}/{model_type}_model.pkl"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model not found: {filename}")

    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Loaded model: {filename}")
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

def load_and_prepare_data(scale_data=True):
    df = pd.read_csv(CLEANED_DATA_FILE)
    numeric_df = df.select_dtypes(include=[np.number])

    X = numeric_df.drop(columns=[TARGET_COLUMN])
    y = numeric_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    if scale_data:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        return X_train_scaled, X_test_scaled, y_train, y_test
    else:
        return X_train, X_test, y_train, y_test

def create_shap_explainer(model, X_train, model_type):
    # Tree-based models use TreeExplainer
    if model_type in ['random_forest', 'xgboost']:
        return shap.TreeExplainer(model)
    # Linear models use LinearExplainer
    elif model_type in ['linear', 'lasso', 'ridge']:
        return shap.LinearExplainer(model, X_train)
    else:
        # Fallback to KernelExplainer (slower but works with any model)
        return shap.KernelExplainer(model.predict, shap.sample(X_train, 100))

def plot_shap_summary(shap_values, X, model_type):
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()

    model_dir = f"{RESULTS_DIR}/{model_type}"
    os.makedirs(model_dir, exist_ok=True)
    filename = f"{model_dir}/shap_summary.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def plot_shap_bar(shap_values, X, model_type):
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()

    model_dir = f"{RESULTS_DIR}/{model_type}"
    os.makedirs(model_dir, exist_ok=True)
    filename = f"{model_dir}/shap_importance.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def plot_shap_dependence(shap_values, X, model_type, top_n=3):
    # Get top features by mean absolute SHAP value
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_shap)[-top_n:][::-1]
    top_features = X.columns[top_indices]

    model_dir = f"{RESULTS_DIR}/{model_type}"
    os.makedirs(model_dir, exist_ok=True)

    for feature in top_features:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values.values, X, show=False)
        plt.tight_layout()
        filename = f"{model_dir}/shap_dependence_{feature}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

def main():
    print("=" * SEPARATOR_WIDTH)
    print("SHAP ANALYSIS")
    print("=" * SEPARATOR_WIDTH)

    available_models = get_available_models()

    if not available_models:
        print("No saved models found. Please train a model first.")
        return

    print("\nAvailable saved models:")
    for idx, model in enumerate(available_models, 1):
        print(f"{idx}. {model.capitalize()}")

    choice = input(f"\nSelect model to analyze (1-{len(available_models)}): ").strip()

    try:
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(available_models):
            model_type = available_models[choice_idx]
        else:
            print(f"Invalid choice. Please select 1-{len(available_models)}.")
            return
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    print(f"\nAnalyzing: {model_type}")

    # Load model
    model = load_model(model_type)

    # Load data (tree models don't need scaling)
    needs_scaling = model_type not in ['random_forest', 'xgboost']
    X_train, X_test, y_train, y_test = load_and_prepare_data(scale_data=needs_scaling)

    print(f"Test set: {X_test.shape[0]} samples")

    # Create SHAP explainer
    print("\nCreating SHAP explainer...")
    explainer = create_shap_explainer(model, X_train, model_type)

    # Calculate SHAP values
    print("Calculating SHAP values (this may take a moment)...")
    shap_values = explainer(X_test)

    # Generate plots
    print("\n" + "=" * SEPARATOR_WIDTH)
    print("GENERATING SHAP PLOTS")
    print("=" * SEPARATOR_WIDTH)

    plot_shap_summary(shap_values, X_test, model_type)
    plot_shap_bar(shap_values, X_test, model_type)
    plot_shap_dependence(shap_values, X_test, model_type)

    print("\n" + "=" * SEPARATOR_WIDTH)
    print("SHAP ANALYSIS COMPLETED")
    print("=" * SEPARATOR_WIDTH)

if __name__ == "__main__":
    main()

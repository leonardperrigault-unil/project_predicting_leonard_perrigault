import pandas as pd
import numpy as np
from src.config import SEPARATOR_WIDTH, RAW_DATA_FILE, CLEANED_DATA_FILE

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def load_data(filepath):
    """Load the dataset from CSV file"""
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")
    return df

def explore_data(df):
    """Initial exploration of the dataset"""
    print("=" * SEPARATOR_WIDTH)
    print("INITIAL DATA EXPLORATION")
    print("=" * SEPARATOR_WIDTH)

    print("\nFirst few rows:")
    print(df.head())

    print("\n\nDataset Info:")
    print(df.info())

    print("\n\nBasic Statistics:")
    print(df.describe())

    print("\n\nMissing Values Summary:")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Percentage', ascending=False)
    print(missing_df)

    return missing_df

def clean_data(df):
    """Main data cleaning function"""
    print("\n" + "=" * SEPARATOR_WIDTH)
    print("STARTING DATA CLEANING")
    print("=" * SEPARATOR_WIDTH)

    df_clean = df.copy()

    # 1. Convert string-formatted numeric columns
    print("\n1. Converting string-formatted numeric columns...")
    object_cols = df_clean.select_dtypes(include=['object']).columns
    converted_count = 0

    for col in object_cols:
        if col in ['Country', 'Abbreviation', 'Capital/Major City', 'Currency-Code',
                   'Largest city', 'Official language']:
            continue

        df_clean[col] = df_clean[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', '').str.strip()
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        if df_clean[col].dtype in ['float64', 'int64']:
            converted_count += 1

    print(f"   Converted {converted_count} columns to numeric")

    # 2. Handle missing values in Life expectancy
    print("\n2. Handling missing values in Life expectancy...")
    if 'Life expectancy' in df_clean.columns:
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=['Life expectancy'])
        removed = initial_rows - len(df_clean)
        print(f"   Removed {removed} rows with missing life expectancy values")

    # 3. Identify numeric and categorical columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

    print(f"\n3. Identified {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns")

    # 4. Handle missing values in numeric columns
    print("\n4. Handling missing values in numeric columns...")
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            missing_pct = (df_clean[col].isnull().sum() / len(df_clean)) * 100

            # If more than 50% missing, consider dropping the column
            if missing_pct > 50:
                print(f"   Dropping column '{col}' ({missing_pct:.1f}% missing)")
                df_clean = df_clean.drop(columns=[col])
            else:
                # Use median imputation for numeric columns
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                print(f"   Filled '{col}' with median value ({median_val:.2f})")

    # 5. Handle missing values in categorical columns
    print("\n5. Handling missing values in categorical columns...")
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            missing_pct = (df_clean[col].isnull().sum() / len(df_clean)) * 100

            if missing_pct > 50:
                print(f"   Dropping column '{col}' ({missing_pct:.1f}% missing)")
                df_clean = df_clean.drop(columns=[col])
            else:
                # For categorical columns: fill missing values with the most frequent value (mode).
                mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col] = df_clean[col].fillna(mode_val)
                print(f"   Filled '{col}' with mode value")

    # 6. Remove duplicates
    print("\n6. Checking for duplicates...")
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"   Removed {duplicates} duplicate rows")
    else:
        print("   No duplicates found")

    # 7. Handle outliers using IQR method for key numeric columns (IQR = Q3-Q1)
    print("\n7. Detecting outliers in numeric columns...")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    outlier_summary = {}

    for col in numeric_cols:
        if col != 'Life expectancy':  # Don't remove outliers from target variable
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # Using 3*IQR for less aggressive outlier removal
            upper_bound = Q3 + 3 * IQR

            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            if outliers > 0:
                outlier_summary[col] = outliers

    if outlier_summary:
        print(f"   Found outliers in {len(outlier_summary)} columns:")
        for col, count in sorted(outlier_summary.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"     - {col}: {count} outliers")
        print(" Outliers are kept but flagged for awareness")

    print(f"\n8. Cleaned dataset: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")

    return df_clean


def save_cleaned_data(df, output_path):
    """Save the cleaned dataset"""
    print("\n" + "=" * SEPARATOR_WIDTH)
    print("SAVING CLEANED DATA")
    print("=" * SEPARATOR_WIDTH)

    df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")
    print(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns")

def main():
    """Main execution function"""
    # Load data
    df = load_data(RAW_DATA_FILE)

    # Explore data
    missing_df = explore_data(df)

    # Clean data
    df_clean = clean_data(df)

    # Save cleaned data
    save_cleaned_data(df_clean, CLEANED_DATA_FILE)

    print("\n" + "=" * SEPARATOR_WIDTH)
    print("DATA CLEANING COMPLETED!")
    print("=" * SEPARATOR_WIDTH)

if __name__ == "__main__":
    main()

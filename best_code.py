import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def quick_feature_engineering(df):
    # creating the most important features quickly
    print("🔧 Creating key features...")

    # Most important interaction features
    df['EconomicVulnerability'] = df['PctPopUnderPov'] * df['PctUnemployed'] / 100
    df['EducationAdvantage'] = df['PctBSorMore'] / (df['PctNotHSGrad'] + 1)
    df['IncomePovertyRatio'] = np.log1p(df['medIncome']) / (df['PctPopUnderPov'] + 1)

    # Removing the most correlated features
    drop_cols = ['numbUrban', 'NumUnderPov', 'medFamInc', 'perCapInc',
                 'NumKidsBornNeverMar', 'NumImmig']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    return df

def quick_model_training(X_train, y_train, X_val, y_val):
    # Training the 3 best models quickly
    print("Training models...")

    models = {}

    # 1. Ridge Regression
    scaler_for_ridge = StandardScaler()
    X_train_scaled = scaler_for_ridge.fit_transform(X_train)
    X_val_scaled = scaler_for_ridge.transform(X_val)

    ridge = Ridge(alpha=10.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    models['Ridge'] = (ridge, scaler_for_ridge)
    # ------------------------------------------------------------------------------
    # 2. Random Forest
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [15, 20],
        'min_samples_split': [5, 10]
    }
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        rf_params,
        n_iter=8,
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    models['RandomForest'] = (rf_search.best_estimator_, None)
    # ------------------------------------------------------------------------------
    # 3. XGBoost
    xgb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.15],
        'max_depth': [5, 7]
    }
    xgb_search = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=42),
        xgb_params,
        n_iter=8,
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    models['XGBoost'] = (xgb_search.best_estimator_, None)
    # ------------------------------------------------------------------------------
    # Evaluate models
    best_model = None
    best_rmse = float('inf')

    for name, (model, scaler) in models.items():
        X_val_pred = X_val if scaler is None else scaler.transform(X_val)
        pred = model.predict(X_val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        r2 = r2_score(y_val, pred)
        print(f"   {name}: RMSE={rmse:.2f}, R²={r2:.3f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = (name, model, scaler)

    return best_model, models

def main():
    print("=" * 50)

    # Load data
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        print(f"Loaded: Train {train_df.shape}, Test {test_df.shape}")
    except Exception as e:
        print(f"Error loading data files: {e}")
        return

    # Feature engineering
    train_processed = quick_feature_engineering(train_df.copy())
    test_processed = quick_feature_engineering(test_df.copy())

    # Prepare features
    feature_cols = [
        col for col in train_processed.columns
        if col not in ['ID', 'ViolentCrimesPerPop']
        and train_processed[col].dtype in ['int64', 'float64']
    ]

    X = train_processed[feature_cols].fillna(0)
    y = train_processed['ViolentCrimesPerPop']
    X_test = test_processed[feature_cols].fillna(0)

    print(f"Features: {len(feature_cols)}")

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train models
    best_model, all_models = quick_model_training(X_train, y_train, X_val, y_val)

    # Report best model on validation
    best_name, best_model_obj, best_scaler = best_model
    X_val_best = X_val if best_scaler is None else best_scaler.transform(X_val)
    best_val_rmse = np.sqrt(mean_squared_error(y_val, best_model_obj.predict(X_val_best)))
    print(f"\nBest model: {best_name} (RMSE: {best_val_rmse:.2f})")

    # Create ensemble prediction (simple average of the three models)
    print("Creating ensemble...")
    ensemble_pred = np.zeros(len(X_test))

    for name, (model, scaler) in all_models.items():
        X_test_pred = X_test if scaler is None else scaler.transform(X_test)
        pred = model.predict(X_test_pred)
        ensemble_pred += pred

    ensemble_pred /= len(all_models)

    # Generate single submission only
    submission = pd.DataFrame({
        'ID': test_processed['ID'],
        'ViolentCrimesPerPop': ensemble_pred
    })

    submission.to_csv('quick_submission.csv', index=False)
    print(f"Submission saved as: quick_submission.csv")

if __name__ == "__main__":
    main()
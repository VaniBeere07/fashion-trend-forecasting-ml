import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

def generate_fashion_dataset(n_weeks=104):
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=n_weeks, freq="W-MON")
    t = np.arange(n_weeks)
    macro_trend = 0.6 * t
    seasonality = 25 * np.sin(2 * np.pi * t / 52.14) + 12 * np.cos(4 * np.pi * t / 52.14)
    svi_signal = 50 + 22 * np.sin(2 * np.pi * t / 52.14 + 0.3) + np.random.normal(0, 4, n_weeks)
    promo = (np.random.rand(n_weeks) > 0.85).astype(int)
    base_demand = 110 + macro_trend + seasonality + (svi_signal * 0.75) + (promo * 40)
    demand = np.maximum(0, base_demand + np.random.normal(0, 6, n_weeks)).astype(int)
    return pd.DataFrame({
        "date": dates,
        "week_num": dates.isocalendar().week.astype(int),
        "search_svi": svi_signal,
        "promo_active": promo,
        "unit_demand": demand
    })

def build_features(df):
    df = df.copy()
    df["sin_week"] = np.sin(2 * np.pi * df["week_num"] / 52.14)
    df["cos_week"] = np.cos(2 * np.pi * df["week_num"] / 52.14)
    for lag in [1, 2, 4]:
        df[f"demand_lag_{lag}"] = df["unit_demand"].shift(lag)
        df[f"svi_lag_{lag}"] = df["search_svi"].shift(lag)
    df["svi_rolling_4w"] = df["svi_lag_1"].rolling(4).mean()
    df["svi_momentum"] = df["svi_lag_1"] / (df["svi_rolling_4w"] + 1e-5)
    return df.dropna().reset_index(drop=True)

def train_and_evaluate(df):
    split = int(len(df) * 0.8)
    features = [
        "sin_week", "cos_week", "promo_active",
        "demand_lag_1", "demand_lag_2", "demand_lag_4",
        "svi_lag_1", "svi_lag_2", "svi_lag_4", "svi_momentum"
    ]
    target = "unit_demand"
    X_train, y_train = df.loc[:split, features], df.loc[:split, target]
    X_test, y_test = df.loc[split:, features], df.loc[split:, target]
    model = lgb.LGBMRegressor(objective="tweedie", n_estimators=300, learning_rate=0.03, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    wape = (np.sum(np.abs(y_test - preds)) / np.sum(y_test)) * 100
    mae = mean_absolute_error(y_test, preds)
    return model, wape, mae

if __name__ == "__main__":
    df_raw = generate_fashion_dataset()
    df_feat = build_features(df_raw)
    model, wape, mae = train_and_evaluate(df_feat)
    print(f"Validation WAPE: {wape:.2f}%, MAE: {mae:.2f} units")

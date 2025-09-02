import pandas as pd
from datetime import timedelta
import numpy as np

def simple_forecast(df, days_ahead=5):
    """
    株価の終値に対してシンプルな線形外挿による予測を行う（デモ用）
    ※ AIではなくあくまで土台作成
    """
    if df.empty or len(df) < 2:
        return None

    df = df.copy()
    df = df[["Close"]].dropna()

    # 線形近似（最も単純なAI代用）
    df["Day"] = range(len(df))
    coeffs = np.polyfit(df["Day"], df["Close"], 1)
    slope, intercept = coeffs

    # 予測データ生成
    last_day = df["Day"].iloc[-1]
    future_days = list(range(last_day + 1, last_day + days_ahead + 1))
    forecast_dates = [df.index[-1] + timedelta(days=i) for i in range(1, days_ahead + 1)]
    forecast_prices = [slope * day + intercept for day in future_days]

    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Forecast": forecast_prices
    }).set_index("Date")

    return forecast_df

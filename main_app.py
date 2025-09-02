import streamlit as st
from data_fetcher import get_stock_data
from chart import show_line_chart, show_candlestick_chart  # ← 追加

st.title("📈 株価チャートアプリ（分離版）")

ticker_list = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "NVDA", "AMD", "NFLX", "COIN"]

ticker = st.selectbox("ウォッチリストからティッカーを選んでください", ticker_list, index=0)

period = st.selectbox("期間", ["1mo", "3mo", "6mo", "1y", "5y", "max"], index=2)

chart_type = st.radio("チャート種類を選んでください", ["ラインチャート", "ローソク足チャート"])

if ticker:
    try:
        df = get_stock_data(ticker, period)
        if not df.empty:
            if chart_type == "ラインチャート":
                show_line_chart(df, ticker)
            else:
                show_candlestick_chart(df, ticker)
        else:
            st.warning("データが取得できませんでした。")
    except Exception as e:
        st.error(f"エラー: {e}")

from predictor import simple_forecast  # ← 新たに追加

# ...（中略）...

if ticker:
    try:
        df = get_stock_data(ticker, period)
        if not df.empty:
            # チャートの表示
            if chart_type == "ラインチャート":
                show_line_chart(df, ticker)
            else:
                show_candlestick_chart(df, ticker)

            # 🔮 予測の表示
            st.markdown("---")
            st.subheader("🔮 株価予測（デモ）")

            forecast_df = simple_forecast(df, days_ahead=5)
            if forecast_df is not None:
                st.line_chart(forecast_df["Forecast"])
            else:
                st.warning("予測データの生成に失敗しました。")
        else:
            st.warning("データが取得できませんでした。")
    except Exception as e:
        st.error(f"エラー: {e}")

import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def show_line_chart(data, ticker):
    """
    ラインチャートを表示する関数
    """
    st.subheader(f"{ticker} の終値チャート")
    st.line_chart(data["Close"])

def show_candlestick_chart(data, ticker):
    """
    ローソク足チャートを表示する関数（Plotly）
    """
    st.subheader(f"{ticker} のローソク足チャート")

    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])

    fig.update_layout(
        xaxis_title='日付',
        yaxis_title='価格（USD）',
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

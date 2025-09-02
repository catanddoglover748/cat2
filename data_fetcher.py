import yfinance as yf

def get_stock_data(ticker, period="6mo"):
    """
    指定したティッカーと期間の株価データを取得する関数。
    """
    data = yf.Ticker(ticker).history(period=period)
    return data

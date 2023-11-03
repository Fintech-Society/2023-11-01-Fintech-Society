import yfinance as yf
 
TICKERS = ("MSFT", "AAPL", "AMZN", "KO","NVDA")

data = yf.download(TICKERS, period="5y")

data["Adj Close"].to_excel("data.xlsx")
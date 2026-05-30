import requests
import os
import yfinance as yf
from langchain.tools import tool

@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price, market cap, P/E ratio, and 52-week range for a stock ticker symbol (e.g. TSLA, AAPL, MSFT)."""
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        price = info.get("currentPrice") or info.get("regularMarketPrice", "N/A")
        market_cap = info.get("marketCap", "N/A")
        pe_ratio = info.get("trailingPE", "N/A")
        week_high = info.get("fiftyTwoWeekHigh", "N/A")
        week_low = info.get("fiftyTwoWeekLow", "N/A")
        name = info.get("longName", ticker)

        if isinstance(market_cap, (int, float)):
            market_cap = f"${market_cap / 1e9:.2f}B"

        return (
            f"📈 {name} ({ticker.upper()})\n"
            f"  Current Price: ${price}\n"
            f"  Market Cap: {market_cap}\n"
            f"  P/E Ratio: {pe_ratio}\n"
            f"  52-Week High: ${week_high} | Low: ${week_low}"
        )
    except Exception as e:
        return f"Error fetching stock data for {ticker}: {str(e)}"

@tool
def get_crypto_price(coin_id: str) -> str:
    """Get the current price, market cap, and 24h change for a cryptocurrency. Use CoinGecko IDs like 'bitcoin', 'ethereum', 'solana'."""
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id.lower()}&vs_currencies=usd&include_market_cap=true&include_24hr_change=true"
        data = requests.get(url, timeout=10).json()
        if coin_id.lower() not in data:
            return f"Coin '{coin_id}' not found. Use CoinGecko IDs like 'bitcoin', 'ethereum'."
        coin = data[coin_id.lower()]
        return (
            f"🪙 {coin_id.capitalize()}\n"
            f"  Price: ${coin['usd']:,.2f}\n"
            f"  Market Cap: ${coin['usd_market_cap']:,.0f}\n"
            f"  24h Change: {coin['usd_24h_change']:.2f}%"
        )
    except Exception as e:
        return f"Error fetching crypto data: {str(e)}"

@tool
def get_exchange_rate(base: str, target: str) -> str:
    """Convert currency exchange rates. Provide base currency (e.g. 'USD') and target currency (e.g. 'EUR', 'GBP', 'JPY')."""
    try:
        api_key = os.getenv("EXCHANGE_RATE_API_KEY")
        url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{base.upper()}/{target.upper()}"
        data = requests.get(url, timeout=10).json()
        if data.get("result") != "success":
            return f"Could not fetch rate for {base}/{target}."
        rate = data["conversion_rate"]
        return f"💱 1 {base.upper()} = {rate} {target.upper()}"
    except Exception as e:
        return f"Exchange rate error: {str(e)}"
import pandas as pd
import requests
def fetch_binance(symbol="BTCUSDT", interval="1d", limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data)
    df.columns = ["open_time","open","high","low","close","volume",
                  "close_time","quote_asset_volume","num_trades",
                  "taker_base_volume","taker_quote_volume","ignore"]
    return df
# Run only when script is executed directly
# -----------------------
if __name__ == "__main__":
    df = fetch_binance()
    df.to_csv("data/raw/btcusdt_1d.csv", index=False)
    print(f"Fetched {len(df)} rows")

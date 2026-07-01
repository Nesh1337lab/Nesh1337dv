import os
import json
import requests
import numpy as np
import pandas as pd
import ccxt

# ================== НАСТРОЙКИ ==================
TELEGRAM_TOKEN = os.getenv("TG_TOKEN", "")
CHAT_ID = os.getenv("TG_CHAT", "")

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TIMEFRAME = "1h"
MAX_BARS = 200

PIVOT_PERIOD = 5
MAX_BARS_BACK = 100
MIN_DIVERGENCES = 1

STATE_FILE = "state.json"

exchange = ccxt.bybit({'options': {'defaultType': 'spot'}})

# ================== СОСТОЯНИЕ (антиспам между запусками) ==================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

# ================== ИНДИКАТОРЫ ==================
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)

def macd_line(close):
    return ema(close, 12) - ema(close, 26)

def macd_hist(close):
    m = macd_line(close)
    return m - ema(m, 9)

def stoch(df, n=14, k=3):
    ll = df['low'].rolling(n).min()
    hh = df['high'].rolling(n).max()
    kf = 100*(df['close']-ll)/(hh-ll).replace(0, np.nan)
    return kf.rolling(k).mean()

def cci(df, n=20):
    tp = (df['high']+df['low']+df['close'])/3
    ma = tp.rolling(n).mean()
    md = (tp-ma).abs().rolling(n).mean()
    return (tp-ma)/(0.015*md.replace(0, np.nan))

def momentum(close, n=10):
    return close - close.shift(n)

def obv(df):
    sign = np.sign(df['close'].diff()).fillna(0)
    return (sign*df['volume']).cumsum()

def vwmacd(df):
    def vwma(n):
        return (df['close']*df['volume']).rolling(n).sum()/df['volume'].rolling(n).sum()
    return vwma(12)-vwma(26)

def cmf(df, n=20):
    mfm = ((df['close']-df['low'])-(df['high']-df['close']))/(df['high']-df['low']).replace(0, np.nan)
    mfv = mfm*df['volume']
    return mfv.rolling(n).sum()/df['volume'].rolling(n).sum()

def mfi(df, n=14):
    tp = (df['high']+df['low']+df['close'])/3
    mf = tp*df['volume']
    pos = mf.where(tp > tp.shift(1), 0).rolling(n).sum()
    neg = mf.where(tp < tp.shift(1), 0).rolling(n).sum()
    mr = pos/neg.replace(0, np.nan)
    return 100-100/(1+mr)

def build_indicators(df):
    return {
        "MACD": macd_line(df['close']),
        "MACD Hist": macd_hist(df['close']),
        "RSI": rsi(df['close']),
        "Stochastic": stoch(df),
        "CCI": cci(df),
        "Momentum": momentum(df['close']),
        "OBV": obv(df),
        "VWmacd": vwmacd(df),
        "CMF": cmf(df),
        "MFI": mfi(df),
    }

# ================== ПОИСК ПИВОТОВ ==================
def find_pivots(series, lb):
    highs, lows = [], []
    vals = series.tolist()
    n = len(vals)
    for i in range(lb, n - lb):
        w = vals[i-lb:i+lb+1]
        c = vals[i]
        if c == max(w) and w.count(c) == 1:
            highs.append(i)
        if c == min(w) and w.count(c) == 1:
            lows.append(i)
    return highs, lows

# ================== ЛОГИКА ДИВЕРГЕНЦИЙ ==================
def check_divergence(df):
    lb = PIVOT_PERIOD
    inds = build_indicators(df)
    high = df['high'].values
    low = df['low'].values

    ph, pl = find_pivots(df['close'], lb)
    signals = []

    if len(ph) >= 2:
        i2, i1 = ph[-1], ph[-2]
        if 0 < (i2 - i1) <= MAX_BARS_BACK and high[i2] > high[i1]:
            confirmed, names = 0, []
            for name, ser in inds.items():
                v1, v2 = ser.iloc[i1], ser.iloc[i2]
                if pd.notna(v1) and pd.notna(v2) and v2 < v1:
                    confirmed += 1
                    names.append(name)
            if confirmed >= MIN_DIVERGENCES:
                signals.append({"type": "BEARISH (Regular)", "pivot_idx": i2, "count": confirmed, "indicators": names})

    if len(pl) >= 2:
        i2, i1 = pl[-1], pl[-2]
        if 0 < (i2 - i1) <= MAX_BARS_BACK and low[i2] < low[i1]:
            confirmed, names = 0, []
            for name, ser in inds.items():
                v1, v2 = ser.iloc[i1], ser.iloc[i2]
                if pd.notna(v1) and pd.notna(v2) and v2 > v1:
                    confirmed += 1
                    names.append(name)
            if confirmed >= MIN_DIVERGENCES:
                signals.append({"type": "BULLISH (Regular)", "pivot_idx": i2, "count": confirmed, "indicators": names})

    return signals

# ================== TELEGRAM ==================
def send_telegram(text):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("[WARN] TG_TOKEN / TG_CHAT не заданы, пропускаю отправку")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=15)
        if r.status_code != 200:
            print("[TG ERROR]", r.text)
    except Exception as e:
        print("[TG EXCEPTION]", e)

# ================== ЗАГРУЗКА ДАННЫХ ==================
def fetch_df(symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=MAX_BARS)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms")
    return df

# ================== ОДИН ПРОГОН ==================
def scan_once(state):
    for symbol in SYMBOLS:
        try:
            df = fetch_df(symbol)
            if len(df) < PIVOT_PERIOD * 2 + 5:
                continue

            signals = check_divergence(df)
            for s in signals:
                key = f"{symbol}:{s['type']}:{s['pivot_idx']}"
                dedup_key = f"{symbol}|{s['type']}"
                if state.get(dedup_key) == key:
                    continue
                state[dedup_key] = key

                price = df['close'].iloc[-1]
                inds_str = ", ".join(s['indicators'])
                emoji = "🔴" if "BEARISH" in s['type'] else "🟢"
                msg = (
                    f"{emoji} <b>Дивергенция</b>\n"
                    f"Пара: <b>{symbol}</b>\n"
                    f"ТФ: {TIMEFRAME}\n"
                    f"Тип: {s['type']}\n"
                    f"Индикаторов: {s['count']} ({inds_str})\n"
                    f"Цена: {price}"
                )
                print("[SIGNAL]", msg.replace("\n", " | "))
                send_telegram(msg)

        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")

def main():
    state = load_state()
    print("Проверяю:", SYMBOLS, "ТФ:", TIMEFRAME)
    scan_once(state)
    save_state(state)

if __name__ == "__main__":
    main()

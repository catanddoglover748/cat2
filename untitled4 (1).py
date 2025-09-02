# =============================
# 1. 必要なライブラリのインポート（ファイルの先頭）
# =============================
import streamlit as st
import yfinance as yf
import finnhub
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import re

import traceback

from datetime import datetime, timedelta

# =============================
# 2. Finnhub APIクライアントの初期化（インポートのすぐ下）
# =============================
st.set_page_config(layout="wide")
api_key = st.secrets["MY_API_KEY"]
api_key = st.secrets.get("MY_API_KEY") or os.getenv("MY_API_KEY")
if not api_key:
    st.stop() 
finnhub_client = finnhub.Client(api_key=api_key)
# ============ PATCH-A: SEC実績取り込み（米国）ユーティリティ ============
import requests, time, math
from functools import lru_cache

# ⚠ SEC の User-Agent は実アプリ名/連絡先メールを入れてください（EDGAR 規約）

APP = st.secrets.get("SEC_APP_NAME", "StockEarningsDash")
MAIL = st.secrets.get("SEC_CONTACT", "gaurrd5300@gmail.com")

SEC_HEADERS = {
    "User-Agent": f"{APP}/1.0 ({MAIL})",
    "Accept-Encoding": "gzip, deflate",
}
# === SEC throttle & helper ===
import time, requests

APP  = st.secrets.get("SEC_APP_NAME", "StockEarningsDash")
MAIL = st.secrets.get("SEC_CONTACT",  "gaurrd5300@gmail.com")
SEC_HEADERS = {
    "User-Agent": f"{APP}/1.0 ({MAIL})",
    "Accept-Encoding": "gzip, deflate",
}

_last_call_ts = {"sec": 0.0}

def sec_get(url, **kw):
    wait = 1.0 - (time.monotonic() - _last_call_ts["sec"])
    if wait > 0:
        time.sleep(wait)
    kw.setdefault("headers", SEC_HEADERS)
    kw.setdefault("timeout", 30)
    r = requests.get(url, **kw)
    _last_call_ts["sec"] = time.monotonic()

    retry, backoff = 0, 1.0
    while r.status_code in (429, 500, 502, 503, 504) and retry < 3:
        time.sleep(backoff)
        r = requests.get(url, **kw)
        _last_call_ts["sec"] = time.monotonic()
        retry += 1
        backoff *= 2
    r.raise_for_status()
    return r

# =============================


# ---- キャッシュ（Streamlit 環境ならこちらを推奨）----
try:
    # Streamlit があるなら 30日キャッシュ。無ければ lru_cache が使われます
    from streamlit.runtime.caching import cache_data as _cache_data
    def cache_days(days: int):
        return _cache_data(ttl=days * 24 * 60 * 60)
except Exception:
    def cache_days(days: int):
        def deco(fn):
            return lru_cache(maxsize=64)(fn)
        return deco

# ---- 1) ティッカー -> CIK 解決 ----
@cache_days(30)
def resolve_cik(ticker: str) -> str:
    t = (ticker or "").upper().strip()
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=20)
    r = sec_get(url)
    #r.raise_for_status()
    data = r.json()  # { "0": {"ticker":"AAPL","cik_str":320193,"title":"Apple Inc."}, ... }
    for _, row in data.items():
        if row.get("ticker", "").upper() == t:
            return f"{int(row['cik_str']):010d}"
    raise ValueError(f"CIK not found for ticker={ticker}")

# ---- 2) 会社ファクト（XBRL facts）を取得 ----
@cache_days(30)
def sec_company_facts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r = sec_get(url)
    return r.json()

# ---- 3) 候補キーから fact を抽出（最新の四半期/年次を優先）----
GAAP_REVENUE_KEYS = [
    "us-gaap:SalesRevenueNet",
    "us-gaap:Revenues",
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
]
GAAP_EPS_DILUTED = "us-gaap:EarningsPerShareDiluted"
GAAP_NET_INCOME   = "us-gaap:NetIncomeLoss"
GAAP_WAD_SHARES   = "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding"

def _pick_latest_quarter(values: list) -> dict | None:
    """
    companyfacts の各 units[USD] の values（list of dict）から
    四半期・年次を含む最新（formが10-Q/10-K）を返す
    """
    if not values:
        return None
    # 10-Q/10-Kを優先、filed（提出日）で降順
    def _key(v):
        return (0 if v.get("form") in ("10-Q", "10-K") else 1, v.get("filed",""))
    vals = sorted(values, key=_key)
    # もっとも「10-Q/10-K かつ filed 最新」に近いものからスキャン（末尾の方が新しい）
    for v in reversed(vals):
        if v.get("form") in ("10-Q", "10-K"):
            return v
    # 無ければ最後の要素
    return values[-1]

def _first_val(facts: dict, keys: list[str]) -> tuple[float|None, dict|None]:
    """
    XBRL facts から keys の順で USD 値を探す。 (value, meta) を返す
    """
    if not facts: 
        return None, None
    for key in keys:
        f = facts.get("facts", {}).get(key)
        if not f:
            continue
        units = f.get("units", {})
        usd = None
        # 典型: "USD", "USD/shares"
        for ukey in ("USD", "USD/shares", "USD/share"):
            if ukey in units:
                usd = units[ukey]
                break
        if not usd:
            continue
        v = _pick_latest_quarter(usd)
        if v is None: 
            continue
        val = v.get("val")
        if val is None:
            continue
        try:
            return float(val), v
        except Exception:
            continue
    return None, None

def _try_compute_eps_diluted(facts: dict) -> tuple[float|None, dict|None]:
    """ EPS Diluted が無いとき NetIncome / WeightedAverageDilutedShares で再計算 """
    net, meta_net = _first_val(facts, [GAAP_NET_INCOME])
    wad, meta_sh = _first_val(facts, [GAAP_WAD_SHARES])
    if net is None or not wad:
        return None, None
    if wd := float(wad):
        try:
            return float(net) / float(wd), meta_net
        except Exception:
            return None, None
    return None, None

def get_us_actuals_from_sec(ticker: str) -> dict:
    """
    返り値:
    {
      'eps_diluted': float (GAAP),
      'revenue': float (USD),
      'period': {'fy':..., 'fp':..., 'filed':..., 'form':...},
      'source': 'SEC XBRL'
    }
    """
    cik = resolve_cik(ticker)
    facts = sec_company_facts(cik)
    rev, meta_r = _first_val(facts, GAAP_REVENUE_KEYS)
    eps, meta_e = _first_val(facts, [GAAP_EPS_DILUTED])
    if eps is None:  # フォールバック計算
        eps, meta_e = _try_compute_eps_diluted(facts)

    if rev is None and eps is None:
        raise RuntimeError(f"SEC XBRL facts not found for {ticker}")

    # period 情報（どちらか取れたほう）
    meta = meta_e or meta_r or {}
    period = {
        "fy": meta.get("fy"),
        "fp": meta.get("fp"),    # 'Q1'..'Q4' or 'FY'
        "filed": meta.get("filed"),
        "form": meta.get("form"),
        "end": meta.get("end"),
    }
    return {
        "eps_diluted": eps,
        "revenue": rev,
        "period": period,
        "source": "SEC XBRL",
    }
# ============ /PATCH-A =========================================================

# ============================================================================================================ 
# ----------------------------
# 📌 1. ページタイトル
# ----------------------------
#st.title("📊 株価チャートビューア（TradingView風）")
# 📌 1. ページタイトル
st.markdown('<div class="tenet-h1"> 株価チャートビューア <span>(TradingView風)</span></div>', unsafe_allow_html=True)

# ----------------------------
# 📌 2. ティッカーとセッション管理
# ----------------------------
# =========================
# 2. ティッカーとセッション管理（PATCH2/3 一式）
# =========================

# --- バリデーション：ティッカーを大文字・英数/ドット/ハイフンに正規化 ---
def _normalize_ticker(t: str) -> str:
    if not t:
        return ""
    t = t.strip().upper()
    # ざっくり英数・ドット・ハイフンのみ許容（1〜10文字）
    return t if re.fullmatch(r"[A-Z0-9.\-]{1,10}", t) else ""

# 1) 初期ウォッチリスト（最初だけ作る）
if "watchlists" not in st.session_state:
    st.session_state.watchlists = {
        "My Favorites": ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META"],
        "AI/Chips":     ["NVDA", "AMD", "AVGO", "TSM"],
        "Streaming":    ["NFLX", "RBLX", "SPOT"],
        "Crypto-linked":["COIN", "MSTR", "HOOD"],
    }
#ticker読み込み
# ---- S&P500 / NASDAQ-100 を自動ロードしてウォッチリストに追加 ----
import pandas as pd
import re

@st.cache_data(ttl=2_592_000)  # 約30日キャッシュ
def load_sp500_symbols() -> list[str]:
    """WikipediaからS&P500構成銘柄を取得して正規化"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)

    # 「Symbol」列のあるテーブルを探す
    sym = None
    for t in tables:
        for col in t.columns:
            if str(col).strip().lower() in ("symbol", "ticker", "code"):
                sym = t[col]
                break
        if sym is not None:
            break

    if sym is None:
        raise RuntimeError("S&P500 symbols not found on page.")

    # yfinance 互換に正規化（BRK.B → BRK-B など）
    tickers = (
        sym.astype(str)
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
        .str.replace(".", "-", regex=False)
        .tolist()
    )

    # 無効文字をはじく（あなたの _normalize_ticker と整合）
    pat = re.compile(r"^[A-Z0-9\.\-]{1,10}$")
    tickers = [t for t in tickers if pat.match(t)]
    return sorted(set(tickers))

@st.cache_data(ttl=2_592_000)  # 約30日キャッシュ
def load_nasdaq100_symbols() -> list[str]:
    """WikipediaからNASDAQ100構成銘柄を取得して正規化"""
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    tables = pd.read_html(url)
    sym = None

    for t in tables:
        for col in t.columns:
            if str(col).strip().lower() in ("symbol", "ticker"):
                sym = t[col]
                break
        if sym is not None:
            break

    if sym is None:
        raise RuntimeError("NASDAQ-100 symbols not found on page.")

    tickers = (
        sym.astype(str)
           .str.upper()
           .str.replace(r"\s+", "", regex=True)
           .str.replace(".", "-", regex=False)
           .tolist()
    )
    pat = re.compile(r"^[A-Z0-9\-]{1,10}$")
    return sorted({t for t in tickers if pat.match(t)})

# 実行：取得できたものだけウォッチリストに追加（失敗時は静かにスキップ）
try:
    sp500_list = load_sp500_symbols()
    if sp500_list:
        st.session_state.watchlists["S&P 500"] = sp500_list
except Exception as e:
    st.caption("⚠️ S&P500 リストの自動取得に失敗しました（後で再試行できます）。")

try:
    ndx_list = load_nasdaq100_symbols()
    if ndx_list:
        st.session_state.watchlists["NASDAQ-100"] = ndx_list
except Exception as e:
    st.caption("⚠️ NASDAQ-100 リストの自動取得に失敗しました（後で再試行できます）。")
# ----------------------------------------------------------------------

if "active_watchlist" not in st.session_state:
    st.session_state.active_watchlist = "My Favorites"

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = st.session_state.watchlists[st.session_state.active_watchlist][0]

# -------------------------------
# 2) ウォッチリスト選択/新規作成/削除   ← PATCH2
# -------------------------------
st.markdown("#### 📁 ウォッチリスト")

c1, c2, c3 = st.columns([2, 2, 1])

with c1:
    # 選択で自動保存（keyで状態保持）
    st.session_state.active_watchlist = st.selectbox(
        "選択",
        list(st.session_state.watchlists.keys()),
        index=list(st.session_state.watchlists.keys()).index(st.session_state.active_watchlist),
        key="watchlist_select",
    )

with c2:
    _new = st.text_input("新規リスト名を作成", placeholder="例) Semis, High Growth など", label_visibility="visible")
    if st.button("＋ 作成", use_container_width=True) and _new:
        name = _new.strip()
        if name and name not in st.session_state.watchlists:
            st.session_state.watchlists[name] = []
            st.session_state.active_watchlist = name

with c3:
    # リスト削除（最低1つは残す）
    can_delete = len(st.session_state.watchlists) > 1
    if st.button("🗑 削除", use_container_width=True, disabled=not can_delete):
        name = st.session_state.active_watchlist
        if can_delete:
            del st.session_state.watchlists[name]
            # 適当な残っているリストへ切替
            st.session_state.active_watchlist = list(st.session_state.watchlists.keys())[0]

# 現在のリスト
curr_list_name = st.session_state.active_watchlist
ticker_list = st.session_state.watchlists[curr_list_name]

# -------------------------------
# 3) 銘柄の追加/重複排除
# -------------------------------
st.markdown("#### ⭐ 銘柄（ティッカー）")

cc1, cc2 = st.columns([3, 1])
with cc1:
    new_ticker = st.text_input("ティッカー追加", placeholder="例) AAPL, NVDA など")

with cc2:
    if st.button("＋ 追加", use_container_width=True):
        t = _normalize_ticker(new_ticker)
        if not t:
            st.warning("⚠️ ティッカーは英数字・ドット・ハイフンのみ、1〜10文字で入力してください。")
        elif t in ticker_list:
            st.info(f"ℹ️ {t} はすでにリストにあります。")
        else:
            ticker_list.append(t)
            st.session_state.watchlists[curr_list_name] = ticker_list

# -------------------------------
# 4) ウォッチリストの表示（TradingView風ボタン ＋ ✖削除） ← PATCH3
# -------------------------------
if ticker_list:
    rows = (len(ticker_list) + 5) // 6   # ボタンを6列グリッドに
    for r in range(rows):
        cols = st.columns(6)
        for i in range(6):
            idx = r * 6 + i
            if idx >= len(ticker_list):
                continue
            t = ticker_list[idx]
            with cols[i]:
                # 選択ボタン
                if st.button(t, key=f"pick_{curr_list_name}_{t}", use_container_width=True):
                    st.session_state.selected_ticker = t
                # 削除ボタン
                st.caption("")
                if st.button("✕", key=f"del_{curr_list_name}_{t}", help="リストから削除", use_container_width=True):
                    ticker_list.remove(t)
                    st.session_state.watchlists[curr_list_name] = ticker_list
                    # すべて消えたときの保険
                    if len(ticker_list) == 0 and "selected_ticker" in st.session_state:
                        del st.session_state["selected_ticker"]
else:
    st.info("このリストにはまだ銘柄がありません。上の入力から追加してください。")

# -------------------------------
# 5) 選択中ティッカーを確定（以降のセクションと連動）
# -------------------------------
# リストが空で selected_ticker がない場合の保険
if ticker_list and "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = ticker_list[0]

ticker = st.session_state.selected_ticker

# -------------------------------
# 6) 参考：ミニ価格ボード（任意・軽量）
# -------------------------------
with st.expander("🪙 ミニ価格ボード（参考）", expanded=False):
    show = ticker_list[:12]  # 負荷軽減で12件まで
    data = []
    for t in show:
        try:
            hist = yf.Ticker(t).history(period="2d")  # 直近データ
            if hist.empty:
                hist = yf.Ticker(t).history(period="1d")
            if not hist.empty:
                # 当日 or 前日比
                close = hist["Close"].iloc[-1]
                base  = hist["Close"].iloc[-2] if len(hist) > 1 else hist["Open"].iloc[-1]
                chg   = (close - base) / base * 100 if base else 0.0
                data.append({"Ticker": t, "Price": f"${close:,.2f}", "Change": f"{chg:+.2f}%"})
        except Exception:
            pass
    if data:
        st.dataframe(data, use_container_width=True, hide_index=True)

    else:
        st.caption("※ データ取得不可の銘柄は表示されません。")
# ----------------------------
# 📌 3. 画面を2カラムに分割
# ----------------------------
col1, col2 = st.columns([1, 4])

# ----------------------------
# 📌 4. 左：ティッカー選択ボタン（TradingView風）
# ----------------------------
with col1:
    st.markdown("### ティッカー選択")
    for t in ticker_list:
        if st.button(t, use_container_width=True):
            st.session_state.selected_ticker = t
            ticker = t  # 即時反映

# ----------------------------
# 📌 5. 右：チャートと操作パネル
# ----------------------------
with col2:
    st.markdown(f"## 選択中: `{ticker}`")

    period = st.selectbox(
        "表示期間を選んでください",
        ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"),
        index=2
    )

    # データ取得・チャート描画
    try:
        data = yf.Ticker(ticker).history(period=period)
        if not data.empty:
            # 移動平均線
            data["SMA20"] = data["Close"].rolling(window=20).mean()
            data["SMA50"] = data["Close"].rolling(window=50).mean()
            data["SMA200"] = data["Close"].rolling(window=200).mean()

            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=data.index,
                        open=data["Open"],
                        high=data["High"],
                        low=data["Low"],
                        close=data["Close"],
                        increasing_line_color="green",
                        decreasing_line_color="red",
                        name="ローソク足",
                    )
                ]
            )

            fig.add_trace(
                go.Scatter(x=data.index, y=data["SMA20"], mode="lines", name="SMA 20日",
                           line=dict(color="blue", width=1))
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data["SMA50"], mode="lines", name="SMA 50日",
                           line=dict(color="red", width=1))
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data["SMA200"], mode="lines", name="SMA 200日",
                           line=dict(color="purple", width=1))
            )

            fig.update_layout(
                title=f"{ticker} ローソク足 + SMA",
                xaxis_title="日付",
                yaxis_title="価格 (USD)",
                xaxis_rangeslider_visible=False,
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ データが取得できませんでした。")
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

# ------------------------------------------
# 📊 決算サマリー表示（チャートの下）
# ------------------------------------------
st.markdown("---")
st.markdown("""
<div class="tenet-h1"> 決算概要</div>
""", unsafe_allow_html=True)
if False:
    company = yf.Ticker(ticker).info.get("shortName", ticker)
    pe  = yf.Ticker(ticker).info.get("trailingPE", None)
    fpe = yf.Ticker(ticker).info.get("forwardPE", None)
    peg = yf.Ticker(ticker).info.get("pegRatio", None)
    mcap= yf.Ticker(ticker).info.get("marketCap", None)
    
    def human_mc(n):
        if not n: return "N/A"
        for u in ["","K","M","B","T"]: 
            if abs(n) < 1000: return f"{n:,.2f}{u}"
            n/=1000
        return f"{n:,.2f}Q"
    
    st.markdown(f"""
    <div class="card">
      <div class="header">
        <div class="logo">🟢</div>
        <div class="title-wrap">
          <div class="tenet-h1">{company}</div>
          <div class="tenet-h2">${ticker} Latest Earnings</div>
        </div>
      </div>
      <div class="kv">
        <span class="tenet-chip">Market Cap: <b>{human_mc(mcap)}</b></span>
        <span class="tenet-chip">P/E: <b>{f"{pe:.2f}" if isinstance(pe,(int,float)) else "N/A"}</b></span>
        <span class="tenet-chip">Forward P/E: <b>{f"{fpe:.2f}" if isinstance(fpe,(int,float)) else "N/A"}</b></span>
        <span class="tenet-chip">PEG: <b>{f"{peg:.2f}" if isinstance(peg,(int,float)) else "N/A"}</b></span>
      </div>
    """, unsafe_allow_html=True)


# ========= ⏬ 決算データ（自動換算・堅牢版） =========
DEBUG = True  # デバッグ時 True, 運用時 False

try:
    def safe_pct(numer, denom):
        try:
            if denom not in (None, 0, 0.0):
                return round((float(numer) - float(denom)) / float(denom) * 100, 2)
        except Exception:
            pass
        return 0.0
    
    def to_billions(v):
        try:
            return float(v) / 1e9
        except Exception:
            return 0.0
    
    def get_shares_outstanding(metrics: dict, ticker: str) -> float:
        return (
            metrics.get("sharesOutstanding")
            or metrics.get("shareOutstanding")   # ← Finnhubではこちらが入ることが多い
            or yf.Ticker(ticker).info.get("sharesOutstanding")
            or 0.0
        )
    def _to_float(x):
        """カンマ付きや文字列もできるだけ float 化"""
        try:
            if isinstance(x, str):
                x = x.replace(",", "").strip()
            return float(x)
        except Exception:
            return None
    
    def extract_ic_number(ic):
        """
        financials_reported の report.ic から Revenue を抽出。
        ic が dict でも list でも、全要素を走査して候補キー/ラベルを探す。
        """
        if not ic:
            return None
    
        CANDS = (
            "Revenue",
            "TotalRevenue",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Total revenue",
            "Total Revenues",
            "Revenues",
        )
    
        # dict のときはキー優先
        if isinstance(ic, dict):
            for k in CANDS:
                v = ic.get(k)
                f = _to_float(v)
                if f is not None:
                    return f
    
        # list のときは全要素をチェック（先頭だけでなく）
        if isinstance(ic, list):
            for row in ic:
                if not isinstance(row, dict):
                    continue
                # 1) 候補キーがそのままある
                for k in CANDS:
                    if k in row:
                        f = _to_float(row[k])
                        if f is not None:
                            return f
                # 2) ラベル/コンセプトに含まれる
                label = str(
                    row.get("label")
                    or row.get("concept")
                    or row.get("name")
                    or ""
                ).lower()
                val = row.get("value") or row.get("val") or row.get("amount")
                for k in CANDS:
                    if k.lower() in label:
                        f = _to_float(val)
                        if f is not None:
                            return f
        return None
except Exception as e:
    if DEBUG:
        st.error(traceback.format_exc())  # 開発中は詳細表示
    else:
        st.warning("⚠️ 決算データの取得でエラーが発生しました。")    
# ============ PATCH-B: 実績（Actual）は SEC から ==========================
eps_actual = 0.0
rev_actual_B = 0.0

try:
    actual = get_us_actuals_from_sec(ticker)   # ← SEC
    if actual.get("eps_diluted") is not None:
        eps_actual = float(actual["eps_diluted"])
    if actual.get("revenue") is not None:
        rev_actual_B = float(actual["revenue"]) / 1e9  # 表示はB(十億USD)に
    # 小さくソース情報をUI表示（お好みで）
    st.caption(f"Source: {actual['source']}  {actual['period']}")
except Exception as e:
    st.warning(f"SEC実績の取得に失敗: {e}")
# ============ /PATCH-B =========================================================
# === Estimates layer (EPS/Revenue; keep separate) ===
eps_est_val = None         # ← UI で使う名前に合わせる
rev_est_B   = None         # Revenue 予想（無ければ None のまま）

try:
    earnings_list = finnhub_client.company_earnings(ticker, limit=1)
    if isinstance(earnings_list, list) and earnings_list:
        e0 = earnings_list[0]
        if e0.get("estimate") is not None:
            eps_est_val = float(e0["estimate"])
except Exception as e:
    st.warning(f"EPS予想の取得に失敗: {e}")

# もし Revenue 予想を別APIで入れる場合はここで rev_est_B に代入
# 例: rev_est_B = <analyst revenue estimate in USD billions>

# ============ PATCH-C: 予想は Finnhub で（混ぜない） ====================
eps_est = 0.0
rev_est_B = None  # N/A を許容

try:
    earnings_list = finnhub_client.company_earnings(ticker, limit=1)
    if isinstance(earnings_list, list) and earnings_list:
        e0 = earnings_list[0]
        if e0.get("estimate") is not None:
            eps_est = float(e0["estimate"])
except Exception as e:
    st.warning(f"EPS予想の取得に失敗: {e}")
# ============ /PATCH-C =========================================================
# ============ PATCH-D: 差分の計算とUI =====================================
def safe_pct(numer, denom):
    try:
        if denom not in (None, 0, 0.0):
            return round((float(numer) - float(denom)) / float(denom) * 100, 2)
    except Exception:
        pass
    return 0.0

eps_diff_pct = safe_pct(eps_actual, eps_est) if eps_est else 0.0
rev_diff_pct = 0.0  # 予想が無い場合は 0 / 表示N/A

# 例: st.metric で
st.metric("EPS (Actual)", f"{eps_actual:.2f}", f"{eps_diff_pct:+.2f}%")
if rev_actual_B:
    st.metric("Revenue (B, Actual)", f"{rev_actual_B:.2f}B",
              f"{rev_diff_pct:+.2f}%"
              if rev_est_B is not None else None)
else:
    st.metric("Revenue (B, Actual)", "N/A")
# ============ /PATCH-D =========================================================

# 🎯 ターゲット価格データ（共有で使う）
price_data = pd.DataFrame({
    "Label": ["Before", "After", "Analyst Target", "AI Target"],
    "Price": [181.75, 176.36, 167.24, 178.20]
})

# =============================
# 🎨 ダークカードUI（画像の雰囲気に寄せる）
# 既存の指標変数(eps_actual, eps_est, rev_actual_B, next_rev_B など)をそのまま利用
# =============================

# --- 1) CSS（角丸・影・ダークテーマ） ---
st.markdown("""
<style>
:root{
  --bg:#0e1117; --card:#1a1f2e; --soft:#22293a; --text:#e6e6ea; --muted:#a7b0c0;
  --good:#28d17c; --warn:#f5a524; --bad:#ff4d4f; --chip:#2b3246; --line:#33405c;
}
.block-container{padding-top:1.0rem;}
.card{background:var(--card); border:1px solid #2a3246; border-radius:18px; padding:16px 18px;
      box-shadow:0 8px 30px rgba(0,0,0,.25); color:var(--text); }
.card h3, .card h4, .card h5{margin:0 0 .3rem 0;}
.kv{display:flex; gap:14px; flex-wrap:wrap; margin:.25rem 0 .6rem;}
.kv .chip{background:var(--chip); border:1px solid #38425f; color:var(--muted);
          padding:6px 10px; border-radius:999px; font-size:.82rem;}
.grid{display:grid; grid-template-columns:1fr 1fr; gap:14px;}
.pill{display:flex; align-items:center; gap:12px; background:var(--soft); border:1px solid #2d3650;
     border-radius:16px; padding:14px;}
.pill .dot{width:36px; height:36px; border-radius:50%; display:flex; align-items:center; justify-content:center;
           font-weight:700; color:#0b0e14;}
.pill .lhs{flex:1;}
.pill .lhs .title{color:var(--muted); font-size:.85rem; margin-bottom:2px;}
.pill .lhs .est{color:var(--muted); font-size:.78rem;}
.delta{font-size:.85rem; font-weight:600;}

.tenet-h1 {
  font-family: var(--tenet-head, "Bebas Neue","Oswald","Anton","Arya",sans-serif);
  font-size: 1.8rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  font-weight: 600;
}

.tenet-h2 {
  font-family: var(--tenet-ui, "Oswald","Arya",sans-serif);
  font-size: 1.0rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
}

.good{color:var(--good);} .bad{color:var(--bad);} .muted{color:var(--muted);}
.header{display:flex; gap:14px; align-items:center;}
.logo{width:42px; height:42px; border-radius:10px; background:#0b0e14; display:flex; align-items:center; justify-content:center;}
.title-wrap{display:flex; flex-direction:column}
.title-top{font-size:1.1rem; font-weight:700;}
.subtitle{font-size:.9rem; color:var(--muted);}
.section-title{margin:.6rem 0 .3rem; color:var(--muted); font-weight:600; letter-spacing:.02em;}

#追記
.divider-card {
  height:22px;
  background:var(--card);
  border:1px solid #2a3246;
  border-radius:14px;
  margin-top:8px;
  box-shadow:0 6px 20px rgba(0,0,0,.25);
}
.card + .divider-card { margin-bottom:14px; }
#追記終了


#追記
/* --- ウォッチリスト：横スライドのタブ --- */
.wl-tabs {
  display: flex;
  gap: 10px;
  overflow-x: auto;
  padding: 6px 2px 8px 2px;
  -webkit-overflow-scrolling: touch;
  scrollbar-width: thin;
}
.wl-tab {
  white-space: nowrap;
  background: var(--chip);
  border: 1px solid #38425f;
  color: var(--text);
  padding: 8px 12px;
  border-radius: 999px;
  font-size: .9rem;
  cursor: pointer;
  user-select: none;
}
.wl-tab.active { background: #2e6af2; border-color:#2e6af2; color:#fff; }

/* --- ティッカー縦リスト --- */
.ticker-list { display: flex; flex-direction: column; gap: 8px; }
.ticker-row {
  display: grid; grid-template-columns: auto 1fr auto auto; gap: 10px;
  align-items: center;
  background: var(--card);
  border: 1px solid #2a3246;
  border-radius: 12px;
  padding: 10px 12px;
}
.tkr-ico {
  width: 26px; height: 26px; border-radius: 50%;
  display:flex; align-items:center; justify-content:center;
  background:#0b0e14; color:#9EE06F; font-size:.8rem; font-weight:700;
}
.tkr-name { display:flex; flex-direction:column; }
.tkr-name .sym { font-weight:700; letter-spacing:.02em; }
.tkr-name .cmp { color:var(--muted); font-size:.82rem; }
.tkr-price { font-weight:700; letter-spacing:.01em; }
.tkr-chg { font-weight:700; }
.tkr-up { color:#28d17c; } .tkr-dn { color:#ff4d4f; } .tkr-flat { color:var(--muted); }

.ticker-row:hover { border-color:#3a6df0; box-shadow:0 0 0 2px rgba(58,109,240,.15) inset; }


#追記終了
</style>
""", unsafe_allow_html=True)

# --- 2) ヘッダー（銘柄名・ティッカー・クォーター） ---
company = yf.Ticker(ticker).info.get("shortName", ticker)

# actual を取った直後（PATCH-B内）で period を保存しておく
period_info = actual.get("period", {}) if 'actual' in locals() else {}

# ヘッダー生成箇所で
fy = period_info.get("fy"); fp = period_info.get("fp"); filed = period_info.get("filed")
quarter_label = f"${ticker} {fp or ''} {fy or ''}".strip()
# 例: "$NVDA Q2 2025" のように表示

market_cap = yf.Ticker(ticker).info.get("marketCap", None)
def human(n):  # 時価総額の簡易整形
    try:
        if n is None: return "N/A"
        for unit in ["","K","M","B","T","Q"]:
            if abs(n) < 1000.0: return f"{n:,.2f}{unit}"
            n/=1000.0
    except: return "N/A"

pe = yf.Ticker(ticker).info.get("trailingPE", None)
fpe = yf.Ticker(ticker).info.get("forwardPE", None)
peg = yf.Ticker(ticker).info.get("pegRatio", None)

st.markdown(f"""
<div class="card">
  <div class="header">
    <div class="logo">🟢</div>
    <div class="title-wrap">
      <div class="title-top">{company}</div>
      <div class="subtitle">{quarter_label}</div>
    </div>
  </div>
  <div class="kv">
    <span class="chip">Market Cap: <b>{human(market_cap)}</b></span>
    <span class="chip">P/E: <b>{f"{pe:.2f}" if isinstance(pe,(int,float)) else "N/A"}</b></span>
    <span class="chip">Forward P/E: <b>{f"{fpe:.2f}" if isinstance(fpe,(int,float)) else "N/A"}</b></span>
    <span class="chip">PEG: <b>{f"{peg:.2f}" if isinstance(peg,(int,float)) else "N/A"}</b></span>
  </div>
""", unsafe_allow_html=True)
#追記
st.markdown('<div class="divider-card"></div>', unsafe_allow_html=True)

# --- 3) ピル型メトリクス（左：EPS系 / 右：Revenue系） ---
def pill_html(label, value, est=None, delta=None, good=True):
    color = "var(--good)" if good else "var(--bad)"
    dot_bg = "#28d17c" if good else "#ff4d4f"
    delta_html = f'<span class="delta {"good" if good else "bad"}">{delta}</span>' if delta else '<span class="delta muted">N/A</span>'
    est_html = f'<div class="est">Est. {est}</div>' if est is not None else ""
    return f"""
    <div class="pill">
      <div class="dot" style="background:{dot_bg}">{value}</div>
      <div class="lhs">
        <div class="title">{label}</div>
        {est_html}
      </div>
      {delta_html}
    </div>
    """

# === Estimates (display strings) ===
# EPS 予想の表示用。0.00 はそのまま表示し、None のみ N/A にする
eps_est = f"{eps_est_val:.2f}" if isinstance(eps_est_val, (int, float)) else "N/A"

# UIで参照している将来ガイダンス系が未定義なら安全に初期化
next_eps_est = "TBD"      # 数値が入るようにしたら float に
next_rev_B   = None       # 将来レベニュー予想（B単位）。数値が入るようにしたら float に
next_rev_diff_pct = 0.0   # 将来レベニューの乖離率（あれば算出）

# ---- 表示用の整形（未定義でも落ちないように） ----
rev_est_B_disp      = f"{next_rev_B:.2f}B" if isinstance(next_rev_B, (int, float)) else "N/A"
next_eps_est_disp   = f"{next_eps_est:.2f}" if isinstance(next_eps_est, (int, float)) else "TBD"
next_rev_est_disp   = f"{next_rev_B:.2f}B" if isinstance(next_rev_B, (int, float)) else "N/A"

# すでに前段で算出しているはずの実績系
#   eps_actual: float (SEC実績)
#   eps_diff_pct = safe_pct(eps_actual, eps_est_val)
#   rev_actual_B: float (実績Revenue, B単位)
#   rev_est_B:   float | None（アナリスト予想があれば）
#   rev_diff_pct = safe_pct(rev_actual_B, rev_est_B)
# これらが未定義のときに落ちないように保険
eps_actual   = eps_actual   if 'eps_actual'   in locals() else None
eps_diff_pct = eps_diff_pct if 'eps_diff_pct' in locals() else 0.0
rev_actual_B = rev_actual_B if 'rev_actual_B' in locals() else None
rev_diff_pct = rev_diff_pct if 'rev_diff_pct' in locals() else 0.0
rev_est_B_disp = f"{rev_est_B:.2f}B" if 'rev_est_B' in locals() and isinstance(rev_est_B, (int, float)) else "N/A"

# ---- ピル表示 ----
grid_html = """
  <div class="grid">
    {pill_eps}
    {pill_rev}
    {pill_next_eps}
    {pill_next_rev}
  </div>
""".format(
    pill_eps = pill_html(
        "EPS",
        f"{eps_actual:.2f}" if isinstance(eps_actual,(int,float)) else "TBD",
        est=f"{eps_est}",
        delta=f"{eps_diff_pct:+.2f}%",
        good=(eps_diff_pct >= 0)
    ),
    pill_rev = pill_html(
        "Revenue",
        f"{rev_actual_B:.2f}B" if isinstance(rev_actual_B,(int,float)) else "N/A",
        est=f"{rev_est_B_disp}",
        delta=f"{rev_diff_pct:+.2f}%",
        good=(rev_diff_pct >= 0)
    ),
    pill_next_eps = pill_html(
        "Next Qtr EPS",
        f"{next_eps_est_disp}",
        est="TBD",        # 予想値が2軸あるなら est に「コンセンサス」等を表示
        delta=None,
        good=True
    ),
    pill_next_rev = pill_html(
        "Next Qtr Rev",
        f"{next_rev_est_disp}",
        est="TBD",
        delta=f"{next_rev_diff_pct:+.2f}%",
        good=(next_rev_diff_pct >= 0)
    )
)

st.markdown(grid_html, unsafe_allow_html=True)

# --- 4) 横棒ターゲット（Before/After/Analyst/AI） ---
# 既存price_dataを活用し、見た目を調整
min_x = min(price_data["Price"]) - 15
max_x = max(price_data["Price"]) + 40

fig_ui = px.bar(
    price_data, x="Price", y="Label", orientation="h", text=price_data["Price"].map(lambda v: f"${v:,.2f}"),
    color="Label",
    color_discrete_map={
        "Before":"#7bb1ff", "After":"#2fb27a", "Analyst Target":"#f5a524", "AI Target":"#ff6161"
    }
)
fig_ui.update_traces(textposition="inside", insidetextanchor="middle")
# 注釈ライン：$90 と $200（例）
fig_ui.add_shape(type="line", x0=90, x1=90, y0=-0.5, y1=3.5, line=dict(dash="dot", width=1, color="#d6a000"))
fig_ui.add_annotation(x=90, y=3.35, text="$90", showarrow=False, font=dict(size=11, color="#d6a000"))
fig_ui.add_shape(type="line", x0=200, x1=200, y0=-0.5, y1=3.5, line=dict(dash="dot", width=1, color="#ffae00"))
fig_ui.add_annotation(x=200, y=3.35, text="$200", showarrow=False, font=dict(size=11, color="#ffae00"))

# 右端に%差の注釈（After/AIのみ例示）
before = float(price_data.loc[price_data["Label"]=="Before","Price"])
after  = float(price_data.loc[price_data["Label"]=="After","Price"])
ai     = float(price_data.loc[price_data["Label"]=="AI Target","Price"])
fig_ui.add_annotation(x=after, y=1, text=f"{(after-before)/before*100:+.2f}%", showarrow=False, xshift=28, font=dict(size=11, color="#2fb27a"))
fig_ui.add_annotation(x=ai, y=3, text=f"{(ai-before)/before*100:+.2f}%",   showarrow=False, xshift=28, font=dict(size=11, color="#ff6161"))

fig_ui.update_layout(
    title="Stock & Target Prices",
    xaxis_title="", yaxis_title="",
    xaxis=dict(range=[min_x, max_x], gridcolor="#22304b", zeroline=False),
    yaxis=dict(showgrid=False),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=370,
    margin=dict(l=10, r=10, t=50, b=20),
)
st.plotly_chart(fig_ui, use_container_width=True)

# --- 5) AI Rating 行 ---
st.markdown("""
<div class="section-title">AI Rating:</div>
<div class="card" style="display:flex; align-items:center; gap:10px; justify-content:flex-start;">
  <div>📊</div><div class="muted">Coming soon</div>
</div>
<p class="muted" style="margin-top:.4rem;">
  <em>*Earnings report released on 2025-08-27. Informational purposes only. Consult with a professional and conduct sufficient research before making investment decisions.*</em>
</p>
</div>  <!-- 最初の .card を閉じる -->
""", unsafe_allow_html=True)

# =============================
# 🤖 決算まとめるくん (β) — ルール/テンプレ版
# =============================
#外部LLMなし
def _fmt_b(x, nd=2):
    try:
        return f"{float(x):,.{nd}f}B"
    except:
        return "N/A"

def _fmt_eps(x):
    try:
        return f"${float(x):.2f}"
    except:
        return "TBD"

def _fmt_pct(x):
    try:
        return f"{float(x):+.2f}%"
    except:
        return "N/A"

def _grade_from_surprise(eps_surprise_pct: float, rev_surprise_pct: float) -> tuple[str, str]:
    """
    簡易グレード＆一言コメント
    """
    s = (eps_surprise_pct or 0.0) * 0.6 + (rev_surprise_pct or 0.0) * 0.4
    if s >= 5:
        return "A", "大幅ビートで内容は強い"
    if s >= 1.5:
        return "B", "無難にビート"
    if s > -1.5:
        return "C", "概ね予想線上"
    if s > -5:
        return "D", "やや弱い（ミス）"
    return "E", "想定より弱い（大幅ミス）"

def _mk_bullet(label, value, est=None, extra=None):
    est_txt = f" (予想 {est})" if est is not None else ""
    extra_txt = f" {extra}" if extra else ""
    return f"- **{label}**: **{value}**{est_txt}{extra_txt}"

def _extract_kpi_from_text(raw: str) -> list[str]:
    """
    任意貼り付けテキストから KPI をいくつか正規表現で拾って箇条書き化
    - AI server / 出荷 / 営業CF / FCF / Storage などの簡易検出
    """
    import re
    bullets = []
    if not raw:
        return bullets

    def num_billions(m):
        # 29.7B / $29.7B / 29.7 billion の正規化
        txt = m.group(0)
        txt = txt.replace("billion", "B")
        return txt

    # 例: “AI server” 近傍の金額
    re_ai = re.compile(r"(AI[\s\-]?(server|solution|サーバ|ソリューション).{0,40}?(\$?\d+(\.\d+)?\s?(B|billion)))", re.IGNORECASE)
    for m in re_ai.finditer(raw):
        bullets.append("AIサーバー関連: " + num_billions(m))

    # 営業CF / フリーCF
    re_cfo = re.compile(r"(operating cash flow|営業キャッシュフロー).{0,40}?(\$?\d+(\.\d+)?\s?(B|billion))", re.IGNORECASE)
    for m in re_cfo.finditer(raw):
        bullets.append("営業キャッシュフロー: " + num_billions(m))

    re_fcf = re.compile(r"(free cash flow|フリーキャッシュフロー).{0,40}?(\$?\d+(\.\d+)?\s?(B|billion))", re.IGNORECASE)
    for m in re_fcf.finditer(raw):
        bullets.append("フリーキャッシュフロー: " + num_billions(m))

    # Storage
    re_storage = re.compile(r"(storage|ストレージ).{0,40}?(\$?\d+(\.\d+)?\s?(B|billion))", re.IGNORECASE)
    for m in re_storage.finditer(raw):
        bullets.append("ストレージ関連: " + num_billions(m))

    # YoY/成長率
    re_yoy = re.compile(r"(YoY|前年比|前年同期比).{0,20}?(\+|-)?\d+(\.\d+)?%", re.IGNORECASE)
    for m in re_yoy.finditer(raw):
        bullets.append("成長率: " + m.group(0))

    # 重複削除
    uniq = []
    seen = set()
    for b in bullets:
        if b not in seen:
            uniq.append(b); seen.add(b)
    return uniq[:8]  # 上限

# ==== ここから UI ====
st.markdown("### 🧠 決算まとめるくん (β)")

# 乖離率は既存変数を再利用
eps_surprise_pct = eps_diff_pct                     # EPSサプライズ率
rev_surprise_pct = rev_diff_pct                     # 売上サプライズ率
grade, grade_comment = _grade_from_surprise(eps_surprise_pct, rev_surprise_pct)

# 任意の貼り付け欄（プレスリリース/決算サマリー/ニュースをペーストでOK）
with st.expander("📎 追加情報（任意：プレスリリース/記事を貼り付け）", expanded=False):
    pasted = st.text_area("貼り付けると AI 風のKPI拾いを試みます（空欄OK）", height=140)
    kpi_bullets = _extract_kpi_from_text(pasted)

# 見出し
company_safe = company if isinstance(company, str) else ticker
st.markdown(f"**{company_safe}  ${ticker}  決算サマリー**")

# ─ 今四半期の実績
left, right = st.columns(2)

with left:
    bullets = []
    bullets.append(_mk_bullet("EPS", _fmt_eps(eps_actual), est=_fmt_eps(eps_est_val)))
    bullets.append(_mk_bullet("売上高", _fmt_b(rev_actual_B), est=_fmt_b(rev_est_B)))
    st.markdown("#### 今四半期業績")
    st.markdown("\n".join(bullets))

with right:
    bullets = []
    if next_eps_est not in ("TBD", None, ""):
        bullets.append(_mk_bullet("次四半期 EPS ガイダンス", _fmt_eps(next_eps_est)))
    if next_rev_B:
        bullets.append(_mk_bullet("次四半期 売上高ガイダンス", _fmt_b(next_rev_B)))
    st.markdown("#### 次四半期ガイダンス")
    if bullets:
        st.markdown("\n".join(bullets))
    else:
        st.caption("ガイダンス情報は取得できませんでした。")

# ─ 重要指標（テキスト抽出）
st.markdown("#### 重要指標（抽出）")
if kpi_bullets:
    st.markdown("\n".join([f"- {b}" for b in kpi_bullets]))
else:
    st.caption("貼り付けテキストから抽出できるKPIは見つかりませんでした（任意テキスト貼り付け欄をご利用ください）。")

# ─ 簡易評価＆AIコメント（テンプレ）
st.markdown("#### 決算内容の注目ポイント（自動生成）")

eps_line = f"EPSは{_fmt_eps(eps_actual)}（予想{_fmt_eps(eps_est_val)}）"
rev_line = f"売上高は{_fmt_b(rev_actual_B)}（予想{_fmt_b(rev_est_B)}）"
surp = f"サプライズ率：EPS {_fmt_pct(eps_surprise_pct)} / Revenue {_fmt_pct(rev_surprise_pct)}"
guide_line = ""
if isinstance(next_eps_est, (int, float)) or (isinstance(next_eps_est, str) and next_eps_est not in ("TBD", "")):
    guide_line += f"次四半期EPSガイダンスは{_fmt_eps(next_eps_est)}。"
if next_rev_B:
    guide_line += f"売上高ガイダンスは{_fmt_b(next_rev_B)}。"

overall = f"総合評価は **{grade}**（{grade_comment}）。"

st.markdown(
    f"- {eps_line}\n"
    f"- {rev_line}\n"
    f"- {surp}\n"
    + (f"- {guide_line}\n" if guide_line else "")
    + f"- {overall}"
)

st.caption(
    "*自動生成のサマリー（β）。参考情報であり、投資判断は自己責任でお願いします。*"
)


# ------------------------------------------
# 🤖 AI Rating（仮置き）
# ------------------------------------------
st.markdown("### 🤖 AI Rating: 📈")
st.caption("*Earnings report released on 2025-08-27. Informational purposes only. Please consult with a professional before investing.*")

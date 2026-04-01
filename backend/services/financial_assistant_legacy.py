import os
import re
import warnings
import pandas as pd
import gradio as gr
from groq import Groq
from rapidfuzz import process as fuzz

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# from langchain_community.memory import ConversationBufferMemory

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==========================================================
# 🔑 CONFIG
# ==========================================================
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_ID = "llama-3.1-8b-instant"
client = Groq(api_key=GROQ_API_KEY)

# # ==========================================================
# # 🧠 CONVERSATION MEMORY
# # ==========================================================
# memory = ConversationBufferMemory(
#     memory_key="chat_history",
#     return_messages=True
# )

# # ==========================================================
# # 🧠 MEMORY → GROQ MESSAGE FORMAT
# # ==========================================================
# def memory_to_messages():
#     """
#     Convert LangChain memory to Groq-compatible messages
#     """
#     msgs = []
#     for m in memory.chat_memory.messages:
#         role = "assistant" if m.type == "ai" else "user"
#         msgs.append({"role": role, "content": m.content})
#     return msgs


# NIFTY SYMBOL LIST
NIFTY_SYMBOLS = [
    "ADANIPORTS","ASIANPAINT","AXISBANK","BAJAJ-AUTO","BAJAJFINSV","BAJFINANCE",
    "BHARTIARTL","BPCL","BRITANNIA","CIPLA","COALINDIA","DRREDDY","EICHERMOT",
    "GAIL","GRASIM","HCLTECH","HDFC","HDFCBANK","HEROMOTOCO","HINDALCO",
    "HINDUNILVR","ICICIBANK","INDUSINDBK","INFRATEL","INFY","IOC","ITC",
    "JSWSTEEL","KOTAKBANK","LT","MARUTI","M&M","NESTLEIND","NTPC","ONGC",
    "POWERGRID","RELIANCE","SBIN","SHREECEM","SUNPHARMA","TATAMOTORS",
    "TATASTEEL","TCS","TECHM","TITAN","ULTRACEMCO","UPL","VEDL","WIPRO","ZEEL"
]

# Fuzzy recognition
COMPANY_NAME_MAP = {
    "RELIANCE": ["reliance", "ril", "reliance industries"],
    "TCS": ["tcs", "tata consultancy services", "tata consultancy"],
    "INFY": ["infosys", "infy"],
    "SBIN": ["sbi", "state bank of india", "state bank"],
    "ITC": ["itc", "itc ltd"],
    "HDFCBANK": ["hdfc bank"],
    "HDFC": ["hdfc"],
    "TATASTEEL": ["tata steel", "tatasteel"],
}

MARKET_DATA = {}
STOCK_META = {}

# ==========================================================
# PATH HELPER
# ==========================================================
def project_path(*parts):
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", *parts))

# ==========================================================
# LOAD STRUCTURED NIFTY DATA
# ==========================================================
def load_market_data():
    global MARKET_DATA, STOCK_META
    nifty_dir = project_path("data", "NIFTY50")
    print("📂 Loading NIFTY50 CSVs from:", nifty_dir)

    if not os.path.isdir(nifty_dir):
        print("⚠️ NIFTY directory missing")
        return

    # --- Load OHLCV data ---
    for fname in os.listdir(nifty_dir):
        if not fname.endswith(".csv") or fname.lower() == "stock_metadata.csv":
            continue

        fp = os.path.join(nifty_dir, fname)
        try:
            df = pd.read_csv(fp)
        except:
            continue

        df.columns = [c.strip() for c in df.columns]

        if "Date" not in df or "Symbol" not in df:
            continue

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

        for _, row in df.iterrows():
            sym = str(row["Symbol"]).upper()
            dt = str(row["Date"])
            if dt == "NaT":
                continue

            MARKET_DATA[(sym, dt)] = {
                "Symbol": sym,
                "Date": dt,
                "Open": row.get("Open"),
                "High": row.get("High"),
                "Low": row.get("Low"),
                "Close": row.get("Close"),
                "VWAP": row.get("VWAP"),
                "Volume": row.get("Volume", row.get("Shares Traded")),
                "Turnover": row.get("Turnover", row.get("Turnover (Cr)")),
                "Trades": row.get("Trades"),
                "Deliverable Volume": row.get("Deliverable Volume"),
                "%Deliverable": row.get("%Deliverable", row.get("%Deliverble")),
            }

    print("✅ Loaded structured rows:", len(MARKET_DATA))

    # --- Load metadata ---
    meta_path = os.path.join(nifty_dir, "stock_metadata.csv")
    if os.path.exists(meta_path):
        df = pd.read_csv(meta_path)
        df.columns = [c.strip() for c in df.columns]

        for _, r in df.iterrows():
            sym = str(r["Symbol"]).upper()
            STOCK_META[sym] = {
                "Company Name": r.get("Company Name", ""),
                "Industry": r.get("Industry", ""),
                "ISIN Code": r.get("ISIN Code", ""),
                "Series": r.get("Series", "")
            }

        print("📘 Loaded stock metadata:", len(STOCK_META))

# ==========================================================
# FUZZY SYMBOL DETECTION
# ==========================================================
def fuzzy_symbol(text: str):
    up = text.upper()
    for s in NIFTY_SYMBOLS:
        if s in up:
            return s

    for sym, aliases in COMPANY_NAME_MAP.items():
        match, score, _ = fuzz.extractOne(text.lower(), aliases)
        if score > 80:
            return sym

    match, score, _ = fuzz.extractOne(up, NIFTY_SYMBOLS)
    return match if score > 85 else None


def extract_symbols(text: str):
    """
    Extract one or more symbols from free-form user text.
    Supports symbol literals (e.g., HDFCBANK), hyphen/space variants
    (e.g., BAJAJ AUTO for BAJAJ-AUTO), and alias phrases in COMPANY_NAME_MAP.
    """
    low = text.lower()
    hits = []

    # Symbol literals and symbol variants with separators.
    for sym in NIFTY_SYMBOLS:
        sym_low = sym.lower()
        variant = re.escape(sym_low).replace(r"\-", r"[-\s]?")
        m = re.search(rf"\b{variant}\b", low)
        if m:
            hits.append((m.start(), sym))

    # Alias phrases (e.g., "tata steel", "state bank of india")
    for sym, aliases in COMPANY_NAME_MAP.items():
        for alias in aliases:
            m = re.search(rf"\b{re.escape(alias.lower())}\b", low)
            if m:
                hits.append((m.start(), sym))

    if not hits:
        guess = fuzzy_symbol(text)
        return [guess] if guess else []

    hits.sort(key=lambda x: x[0])
    ordered = []
    seen = set()
    for _, sym in hits:
        if sym not in seen:
            ordered.append(sym)
            seen.add(sym)
    return ordered

# ==========================================================
# FUZZY DATE DETECTION
# ==========================================================
MONTH_MAP = {
    "jan":"01","feb":"02","mar":"03","apr":"04","may":"05","jun":"06",
    "jul":"07","aug":"08","sep":"09","oct":"10","nov":"11","dec":"12"
}

def match_all_years(symbol, month, day):
    out=[]
    for (sym, dt), _ in MARKET_DATA.items():
        if sym == symbol and dt[5:7] == month and dt[8:10] == day:
            out.append(dt)
    return sorted(out) if out else None

def fuzzy_date(text, symbol):
    # YYYY-MM-DD
    m = re.search(r"(\d{4})[-/](\d{2})[-/](\d{2})", text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # 5 May, 5th May
    m = re.search(r"(\d{1,2})(st|nd|rd|th)?\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", text, re.I)
    if m:
        day = m.group(1).zfill(2)
        month = MONTH_MAP[m.group(3).lower()]
        return match_all_years(symbol, month, day)

    # May 5
    m = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2})", text, re.I)
    if m:
        month = MONTH_MAP[m.group(1).lower()]
        day = m.group(2).zfill(2)
        return match_all_years(symbol, month, day)

    return None

# ==========================================================
# YEAR RANGE DETECTION FOR TREND ANALYSIS
# ==========================================================
def parse_year_range(text):
    m = re.search(r"\b((?:19|20)\d{2})\s*[-–]\s*((?:19|20)\d{2})\b", text)
    if m:
        y1,y2 = int(m.group(1)), int(m.group(2))
        return (min(y1,y2), max(y1,y2))

    m = re.search(r"between\s+((?:19|20)\d{2})\s+and\s+((?:19|20)\d{2})", text, re.I)
    if m:
        return (min(int(m.group(1)),int(m.group(2))),
                max(int(m.group(1)),int(m.group(2))))

    m = re.search(r"from\s+((?:19|20)\d{2})\s+to\s+((?:19|20)\d{2})", text, re.I)
    if m:
        return (min(int(m.group(1)),int(m.group(2))),
                max(int(m.group(1)),int(m.group(2))))
    return None

def last_day_of_month(year: int, month: int) -> int:
    # April, June, September, November
    if month in {4, 6, 9, 11}:
        return 30

    # February
    if month == 2:
        # Leap year check
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return 29
        return 28

    # All other months
    return 31


def parse_month_year_range(text):
    """
    Handles queries like:
    - January 2016 to June 2016
    - Jan 2016 – Jun 2016
    - from January 2016 to June 2016
    """

    m = re.search(
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(19|20\d{2})"
        r".*?"
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(19|20\d{2})",
        text,
        re.I
    )

    if not m:
        return None

    start_month = MONTH_MAP[m.group(1).lower()]
    start_year = int(m.group(2))
    end_month = MONTH_MAP[m.group(3).lower()]
    end_year = int(m.group(4))

    start_date = f"{start_year}-{start_month}-01"

    end_month_int = int(end_month)
    end_day = last_day_of_month(end_year, end_month_int)
    end_date = f"{end_year}-{end_month}-{end_day}"

    return start_date, end_date


def multi_year_intent(text):
    keys=["across","every year","historically","year-wise","year wise"]
    t=text.lower()
    return any(k in t for k in keys)

def trend_intent(text):
    keys=["trend","performance","between","over the period","from","to"]
    t=text.lower()
    return any(k in t for k in keys)

# ==========================================================
# STRUCTURED ANSWER FOR SINGLE DAY
# ==========================================================
def structured_answer(symbol,date):
    row = MARKET_DATA.get((symbol,date))
    if not row:
        return None
    
    meta = STOCK_META.get(symbol, {})
    return (
        f"Here is the trading summary for {symbol} on {date}:\n"
        f"- Open: {row['Open']}\n"
        f"- High: {row['High']}\n"
        f"- Low: {row['Low']}\n"
        f"- Close: {row['Close']}\n"
        f"- VWAP: {row['VWAP']}\n"
        f"- Volume: {row['Volume']}\n"
        f"- Turnover: {row['Turnover']}\n"
        f"- Trades: {row['Trades']}\n\n"
        "Stock metadata:\n"
        f"- Company: {meta.get('Company Name','')}\n"
        f"- Industry: {meta.get('Industry','')}\n"
        f"- ISIN: {meta.get('ISIN Code','')}\n"
    )

# ==========================================================
# MULTI-YEAR SAME DATE COMPARISON
# ==========================================================
def multi_year_comparison(symbol, dates):
    out=[]
    rows=[]
    for d in dates:
        row=MARKET_DATA.get((symbol,d))
        if row:
            rows.append(row)
            out.append(f"- {d}: Close {row['Close']}, Vol {row['Volume']}")

    if not out:
        return f"No records found for {symbol} on these matching dates."

    text="📊 Multi-year comparison:\n"+"\n".join(out)
    return text

# ==========================================================
# TREND ANALYSIS VIA LLM
# ==========================================================
def trend_analysis(symbol, start_year, end_year):
    rows = []
    for (sym, dt), r in MARKET_DATA.items():
        if sym == symbol:
            year = int(dt[:4])
            if start_year <= year <= end_year:
                rows.append(r)

    if not rows:
        return (
            f"I searched for {symbol} between {start_year} and {end_year}, "
            "but found no trading records in that range."
        )

    rows = sorted(rows, key=lambda r: r["Date"])

    # ---- SAFE FLOAT EXTRACTOR ----
    def safe_float(x):
        try:
            if x is None or pd.isna(x):
                return None
            return float(x)
        except:
            return None

    # ---- FIRST & LAST CLOSE ----
    first_close = None
    first_date = None
    for r in rows:
        c = safe_float(r["Close"])
        if c is not None:
            first_close = c
            first_date = r["Date"]
            break

    last_close = None
    last_date = None
    for r in reversed(rows):
        c = safe_float(r["Close"])
        if c is not None:
            last_close = c
            last_date = r["Date"]
            break

    # ---- HIGH / LOW / VOLUME ----
    highs = [safe_float(r["High"]) for r in rows]
    lows = [safe_float(r["Low"]) for r in rows]
    vols = [safe_float(r["Volume"]) for r in rows]

    highs = [h for h in highs if h is not None]
    lows = [l for l in lows if l is not None]
    vols = [v for v in vols if v is not None]

    overall_high = max(highs) if highs else None
    overall_low = min(lows) if lows else None
    avg_vol = sum(vols) / len(vols) if vols else None

    stats_str = [
        f"Trend summary for {symbol} between {start_year} and {end_year}:",
        f"- Records found: {len(rows)}",
    ]

    if first_close is not None and last_close is not None:
        try:
            pct = (last_close - first_close) / first_close * 100
            stats_str.append(
                f"- First close ({first_date}): {first_close}, "
                f"Last close ({last_date}): {last_close}, "
                f"Change: {pct:.2f}%"
            )
        except:
            pass

    if overall_high is not None:
        stats_str.append(f"- Highest price: {overall_high}")

    if overall_low is not None:
        stats_str.append(f"- Lowest price: {overall_low}")

    if avg_vol is not None:
        stats_str.append(f"- Average volume: {avg_vol:.0f}")

    stats_block = "\n".join(stats_str)

    # ---- LLM INTERPRETATION ----
    prompt = f"""
You are a financial analyst. Based on the following summary statistics,
explain the multi-year price trend of {symbol} between {start_year} and {end_year}.
Discuss volatility, demand–supply cycles, investor sentiment, and macro influences.

Keep the explanation clear, very concise, not too long and finance-focused.
Avoid follow-up questions or suggestions.
{stats_block}
"""


    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.4,
    )


    explanation = resp.choices[0].message.content.strip()


    return stats_block + "\n\n📘 Trend Interpretation:\n" + explanation


# ==========================================================
# HYBRID PRICE EXPLANATION (STRUCTURED + THEORY)
# ==========================================================
def hybrid_explanation(symbol, date, mode="detailed"):
    """
    mode = "simple"   → short OHLC explanation
    mode = "detailed" → full market explanation
    """

    raw = structured_answer(symbol, date)
    if not raw:
        return None

    row = MARKET_DATA[(symbol, date)]

    # --------------------------------------------------
    # SIMPLE EXPLANATION (for "Explain OHLC data")
    # --------------------------------------------------
    if mode == "simple":
        explanation = (
            f"On {date}, {symbol} opened at {row['Open']} and traded between "
            f"a high of {row['High']} and a low of {row['Low']}. "
            f"It closed at {row['Close']}, which shows the final market consensus "
            f"for the day.\n\n"
            f"The VWAP of {row['VWAP']} indicates the average price weighted by volume, "
            f"helping assess whether the stock traded at a premium or discount. "
            f"Volume of {row['Volume']} reflects the overall trading activity."
        )

        return raw + "\n📘 Simple OHLC Explanation:\n" + explanation

    # --------------------------------------------------
    # DETAILED EXPLANATION (price action / sentiment)
    # --------------------------------------------------
    
    prompt = f"""
Explain the price action of {symbol} on {date} using the OHLC and VWAP data below.

Focus on:
- intraday momentum
- volatility
- VWAP interpretation
- overall market sentiment

Rules:
- Keep explanation under 5 bullet points
- Be concise and factual
- No follow-up questions
- No macroeconomic theory

Data:
Open={row['Open']}
High={row['High']}
Low={row['Low']}
Close={row['Close']}
VWAP={row['VWAP']}
Volume={row['Volume']}

Keep the explanation clear, very concise, not too long and finance-focused.
Avoid follow-up questions or suggestions. 

"""
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.4
    )

    return raw + "\n📘 Market Explanation:\n" + resp.choices[0].message.content.strip()


# ==========================================================
# BUILD RAG INDEX
# ==========================================================
def load_or_build_rag():
    index_path=project_path("data","finance_index")

    if os.path.exists(index_path):
        print("📦 Loading existing FAISS index…")
        emb=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        try:
            return FAISS.load_local(index_path,emb,allow_dangerous_deserialization=True)
        except:
            print("⚠️ Load failed, rebuilding...")

    print("📥 Building new FAISS index...")
    corpus_path=project_path("data","financial_corpus")
    loader=DirectoryLoader(corpus_path, glob="**/*.txt", loader_cls=TextLoader)

    docs=[]
    for fp in loader.file_paths:
        try:
            txt=open(fp,"r",encoding="utf-8").read()
            docs.append(txt)
        except:
            print("⚠️ Skipping file:", fp)

    splitter=RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks=splitter.split_text("\n".join(docs))

    emb=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vect=FAISS.from_texts(chunks,emb)
    vect.save_local(index_path)

    print("✅ Saved FAISS index.")
    return vect

vectorstore=load_or_build_rag()
retriever=vectorstore.as_retriever(search_kwargs={"k":3})
load_market_data()

# ==========================================================
# RAG ANSWER
# ==========================================================
def rag_answer(q):
    docs = retriever.invoke(q)
    ctx = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are a financial assistant.

Use ONLY the context below.
If the exact answer is not present:
- give the closest relevant financial insight
- explain what information is missing
- suggest how the user can refine the question (date, company, period)

Context:
{ctx}

Question:
{q}
"""

    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.3,
    )

    return resp.choices[0].message.content.strip()



# ==========================================================
# MULTI SYMBOL TREND
# ==========================================================
def multi_symbol_trend(symbols, start_year, end_year):
    """
    Produce multi-company trend analysis for the same year range.
    Runs trend_analysis() for each symbol and concatenates the results.
    """
    outputs = []
    for sym in symbols:
        res = trend_analysis(sym, start_year, end_year)
        outputs.append(f"===== {sym} Trend ({start_year}–{end_year}) =====\n{res}\n")
    return "\n\n".join(outputs)


# ==========================================================
# RETURN COMPARISON
# ==========================================================
def compute_return_stats(symbol, start_year=None, end_year=None):
    """
    Compute close-to-close return stats for one symbol.
    If no year range is supplied, uses full available history.
    """
    rows = []
    for (sym, dt), row in MARKET_DATA.items():
        if sym != symbol:
            continue
        yr = int(dt[:4])
        if start_year is not None and yr < start_year:
            continue
        if end_year is not None and yr > end_year:
            continue
        rows.append(row)

    if not rows:
        return None

    rows = sorted(rows, key=lambda r: r["Date"])
    rows = [r for r in rows if r.get("Close") is not None and not pd.isna(r.get("Close"))]
    if not rows:
        return None

    first = rows[0]
    last = rows[-1]
    first_close = float(first["Close"])
    last_close = float(last["Close"])
    if first_close == 0:
        return None

    return_pct = (last_close - first_close) / first_close * 100.0
    return {
        "symbol": symbol,
        "start_date": first["Date"],
        "end_date": last["Date"],
        "start_close": first_close,
        "end_close": last_close,
        "return_pct": return_pct,
        "n_days": len(rows),
    }


def compare_returns(symbols, start_year=None, end_year=None):
    stats = []
    for sym in symbols:
        s = compute_return_stats(sym, start_year, end_year)
        if s:
            stats.append(s)

    if not stats:
        if start_year is not None and end_year is not None:
            return f"No return data found for the requested symbols in {start_year}-{end_year}."
        return "No return data found for the requested symbols."

    lines = []
    if start_year is not None and end_year is not None:
        lines.append(f"Return comparison ({start_year}-{end_year}):")
    else:
        lines.append("Return comparison (full available history per symbol):")

    for s in stats:
        lines.append(
            f"- {s['symbol']}: {s['return_pct']:.2f}% "
            f"({s['start_date']} close {s['start_close']:.2f} -> "
            f"{s['end_date']} close {s['end_close']:.2f}, {s['n_days']} trading days)"
        )

    if len(stats) >= 2:
        best = max(stats, key=lambda x: x["return_pct"])
        worst = min(stats, key=lambda x: x["return_pct"])
        lines.append(
            f"\nBest performer: {best['symbol']} ({best['return_pct']:.2f}%). "
            f"Weakest: {worst['symbol']} ({worst['return_pct']:.2f}%)."
        )

    return "\n".join(lines)


# ==========================================================
# MAIN ROUTER
# ==========================================================
def handle_query(q, context):
    q = q.strip()
    if not q:
        return "Please enter a question.", context

    lower_q = q.lower()


    # -------------------------------------------------
    # INTENT DETECTION
    # -------------------------------------------------
    is_trend = re.search(r"\btrend\b|\bperformance\b|\bgrowth\b", lower_q)
    is_explain = re.search(r"\bexplain\b|\bwhy\b|\binterpret\b", lower_q)
    is_compare = re.search(r"\bcompare\b|\bvs\b|\bversus\b|\bboth\b", lower_q)

    # Prefer explicit symbols in current query over chat context.
    symbols = extract_symbols(q)

    # -------------------------------------------------
    # 1B) DIRECT RETURN COMPARISON ACROSS MULTIPLE SYMBOLS
    # -------------------------------------------------
    if is_compare and len(symbols) >= 2:
        yr = parse_year_range(q)
        if yr:
            start_year, end_year = yr
            return compare_returns(symbols, start_year, end_year), context
        return compare_returns(symbols), context

    # -------------------------------------------------
    # CASE A: Follow-up intents using context (no symbol/date in query)
    # -------------------------------------------------
    if not symbols and context.get("last_symbol") and context.get("last_dates"):
        symbol = context["last_symbol"]
        dates = context["last_dates"]  # list of YYYY-MM-DD

        # Trend analysis follow-up
        if "trend" in lower_q:
            years = sorted({int(d[:4]) for d in dates})
            start_year, end_year = years[0], years[-1]
            return trend_analysis(symbol, start_year, end_year), context

        # Average price follow-up
        if "average" in lower_q:
            closes = []
            for d in dates:
                row = MARKET_DATA.get((symbol, d))
                if row and row.get("Close") is not None and not pd.isna(row.get("Close")):
                    closes.append(float(row["Close"]))
            if closes:
                avg_close = sum(closes) / len(closes)
                return f"Average Close for {symbol} over {len(closes)} trading days: {avg_close:.2f}", context
            return f"Could not compute average close for {symbol}.", context

        # Highest / Lowest follow-up
        if "highest" in lower_q or "max" in lower_q:
            best = None
            for d in dates:
                row = MARKET_DATA.get((symbol, d))
                if row and row.get("High") is not None and not pd.isna(row.get("High")):
                    val = float(row["High"])
                    if best is None or val > best[0]:
                        best = (val, d)
            if best:
                return f"Highest High for {symbol}: {best[0]} on {best[1]}", context

        if "lowest" in lower_q or "min" in lower_q:
            best = None
            for d in dates:
                row = MARKET_DATA.get((symbol, d))
                if row and row.get("Low") is not None and not pd.isna(row.get("Low")):
                    val = float(row["Low"])
                    if best is None or val < best[0]:
                        best = (val, d)
            if best:
                return f"Lowest Low for {symbol}: {best[0]} on {best[1]}", context

        # Monthly summary follow-up
        if "monthly" in lower_q or "month" in lower_q:
            months = sorted({d[:7] for d in dates})
            return (
                f"Monthly coverage for {symbol} in this range:\n" +
                "\n".join("• " + m for m in months),
                context
            )

    # -------------------------------------------------
    # CASE 0: Follow-up YEAR ONLY (e.g. "2016")
    # -------------------------------------------------
    year_only = re.fullmatch(r"(19|20)\d{2}", q)
    if year_only and (not symbols) and context.get("last_symbol") and context.get("last_dates"):
        symbol = context["last_symbol"]
        dates = context["last_dates"]
        year = year_only.group(0)

        chosen = [d for d in dates if d.startswith(year)]
        if chosen:
            # keep chosen dates as last_dates (useful for compare)
            return structured_answer(symbol, chosen[0]), {
                "last_symbol": symbol,
                "last_dates": dates
            }

        return (
            f"I searched for {symbol} on {year}-{dates[0][5:]}, but found no trading record.\n\n"
            "Available dates:\n" + "\n".join("• " + d for d in dates),
            context
        )

    # -------------------------------------------------
    # CASE: "Compare both ..." follow-up
    # -------------------------------------------------
    if (not symbols) and re.search(r"\bcompare\b|\bboth\b", lower_q):
        if context.get("last_symbol") and context.get("last_dates"):
            symbol = context["last_symbol"]
            dates = context["last_dates"]

            years = re.findall(r"\b(?:19|20)\d{2}\b", q)
            chosen = [d for d in dates if any(d.startswith(y) for y in years)]

            if len(chosen) >= 2:
                return multi_year_comparison(symbol, chosen), {
                    "last_symbol": symbol,
                    "last_dates": dates
                }

            return (
                "I can compare prices for these available dates:\n"
                + "\n".join("• " + d for d in dates)
                + "\n\nPlease mention the years explicitly (e.g., 2016 and 2020).",
                context
            )

    # -------------------------------------------------
    # 2) TREND ANALYSIS (explicit year range in the query)
    # -------------------------------------------------
    if symbols and is_trend:
        yr = parse_year_range(q)
        if yr:
            start_year, end_year = yr
            if len(symbols) >= 2:
                return multi_symbol_trend(symbols, start_year, end_year), context
            return trend_analysis(symbols[0], start_year, end_year), context


    # -------------------------------------------------
    # 3) DATE-BASED LOGIC
    # -------------------------------------------------
    if symbols:
        symbol = symbols[0]

        # (a) Month-range query like "Jan 2016 to Jun 2016"
        month_range = parse_month_year_range(q)
        if month_range:
            start_date, end_date = month_range

            rows = [
                r for (sym, dt), r in MARKET_DATA.items()
                if sym == symbol and start_date <= dt <= end_date
            ]

            if not rows:
                return (
                    f"I searched for {symbol} between {start_date} and {end_date}, "
                    "but found no trading records in that period.",
                    context
                )

            # IMPORTANT: store all dates so follow-ups like "trend analysis" work
            date_list = sorted({r["Date"] for r in rows if r.get("Date")})
            new_context = {"last_symbol": symbol, "last_dates": date_list}

            return (
                f"Found {len(rows)} trading days for {symbol} from {start_date} to {end_date}.\n\n"
                "You can ask:\n"
                "- Average price\n"
                "- Trend analysis\n"
                "- Highest / lowest price\n"
                "- Monthly summary",
                new_context
            )

        # (b) Single-day or same-day-multi-year query
        dt = fuzzy_date(q, symbol)

        if isinstance(dt, str):
            # SIMPLE explanation intent
            if re.search(r"\bexplain\b.*\bohlc\b|\bohlc\b.*\bexplain\b", lower_q):
                return hybrid_explanation(symbol, dt, mode="simple"), context

            # DETAILED explanation intent
            if "explain" in lower_q:
                return hybrid_explanation(symbol, dt, mode="detailed"), context

            # Default structured answer
            ans = structured_answer(symbol, dt)
            if ans:
                return ans, {"last_symbol": symbol, "last_dates": None}


        if isinstance(dt, list):
            return (
                f"Multiple years match this calendar date for {symbol}:\n"
                + "\n".join("• " + d for d in dt)
                + "\n\nPlease specify the exact year (YYYY-MM-DD), or ask for a multi-year comparison.",
                {"last_symbol": symbol, "last_dates": dt}
            )

    # -------------------------------------------------
    # 4) FALLBACK → RAG
    # -------------------------------------------------
    return rag_answer(q), context




# ==========================================================
# UI (Chatbot)
# ==========================================================
import tempfile

CUSTOM_CSS = """
/* Center heading */
#title-block h1 {
    text-align: center;
    font-size: 2.2rem;
    font-weight: 700;
}

#title-block p {
    text-align: center;
    font-size: 1.05rem;
    color: #b5b5b5;
    margin-top: -10px;
}

/* Chatbot container */
.gr-chatbot {
    border-radius: 12px;
}

/* Textbox */
textarea {
    font-size: 1rem !important;
}

/* Buttons */
#send-btn button {
    background: linear-gradient(135deg, #b11226, #4b0000);
    color: #ffffff;
    font-weight: 700;
    border: none;
}

#clear-btn button {
    background: linear-gradient(135deg, #8b0000, #1a1a1a);
    color: #ffffff;
    font-weight: 600;
    border: none;
}

#download-btn button {
    background: linear-gradient(135deg, #2b2b2b, #000000);
    color: #ffffff;
    font-weight: 600;
    border: 1px solid #8b0000;
}


/* Button hover */
button:hover {
    filter: brightness(1.05);
    transition: all 0.2s ease-in-out;
}


/* File box */
.gr-file {
    border-radius: 10px;
}
"""


with gr.Blocks(
    title="Financial LLM – NIFTY50 QA + RAG",
    css=CUSTOM_CSS,
) as demo:

    with gr.Column(elem_id="title-block"):
        gr.Markdown("""
        # ManaFinance AI  
        Ask stock/date questions, trend analysis, or finance concepts using structured data + RAG
        """)


    chatbot = gr.Chatbot(height=500, label="Financial Assistant", type="messages")
    user_msg = gr.Textbox(label="Your Question", placeholder="Ask something...")
    send_btn = gr.Button("Send", elem_id="send-btn")
    clear_btn = gr.Button("Clear Chat", elem_id="clear-btn")
    download_btn = gr.Button("Download Chat History", elem_id="download-btn")


    chat_file = gr.File(label="Chat History File")

    history_state = gr.State([])  # list of {"role":..., "content":...}

    # CHAT HANDLER ----------------------------------------
    def chat_handler(message, history, context):
        bot_reply, context = handle_query(message, context)

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_reply})

        return history, history, context
    
    context_state = gr.State({
        "last_symbol": None,
        "last_dates": None,     # list like ['2016-05-05', '2020-05-05']
    })



    send_btn.click(
        fn=chat_handler,
        inputs=[user_msg, history_state, context_state],
        outputs=[chatbot, history_state, context_state],
    )


    # CLEAR CHAT ------------------------------------------
    def clear_chat():
        return [], [], {"last_symbol": None, "last_dates": None}

    clear_btn.click(clear_chat, None, [chatbot, history_state, context_state])

    # DOWNLOAD CHAT HISTORY -------------------------------
    def export_chat(history):
        if not history:
            return None

        text = "==== Chat History ====\n\n"
        for msg in history:
            role = msg["role"].capitalize()
            content = msg["content"]
            text += f"{role}: {content}\n\n"

        tmp_path = tempfile.mktemp(suffix=".txt")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(text)

        return tmp_path


    download_btn.click(
        fn=export_chat,
        inputs=[history_state],
        outputs=[chat_file]
    )


if __name__ == "__main__":
    demo.launch()










# import os
# import re
# import warnings
# import pandas as pd
# import gradio as gr
# from groq import Groq
# from rapidfuzz import process as fuzz

# from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS

# warnings.filterwarnings("ignore", category=DeprecationWarning)

# # ==========================================================
# # 🔑 CONFIG
# # ==========================================================
# MODEL_ID = "llama-3.1-8b-instant"
# client = Groq(api_key=GROQ_API_KEY)

# # NIFTY SYMBOLS
# NIFTY_SYMBOLS = [
#     "ADANIPORTS","ASIANPAINT","AXISBANK","BAJAJ-AUTO","BAJAJFINSV","BAJFINANCE",
#     "BHARTIARTL","BPCL","BRITANNIA","CIPLA","COALINDIA","DRREDDY","EICHERMOT",
#     "GAIL","GRASIM","HCLTECH","HDFC","HDFCBANK","HEROMOTOCO","HINDALCO",
#     "HINDUNILVR","ICICIBANK","INDUSINDBK","INFRATEL","INFY","IOC","ITC",
#     "JSWSTEEL","KOTAKBANK","LT","MARUTI","M&M","NESTLEIND","NTPC","ONGC",
#     "POWERGRID","RELIANCE","SBIN","SHREECEM","SUNPHARMA","TATAMOTORS",
#     "TATASTEEL","TCS","TECHM","TITAN","ULTRACEMCO","UPL","VEDL","WIPRO","ZEEL"
# ]

# # For fuzzy symbol inference
# COMPANY_NAME_MAP = {
#     "RELIANCE": ["reliance", "ril", "reliance industries"],
#     "TCS": ["tcs", "tata consultancy", "tata consultancy services"],
#     "INFY": ["infosys", "infy"],
#     "SBIN": ["sbi", "state bank", "state bank of india"],
#     "HDFC": ["hdfc"],
#     "HDFCBANK": ["hdfc bank"],
#     "ITC": ["itc", "itc ltd"],
#     "TATAMOTORS": ["tata motors"],
#     "POWERGRID": ["power grid"],
#     # extend as needed...
# }

# MARKET_DATA = {}   # (symbol, YYYY-MM-DD) → row dict
# STOCK_META = {}    # symbol → metadata dict


# # ==========================================================
# # Helpers
# # ==========================================================
# def project_path(*parts):
#     here = os.path.dirname(os.path.abspath(__file__))
#     return os.path.normpath(os.path.join(here, "..", "..", *parts))


# # ==========================================================
# # 📊 LOAD NIFTY50 STRUCTURED MARKET DATA
# # ==========================================================
# def load_market_data():
#     global MARKET_DATA, STOCK_META

#     nifty_dir = project_path("data", "NIFTY50")
#     print("📂 Loading NIFTY50 CSVs from:", nifty_dir)

#     if not os.path.isdir(nifty_dir):
#         print("⚠️ NIFTY50 folder missing.")
#         return

#     # ---- Load OHLCV data ----
#     for fname in os.listdir(nifty_dir):
#         if not fname.endswith(".csv"):
#             continue
#         if fname.lower() == "stock_metadata.csv":
#             continue

#         fpath = os.path.join(nifty_dir, fname)
#         try:
#             df = pd.read_csv(fpath)
#         except Exception:
#             continue

#         df.columns = [c.strip() for c in df.columns]
#         if "Date" not in df.columns or "Symbol" not in df.columns:
#             continue

#         df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

#         for _, row in df.iterrows():
#             sym = str(row["Symbol"]).upper()
#             dt = str(row["Date"])
#             if dt == "NaT":
#                 continue

#             key = (sym, dt)
#             MARKET_DATA[key] = {
#                 "Symbol": sym,
#                 "Date": dt,
#                 "Open": row.get("Open"),
#                 "High": row.get("High"),
#                 "Low": row.get("Low"),
#                 "Close": row.get("Close"),
#                 "VWAP": row.get("VWAP"),
#                 "Volume": row.get("Volume", row.get("Shares Traded")),
#                 "Turnover": row.get("Turnover", row.get("Turnover (Cr)")),
#                 "Trades": row.get("Trades"),
#                 "Deliverable Volume": row.get("Deliverable Volume"),
#                 "%Deliverable": row.get("%Deliverable", row.get("%Deliverble")),
#             }

#     print("✅ Loaded structured rows:", len(MARKET_DATA))

#     # ---- Load metadata ----
#     meta_path = os.path.join(nifty_dir, "stock_metadata.csv")
#     if os.path.exists(meta_path):
#         df = pd.read_csv(meta_path)
#         df.columns = [c.strip() for c in df.columns]

#         for _, r in df.iterrows():
#             s = str(r["Symbol"]).upper()
#             STOCK_META[s] = {
#                 "Company Name": r.get("Company Name", ""),
#                 "Industry": r.get("Industry", ""),
#                 "ISIN Code": r.get("ISIN Code", ""),
#                 "Series": r.get("Series", "")
#             }

#         print("📘 Loaded stock metadata:", len(STOCK_META))


# # ==========================================================
# # 🔍 FUZZY SYMBOL MATCHING
# # ==========================================================
# def fuzzy_symbol(text: str):
#     """
#     Returns:
#         best matching symbol OR None.
#     """
#     up = text.upper()

#     # 1. Direct symbol match
#     for s in NIFTY_SYMBOLS:
#         if s in up:
#             return s

#     # 2. Fuzzy match company aliases
#     for sym, words in COMPANY_NAME_MAP.items():
#         match, score, _ = fuzz.extractOne(text.lower(), words)
#         if score > 80:
#             return sym

#     # 3. Fuzzy match symbols directly
#     match, score, _ = fuzz.extractOne(up, NIFTY_SYMBOLS)
#     if score > 85:
#         return match

#     return None


# # ==========================================================
# # 📅 FUZZY DATE PARSING
# # ==========================================================
# MONTH_MAP = {
#     "jan": "01", "january": "01",
#     "feb": "02", "february": "02",
#     "mar": "03", "march": "03",
#     "apr": "04", "april": "04",
#     "may": "05",
#     "jun": "06", "june": "06",
#     "jul": "07", "july": "07",
#     "aug": "08", "august": "08",
#     "sep": "09", "september": "09",
#     "oct": "10", "october": "10",
#     "nov": "11", "november": "11",
#     "dec": "12", "december": "12",
# }


# def match_all_years(symbol: str, month: str, day: str):
#     """
#     Return list of all YYYY-MM-DD where this symbol traded on that month-day.
#     """
#     candidates = []
#     for (sym, dt), _row in MARKET_DATA.items():
#         if sym == symbol and dt[5:7] == month and dt[8:10] == day:
#             candidates.append(dt)

#     if len(candidates) == 0:
#         return None
#     elif len(candidates) == 1:
#         return candidates[0]
#     else:
#         return sorted(candidates)


# def fuzzy_date(text: str, symbol: str):
#     """
#     Returns:
#       - single YYYY-MM-DD string, or
#       - list of YYYY-MM-DD (multi-year match), or
#       - None
#     """
#     # 1. Try exact YYYY-MM-DD / YYYY/MM/DD
#     m = re.search(r"(\d{4})[-/](\d{2})[-/](\d{2})", text)
#     if m:
#         return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

#     # 2. "5th May" / "5 May"
#     m = re.search(
#         r"(\d{1,2})(st|nd|rd|th)?\s+"
#         r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*",
#         text,
#         re.IGNORECASE,
#     )
#     if m:
#         day = m.group(1).zfill(2)
#         month = MONTH_MAP[m.group(3).lower()]
#         return match_all_years(symbol, month, day)

#     # 3. "May 5" / "May 5th"
#     m = re.search(
#         r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2})(st|nd|rd|th)?",
#         text,
#         re.IGNORECASE,
#     )
#     if m:
#         month = MONTH_MAP[m.group(1).lower()]
#         day = m.group(2).zfill(2)
#         return match_all_years(symbol, month, day)

#     return None


# # ==========================================================
# # 📆 YEAR RANGE PARSER (for trend analysis)
# # ==========================================================
# def parse_year_range(text: str):
#     """
#     Detect patterns like:
#     - 2005-2010
#     - 2005–2010
#     - between 2005 and 2010
#     - from 2005 to 2010
#     Returns (start_year, end_year) or None
#     """

#     # 1. Direct range: "2005-2010" or "2005–2010"
#     m = re.search(r"\b((?:19|20)\d{2})\s*[-–]\s*((?:19|20)\d{2})\b", text)
#     if m:
#         y1 = int(m.group(1))
#         y2 = int(m.group(2))
#         return (min(y1, y2), max(y1, y2))

#     # 2. "between 2005 and 2010"
#     m = re.search(r"between\s+((?:19|20)\d{2})\s+and\s+((?:19|20)\d{2})", text, re.IGNORECASE)
#     if m:
#         y1 = int(m.group(1))
#         y2 = int(m.group(2))
#         return (min(y1, y2), max(y1, y2))

#     # 3. "from 2005 to 2010"
#     m = re.search(r"from\s+((?:19|20)\d{2})\s+to\s+((?:19|20)\d{2})", text, re.IGNORECASE)
#     if m:
#         y1 = int(m.group(1))
#         y2 = int(m.group(2))
#         return (min(y1, y2), max(y1, y2))

#     return None


# def multi_year_intent(text: str) -> bool:
#     """
#     Detect if user wants multi-year comparison for same day-month.
#     """
#     kwords = [
#         "across all years",
#         "across all may",  # generic
#         "over the years",
#         "across years",
#         "every year",
#         "each year",
#         "year-wise",
#         "year wise",
#         "historically",
#         "on this date over",
#         "on this day over",
#     ]
#     low = text.lower()
#     return any(k in low for k in kwords)


# def trend_intent(text: str) -> bool:
#     """
#     Detect if user is asking for multi-year trend analysis.
#     """
#     kwords = [
#         "trend",
#         "performance",
#         "between",
#         "from",
#         "to",
#         "over the period",
#         "over the years",
#     ]
#     low = text.lower()
#     return any(k in low for k in kwords)


# # ==========================================================
# # STRUCTURED ANSWERS (single day)
# # ==========================================================
# def structured_answer(symbol: str, date_str: str):
#     key = (symbol, date_str)
#     row = MARKET_DATA.get(key)
#     if not row:
#         return None

#     meta = STOCK_META.get(symbol, {})

#     lines = [
#         f"Here is the trading summary for {symbol} on {date_str}:",
#         f"- Open: {row['Open']}",
#         f"- High: {row['High']}",
#         f"- Low: {row['Low']}",
#         f"- Close: {row['Close']}",
#         f"- VWAP: {row['VWAP']}",
#         f"- Volume: {row['Volume']}",
#         f"- Turnover: {row['Turnover']}",
#         f"- Trades: {row['Trades']}",
#     ]

#     lines.append("")
#     lines.append("Stock metadata:")
#     lines.append(f"- Company: {meta.get('Company Name', '')}")
#     lines.append(f"- Industry: {meta.get('Industry', '')}")
#     lines.append(f"- ISIN: {meta.get('ISIN Code', '')}")

#     return "\n".join(lines)


# # ==========================================================
# # MULTI-YEAR SAME-DATE COMPARISON
# # ==========================================================
# def multi_year_comparison(symbol: str, date_list):
#     """
#     date_list: list of YYYY-MM-DD for same month-day across years.
#     """
#     rows = []
#     for dt in sorted(date_list):
#         r = MARKET_DATA.get((symbol, dt))
#         if r:
#             rows.append(r)

#     if not rows:
#         return f"I could not find any structured records for {symbol} on those dates."

#     lines = [f"📊 Multi-year comparison for {symbol} on this calendar date:\n"]
#     closes = []
#     years = []

#     for r in rows:
#         year = r["Date"][:4]
#         years.append(int(year))
#         close = r["Close"]
#         closes.append(float(close) if close is not None and not pd.isna(close) else None)

#         lines.append(
#             f"- {r['Date']}: Close {r['Close']}, "
#             f"Open {r['Open']}, High {r['High']}, Low {r['Low']}, "
#             f"Volume {r['Volume']}"
#         )

#     # Basic trend note if we have at least two valid closes
#     valid_closes = [c for c in closes if c is not None]
#     if len(valid_closes) >= 2:
#         first_close = valid_closes[0]
#         last_close = valid_closes[-1]
#         try:
#             pct = (last_close - first_close) / first_close * 100.0
#             lines.append("")
#             if pct > 0:
#                 lines.append(
#                     f"Overall, between {rows[0]['Date']} and {rows[-1]['Date']}, "
#                     f"the closing price on this date increased by approximately {pct:.2f}%."
#                 )
#             else:
#                 lines.append(
#                     f"Overall, between {rows[0]['Date']} and {rows[-1]['Date']}, "
#                     f"the closing price on this date decreased by approximately {abs(pct):.2f}%."
#                 )
#         except Exception:
#             pass

#     return "\n".join(lines)


# # ==========================================================
# # TREND ANALYSIS OVER YEAR RANGE
# # ==========================================================
# def trend_analysis(symbol: str, start_year: int, end_year: int):
#     """
#     Aggregates data for symbol between start_year and end_year (inclusive),
#     computes summary stats, then asks LLM to explain the trend.
#     """
#     all_rows = []
#     for (sym, dt), row in MARKET_DATA.items():
#         if sym != symbol:
#             continue
#         year = int(dt[:4])
#         if start_year <= year <= end_year:
#             all_rows.append(row)

#     if not all_rows:
#         return (
#             f"I searched for {symbol} between {start_year} and {end_year}, "
#             "but found no trading records in that range."
#         )

#     all_rows = sorted(all_rows, key=lambda r: r["Date"])
#     # Collect stats
#     def safe_float(x):
#         try:
#             if x is None or pd.isna(x):
#                 return None
#             return float(x)
#         except Exception:
#             return None

#     # first and last close
#     first_close = None
#     first_date = None
#     for r in all_rows:
#         c = safe_float(r["Close"])
#         if c is not None:
#             first_close = c
#             first_date = r["Date"]
#             break

#     last_close = None
#     last_date = None
#     for r in reversed(all_rows):
#         c = safe_float(r["Close"])
#         if c is not None:
#             last_close = c
#             last_date = r["Date"]
#             break

#     highs = [safe_float(r["High"]) for r in all_rows]
#     lows = [safe_float(r["Low"]) for r in all_rows]
#     vols = [safe_float(r["Volume"]) for r in all_rows]

#     highs = [h for h in highs if h is not None]
#     lows = [l for l in lows if l is not None]
#     vols = [v for v in vols if v is not None]

#     overall_high = max(highs) if highs else None
#     overall_low = min(lows) if lows else None
#     avg_vol = sum(vols) / len(vols) if vols else None

#     stats_str = [
#         f"Trend summary for {symbol} between {start_year} and {end_year}:",
#         f"- Number of trading records found: {len(all_rows)}",
#     ]
#     if first_close is not None and last_close is not None:
#         try:
#             pct = (last_close - first_close) / first_close * 100.0
#             stats_str.append(
#                 f"- First close ({first_date}): {first_close}, "
#                 f"Last close ({last_date}): {last_close}, "
#                 f"Total change: {pct:.2f}%"
#             )
#         except Exception:
#             pass
#     if overall_high is not None:
#         stats_str.append(f"- Highest recorded price: {overall_high}")
#     if overall_low is not None:
#         stats_str.append(f"- Lowest recorded price: {overall_low}")
#     if avg_vol is not None:
#         stats_str.append(f"- Average volume: {avg_vol:.0f}")

#     stats_block = "\n".join(stats_str)

#     # Ask LLM to interpret this trend
#     prompt = f"""
# You are a financial analyst. Based on the following summary statistics,
# explain the multi-year price trend of {symbol} between {start_year} and {end_year}.
# Discuss bull vs bear phases, volatility, and investor sentiment in 2–3 paragraphs.

# {stats_block}
# """

#     resp = client.chat.completions.create(
#         model=MODEL_ID,
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=400,
#         temperature=0.4,
#     )

#     explanation = resp.choices[0].message.content.strip()
#     return stats_block + "\n\n📘 Trend Interpretation:\n" + explanation


# # ==========================================================
# # HYBRID SINGLE-DAY EXPLANATION
# # ==========================================================
# def hybrid_explanation(symbol: str, date_str: str):
#     key = (symbol, date_str)
#     row = MARKET_DATA.get(key)
#     if not row:
#         return None

#     base_summary = structured_answer(symbol, date_str)

#     prompt = f"""
# Explain the price action of {symbol} on {date_str} using financial theory.

# Open: {row['Open']}
# High: {row['High']}
# Low: {row['Low']}
# Close: {row['Close']}
# VWAP: {row['VWAP']}
# Volume: {row['Volume']}

# Discuss:
# - Intraday volatility
# - Direction from open to close
# - Relation to VWAP
# - Demand/supply sentiment
# - Possible market psychology
# """

#     resp = client.chat.completions.create(
#         model=MODEL_ID,
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=350,
#         temperature=0.4,
#     )

#     explanation = resp.choices[0].message.content.strip()
#     return base_summary + "\n\n📘 Market Explanation:\n" + explanation


# # ==========================================================
# # RAG INDEX BUILDING
# # ==========================================================
# def load_or_build_rag():
#     index_path = project_path("data", "finance_index")

#     if os.path.exists(index_path):
#         print("📦 Loading existing FAISS index…")
#         emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         try:
#             return FAISS.load_local(index_path, emb, allow_dangerous_deserialization=True)
#         except Exception:
#             print("⚠️ Failed to load, rebuilding…")

#     print("📥 Building FAISS index from financial_corpus…")
#     corpus = project_path("data", "financial_corpus")

#     loader = DirectoryLoader(corpus, glob="**/*.txt", loader_cls=TextLoader)

#     docs = []
#     for p in loader.file_paths:
#         try:
#             t = open(p, "r", encoding="utf-8").read()
#             docs.append(t)
#         except Exception:
#             print("⚠️ Skipping unreadable file:", p)

#     splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
#     chunks = splitter.split_text("\n".join(docs))

#     emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vect = FAISS.from_texts(chunks, emb)
#     vect.save_local(index_path)

#     print("✅ FAISS index saved.")
#     return vect


# vectorstore = load_or_build_rag()
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# load_market_data()


# # ==========================================================
# # RAG ANSWERING
# # ==========================================================
# def rag_answer(question: str):
#     docs = retriever.invoke(question)
#     ctx = "\n\n".join(d.page_content for d in docs)

#     prompt = f"""
# Use ONLY the context below. If the exact answer is not present, give the
# closest relevant financial insight and suggest how the user can refine
# the question (e.g., by giving a specific date or stock).

# Context:
# {ctx}

# Question:
# {question}
# """

#     resp = client.chat.completions.create(
#         model=MODEL_ID,
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=400,
#         temperature=0.3,
#     )

#     return resp.choices[0].message.content.strip()


# # ==========================================================
# # MAIN ROUTER
# # ==========================================================
# def handle_query(q: str):
#     q = q.strip()
#     if not q:
#         return "Please enter a question."

#     # 1) Fuzzy symbol detection (central to all structured logic)
#     symbol = fuzzy_symbol(q)

#     # 2) Trend analysis over year range (e.g., "INFY performance between 2009–2014")
#     if symbol and trend_intent(q):
#         yr_range = parse_year_range(q)
#         if yr_range:
#             y1, y2 = yr_range
#             return trend_analysis(symbol, y1, y2)

#     # 3) Single-date or multi-year same-date logic
#     if symbol:
#         fd = fuzzy_date(q, symbol)

#         # Single exact date → either hybrid explanation or raw structured
#         if isinstance(fd, str):
#             # If user says "explain", do hybrid theory answer
#             if "explain" in q.lower():
#                 ans = hybrid_explanation(symbol, fd)
#                 if ans:
#                     return ans

#             ans = structured_answer(symbol, fd)
#             if ans:
#                 return ans
#             else:
#                 return f"I searched for {symbol} on {fd}, but found no exact match in my structured data."

#         # Multiple matching dates across years → multi-year comparison or clarification
#         if isinstance(fd, list):
#             if multi_year_intent(q):
#                 return multi_year_comparison(symbol, fd)
#             else:
#                 return (
#                     f"I found multiple years where {symbol} traded on this date:\n"
#                     + "\n".join("• " + d for d in fd)
#                     + "\n\nPlease specify the exact year (YYYY-MM-DD), "
#                       "or ask 'Compare {symbol} prices across all these years' for a multi-year view."
#                 )

#     # 4) Otherwise: conceptual finance via RAG
#     return rag_answer(q)


# # ==========================================================
# # UI
# # ==========================================================
# import tempfile

# with gr.Blocks(title="💬 Financial Assistant Chatbot") as demo:

#     gr.Markdown("""
#     # 💹 Financial LLM — NIFTY50 QA + RAG + Translation  
#     Ask any stock/date question, conceptual finance question, or translation.  
#     Now with full chatbot history + export!
#     """)

#     chatbot = gr.Chatbot(height=500, label="Financial Assistant")
#     user_msg = gr.Textbox(
#         placeholder="Ask anything e.g., 'Explain RELIANCE on 2010-11-12' ...",
#         label="Your Question"
#     )
#     send_btn = gr.Button("Send")
#     clear_btn = gr.Button("Clear Chat")
#     download_btn = gr.Button("Download Chat History")
#     chat_file = gr.File(label="Download history", interactive=False)


#     # -----------------------
#     # INTERNAL STATE (history)
#     # -----------------------
#     history_state = gr.State([])   # list of (user, bot)


#     # -----------------------
#     # CHAT HANDLER
#     # -----------------------
#     def chat_handler(message, history):
#         bot_reply = handle_query(message)

#     # Convert to Chatbot required format:
#         history.append({"role": "user", "content": message})
#         history.append({"role": "assistant", "content": bot_reply})

#         return history, history

#     # -----------------------
#     # CLEAR CHAT
#     # -----------------------
#     def clear_chat():
#         return [], []

#     clear_btn.click(
#         fn=clear_chat,
#         inputs=None,
#         outputs=[chatbot, history_state]
#     )


#     # -----------------------
#     # DOWNLOAD CHAT HISTORY
#     # -----------------------
#     def export_chat(history):
#         if not history:
#             return None

#         text = "==== Chat History ====\n\n"
#         for u, b in history:
#             text += f"User: {u}\nBot: {b}\n\n"

#         tmp_path = tempfile.mktemp(suffix=".txt")
#         with open(tmp_path, "w", encoding="utf-8") as f:
#             f.write(text)

#         return tmp_path

#     download_btn.click(
#         fn=export_chat,
#         inputs=[history_state],
#         outputs=[chat_file]
#     )


# if __name__ == "__main__":
#     demo.launch()











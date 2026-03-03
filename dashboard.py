import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from data_fetch import get_stock_data
from monte_carlo import run_simulation
from risk_metrics import calculate_metrics

st.set_page_config(page_title="Pro Stock Forecaster", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# STOCK REGISTRY:  ticker → (display_name, person_name)
# ─────────────────────────────────────────────────────────────────────────────
STOCK_INFO = {
    # Auto (India)
    "TATAMOTORS.NS": ("Tata Motors", "Rithin Reji"),
    "M&M.NS":        ("Mahindra & Mahindra", "Vinamra Gupta"),
    "OLECTRA.NS":    ("Olectra Greentech", "Aryan Jha"),
    "ATHERENERG.NS": ("Ather Energy", ""),
    # Auto (Global)
    "TSLA":    ("Tesla Inc.", ""),
    "P911.DE": ("Porsche AG", "Gautam Poturaju"),
    "F":       ("Ford Motor Co.", "Archana V"),
    "VOW3.DE": ("Volkswagen AG", "Sunidhi Datar"),
    "HYMTF":   ("Hyundai Motor", "Samarth Rao"),
    # Tyres (India)
    "APOLLOTYRE.NS": ("Apollo Tyres", "Anirudh Agarwal"),
    "MRF.NS":        ("MRF Ltd.", "Shrisai Hari"),
    "JKTYRE.NS":     ("JK Tyre & Industries", "Swayam Panigrahi"),
    "CEATLTD.NS":    ("CEAT Ltd.", "Harshini Venkat"),
    # Banking (India)
    "SBIN.NS":      ("State Bank of India", "Anoushka Gadhwal"),
    "HDFCBANK.NS":  ("HDFC Bank", "Ryan Kidangan"),
    "ICICIBANK.NS": ("ICICI Bank", "Himangshi Bose"),
    "AXISBANK.NS":  ("Axis Bank", "Bismaya Nayak"),
    # Pharma (India)
    "LAURUSLABS.NS": ("Laurus Labs", "Satvik Sharma"),
    "AUROPHARMA.NS": ("Aurobindo Pharma", "Arya Mukharjee"),
    "SUNPHARMA.NS":  ("Sun Pharma", "Yogesh Bolkotagi"),
    "DIVISLAB.NS":   ("Divi's Laboratories", "Bhavansh Madan"),
    # Consumer & Hotels (India)
    "ITC.NS":       ("ITC Ltd.", "Gajanan Kudva / Srutayus Das"),
    "CHALET.NS":    ("Chalet Hotels", "Shreya Joshi"),
    "MHRIL.NS":     ("Mahindra Holidays", "Gowri Shetty"),
    "INDHOTEL.NS":  ("Indian Hotels Co.", "Aarohi Jain"),
    "HUL.NS":       ("Hindustan Unilever", "Suhina Sarkar"),
    "NESTLEIND.NS": ("Nestlé India", "Saaraansh Razdan"),
    # Cement (India)
    "SHREECEM.NS":   ("Shree Cement", "Anjor Singh"),
    "ULTRACEMCO.NS": ("UltraTech Cement", "Rahul Gowda"),
    "DALBHARAT.NS":  ("Dalmia Bharat", "Kushagra Shukla"),
    "RAMCOCEM.NS":   ("Ramco Cements", "Grace Rebecca David"),
    # AMC / Finance (India)
    "ABSLAMC.NS":    ("Aditya Birla Sun Life AMC", "Pallewar Pranav"),
    "HDFCAMC.NS":    ("HDFC AMC", "Rittika Saraswat"),
    "NAM-INDIA.NS":  ("Nippon Life India AMC", "Sam Phillips"),
    "UTIAMC.NS":     ("UTI AMC", "Abhinav Singh"),
    # Tech (US / Global)
    "NVDA":  ("NVIDIA Corp.", "Sijal Verma"),
    "MSFT":  ("Microsoft Corp.", "Gurleen Kaur"),
    "GOOGL": ("Alphabet Inc.", "Anugraha AB"),
    "META":  ("Meta Platforms", "Senjuti Pal"),
    "IBM":   ("IBM Corp.", "Biba Pattnaik"),
    "ASML":  ("ASML Holding", "Adaa Gujral"),
    "INTC":  ("Intel Corp.", "Aditi Ranjan"),
    "QCOM":  ("Qualcomm Inc.", "Arpit Sharma"),
    "CRM":   ("Salesforce Inc.", "Rishit Hotchandani"),
    "PLTR":  ("Palantir Technologies", "Krrish Bahuguna"),
    "CRWD":  ("CrowdStrike Holdings", "Ashi Beniwal"),
    "ORCL":  ("Oracle Corp.", "Ruchita Gowri"),
    # Media & Consumer (US)
    "WBD":  ("Warner Bros. Discovery", "Dhairya Vanker"),
    "NFLX": ("Netflix Inc.", "Hiya Phatnani"),
    "DIS":  ("Walt Disney Co.", "Siya Sharma"),
    "PARA": ("Paramount Global", "Tanvi Gujarathi"),
    "PG":   ("Procter & Gamble", "Nayan Kanchan"),
    "WMT":  ("Walmart Inc.", ""),
    # Defense (US)
    "LMT": ("Lockheed Martin", "Siddhant Mehta"),
    "GD":  ("General Dynamics", "Shlok Pratap Singh"),
    "NOC": ("Northrop Grumman", "Harshdeep Roshan"),
    "RTX": ("RTX Corporation", "Prandeep Poddar"),
}

# ── Helper: is this an Indian stock? ────────────────────────────────────────
def _is_indian(ticker: str) -> bool:
    return ticker.endswith(".NS") or ticker.endswith(".BO")


def _currency_symbol(ticker: str) -> str:
    return "₹" if _is_indian(ticker) else "$"


def _fmt_price(value, ticker):
    sym = _currency_symbol(ticker)
    return f"{sym}{value:,.2f}"


# ── Build the sector → ticker list ──────────────────────────────────────────
GLOBAL_STOCKS = {
    "Auto (India)":              ["TATAMOTORS.NS", "M&M.NS", "OLECTRA.NS", "ATHERENERG.NS"],
    "Auto (Global)":             ["TSLA", "P911.DE", "F", "VOW3.DE", "HYMTF"],
    "Tyres (India)":             ["APOLLOTYRE.NS", "MRF.NS", "JKTYRE.NS", "CEATLTD.NS"],
    "Banking (India)":           ["SBIN.NS", "HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS"],
    "Pharma (India)":            ["LAURUSLABS.NS", "AUROPHARMA.NS", "SUNPHARMA.NS", "DIVISLAB.NS"],
    "Consumer & Hotels (India)": ["ITC.NS", "CHALET.NS", "MHRIL.NS", "INDHOTEL.NS", "HUL.NS", "NESTLEIND.NS"],
    "Cement (India)":            ["SHREECEM.NS", "ULTRACEMCO.NS", "DALBHARAT.NS", "RAMCOCEM.NS"],
    "AMC / Finance (India)":     ["ABSLAMC.NS", "HDFCAMC.NS", "NAM-INDIA.NS", "UTIAMC.NS"],
    "Tech (US/Global)":          ["NVDA", "MSFT", "GOOGL", "META", "IBM", "ASML", "INTC", "QCOM", "CRM", "PLTR", "CRWD"],
    "Media & Consumer (US)":     ["WBD", "NFLX", "DIS", "PARA", "PG", "WMT"],
    "Defense (US)":              ["LMT", "GD", "NOC", "RTX"],
}


def _display_name(ticker):
    """Returns 'Company Name (Person)' or just 'Company Name'."""
    info = STOCK_INFO.get(ticker)
    if info:
        company, person = info
        return f"{company} ({person})" if person else company
    return ticker


# ═════════════════════════════════════════════════════════════════════════════
#  STREAMLIT APP
# ═════════════════════════════════════════════════════════════════════════════
st.title("🏛️ Institutional Equity Lab & Forecasting")
st.markdown("---")

# ── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.header("1. Market Selection")
category = st.sidebar.selectbox("Filter by Sector", list(GLOBAL_STOCKS.keys()))

# Show company name + person in the dropdown
ticker_list = GLOBAL_STOCKS[category]
display_list = [_display_name(t) for t in ticker_list]
selected_display = st.sidebar.selectbox("Select Company", display_list)
selected_ticker = ticker_list[display_list.index(selected_display)]

custom_ticker = st.sidebar.text_input("OR Type Custom (e.g. RELIANCE.NS)")
ticker = custom_ticker.strip() if custom_ticker.strip() else selected_ticker
cur = _currency_symbol(ticker)

st.sidebar.header("2. Simulation Parameters")
sims = st.sidebar.slider("Number of Simulations", 5000, 50000, 10000)
years = st.sidebar.slider("Investment Horizon (Years)", 0.5, 10.0, 1.0)
crash_scenario = st.sidebar.slider("Simulate Market Stress (%)", 0, 50, 0)

try:
    # 1. Fetch data (multi-source with auto failover)
    with st.spinner(f"Fetching data for {ticker} (trying multiple sources)…"):
        s0, auto_mu, auto_sigma, source_name = get_stock_data(ticker)

    company_label = _display_name(ticker)
    st.sidebar.success(f"✅ {company_label}\nSource: {source_name}")

    # 2. Adjust for stress scenario
    adjusted_mu = auto_mu - (crash_scenario / 100)

    # 3. Run Monte-Carlo simulation
    with st.spinner("Running simulation…"):
        paths, low_band, high_band = run_simulation(
            s0, adjusted_mu, auto_sigma, years, n_sims=sims
        )
    final_prices = paths[-1]

    # 4. Calculate metrics
    metrics = calculate_metrics(final_prices, s0, adjusted_mu, auto_sigma)

    # ── COMPANY HEADER ──────────────────────────────────────────────────────
    info = STOCK_INFO.get(ticker, (ticker, ""))
    co_name, person = info
    person_badge = f"&nbsp;&nbsp;👤 <i>{person}</i>" if person else ""
    st.markdown(
        f"<h2>{co_name} &nbsp;<code>{ticker}</code>{person_badge}</h2>",
        unsafe_allow_html=True,
    )

    # ── TOP SIGNAL BANNER ───────────────────────────────────────────────────
    if "BUY" in metrics["Signal"]:
        sig_color = "#1b8a2a"
    elif "HOLD" in metrics["Signal"]:
        sig_color = "#c47f17"
    else:
        sig_color = "#b52a2a"

    st.markdown(
        f"""
        <div style="background:{sig_color};padding:25px;border-radius:12px;
                     text-align:center;margin-bottom:25px;">
            <h1 style="color:white;margin:0;">ANALYSIS SIGNAL: {metrics['Signal']}</h1>
            <h3 style="color:white;margin-top:10px;">
                Price Target in {years} yr: <b>{_fmt_price(metrics['Expected Price'], ticker)}</b>
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KEY METRICS ROW ─────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", _fmt_price(s0, ticker))
    c2.metric(
        "Target (Expected)",
        _fmt_price(metrics["Expected Price"], ticker),
        f"{((metrics['Expected Price'] / s0) - 1):.1%}",
    )
    c3.metric("Downside Risk (VaR 95%)", f"{metrics['VaR 95% (Rel)']:.1%}")
    c4.metric("Prob. of Profit", f"{metrics['Prob. of Profit']:.1f}%")

    st.markdown("---")

    # ── DETAILED METRICS ────────────────────────────────────────────────────
    st.subheader("📋 Detailed Analysis")

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.markdown("**📈 Price Targets**")
        st.write(f"Expected: {_fmt_price(metrics['Expected Price'], ticker)}")
        st.write(f"Median: {_fmt_price(metrics['Median Price'], ticker)}")
        st.write(f"Best Case: {_fmt_price(metrics['Best Case Price'], ticker)}")
        st.write(f"Worst Case: {_fmt_price(metrics['Worst Case Price'], ticker)}")
        st.write(f"90th %-ile: {_fmt_price(metrics['90th Percentile Price'], ticker)}")
        st.write(f"10th %-ile: {_fmt_price(metrics['10th Percentile Price'], ticker)}")

    with d2:
        st.markdown("**⚠️ Risk Metrics**")
        st.write(f"VaR 95%: {metrics['VaR 95% (Rel)']:.2%}")
        st.write(f"CVaR 95%: {metrics['CVaR 95%']:.2%}")
        st.write(f"VaR 99%: {metrics['VaR 99% (Rel)']:.2%}")
        st.write(f"CVaR 99%: {metrics['CVaR 99%']:.2%}")
        st.write(f"Max Drawdown: {metrics['Max Drawdown']:.1f}%")
        st.write(f"Annual Volatility: {metrics['Volatility (Annual)']:.1f}%")

    with d3:
        st.markdown("**📊 Probability Analysis**")
        st.write(f"Prob. of Profit: {metrics['Prob. of Profit']:.1f}%")
        st.write(f"Prob. of >10% Gain: {metrics['Prob. of >10% Gain']:.1f}%")
        st.write(f"Prob. of >25% Gain: {metrics['Prob. of >25% Gain']:.1f}%")
        st.write(f"Prob. of >10% Loss: {metrics['Prob. of >10% Loss']:.1f}%")
        st.write(f"Avg Upside: +{metrics['Avg Upside']:.1f}%")
        st.write(f"Avg Downside: {metrics['Avg Downside']:.1f}%")

    with d4:
        st.markdown("**🏆 Performance Ratios**")
        st.write(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
        st.write(f"Sortino Ratio: {metrics['Sortino Ratio']:.2f}")
        st.write(f"Risk-Reward: {metrics['Risk-Reward Ratio']:.2f}")
        st.write(f"Expected Return: {metrics['Expected Return']:.1f}%")
        st.write(f"Max Upside: +{metrics['Max Upside']:.1f}%")

    st.markdown("---")

    # ── PLOTS ───────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        fig_paths = go.Figure()
        x_axis = np.arange(len(low_band))
        fig_paths.add_trace(go.Scatter(
            x=x_axis, y=high_band, fill=None, mode="lines",
            line_color="rgba(0,255,0,0.1)", name="Top 5% Outcome",
        ))
        fig_paths.add_trace(go.Scatter(
            x=x_axis, y=low_band, fill="tonexty", mode="lines",
            line_color="rgba(255,0,0,0.1)", name="Worst 5% Outcome",
        ))
        fig_paths.add_trace(go.Scatter(
            y=np.mean(paths, axis=1), mode="lines",
            line=dict(color="gold", width=3), name="Expected Growth",
        ))
        fig_paths.update_layout(
            title="Monte Carlo Confidence Bands",
            template="plotly_dark",
            xaxis_title="Trading Days",
            yaxis_title=f"Price ({cur})",
        )
        st.plotly_chart(fig_paths, use_container_width=True)

    with col_right:
        fig_hist = px.histogram(
            final_prices, nbins=60,
            title="Terminal Price Distribution",
            template="plotly_dark",
            color_discrete_sequence=["#00CC96"],
        )
        fig_hist.add_vline(
            x=s0, line_dash="dash", line_color="yellow",
            annotation_text=f"Current {_fmt_price(s0, ticker)}",
        )
        fig_hist.add_vline(
            x=metrics["Expected Price"], line_dash="dot", line_color="cyan",
            annotation_text=f"Target {_fmt_price(metrics['Expected Price'], ticker)}",
        )
        fig_hist.update_layout(xaxis_title=f"Price ({cur})", yaxis_title="Frequency")
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── BENCHMARK COMPARISON ────────────────────────────────────────────────
    st.subheader("📊 Performance vs. Market Benchmarks")
    if _is_indian(ticker):
        bench_name = "Nifty 50 Average (12% Annually)"
        bench_rate = 1.12
    else:
        bench_name = "S&P 500 Average (10% Annually)"
        bench_rate = 1.10

    market_growth = s0 * (bench_rate ** years)
    comparison_data = {
        "Scenario": [f"{co_name} Forecasted", bench_name],
        "Expected Price": [
            _fmt_price(metrics["Expected Price"], ticker),
            _fmt_price(market_growth, ticker),
        ],
        "Potential Return": [
            f"{((metrics['Expected Price'] / s0) - 1):.1%}",
            f"{((market_growth / s0) - 1):.1%}",
        ],
    }
    st.table(pd.DataFrame(comparison_data))

    # ── FOOTER ──────────────────────────────────────────────────────────────
    st.caption(
        f"Data source: {source_name} • "
        f"Simulations: {sims:,} • "
        f"Horizon: {years} yr • "
        f"Stress applied: {crash_scenario}%"
    )

except ValueError as e:
    st.error(f"Data Error: {e}")
except Exception as e:
    st.error(f"Unexpected error for {ticker}: {type(e).__name__}: {e}")
    st.exception(e)
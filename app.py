"""
Streamlit web interface for DeribitOptionsScanner.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

from scanner import DeribitOptionsScanner, OptionFilters

load_dotenv()

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Deribit Options Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Deribit Options Scanner")
st.caption("Сканер опционов в реальном времени на основе данных Deribit API")

# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ Параметры сканирования")

scan_mode = st.sidebar.radio(
    "Режим",
    ["Быстрый", "Продвинутый"],
    help="Быстрый режим использует готовые значения, продвинутый открывает все настройки.",
)

available_currencies = DeribitOptionsScanner().get_supported_option_currencies()
if not available_currencies:
    available_currencies = ["BTC", "ETH"]

currency_default_index = available_currencies.index("BTC") if "BTC" in available_currencies else 0
currency = st.sidebar.selectbox("Валюта", available_currencies, index=currency_default_index)

pair_type_ui = st.sidebar.selectbox(
    "Тип пар",
    ["Все", "Только инверсные", "Только неинверсные"],
    index=0,
)
pair_type_map = {
    "Все": "all",
    "Только инверсные": "inverse",
    "Только неинверсные": "non_inverse",
}
pair_type = pair_type_map[pair_type_ui]

st.sidebar.subheader("Диапазоны фильтров")

defaults = {
    "Быстрый": {"iv_min": 0.25, "iv_max": 1.2, "delta_min": -0.35, "delta_max": 0.35, "dte_min": 7, "dte_max": 60},
    "Продвинутый": {"iv_min": 0.2, "iv_max": 1.5, "delta_min": -0.5, "delta_max": 0.5, "dte_min": 7, "dte_max": 90},
}

col_iv1, col_iv2 = st.sidebar.columns(2)
iv_min = col_iv1.number_input("IV мин", min_value=0.0, max_value=5.0, value=defaults[scan_mode]["iv_min"], step=0.05)
iv_max = col_iv2.number_input("IV макс", min_value=0.0, max_value=5.0, value=defaults[scan_mode]["iv_max"], step=0.05)

col_d1, col_d2 = st.sidebar.columns(2)
delta_min = col_d1.number_input("Delta мин", min_value=-1.0, max_value=1.0, value=defaults[scan_mode]["delta_min"], step=0.05)
delta_max = col_d2.number_input("Delta макс", min_value=-1.0, max_value=1.0, value=defaults[scan_mode]["delta_max"], step=0.05)

col_dte1, col_dte2 = st.sidebar.columns(2)
dte_min = col_dte1.number_input("DTE мин", min_value=0, max_value=365, value=defaults[scan_mode]["dte_min"], step=1)
dte_max = col_dte2.number_input("DTE макс", min_value=0, max_value=730, value=defaults[scan_mode]["dte_max"], step=1)

min_volume = st.sidebar.number_input("Мин. объём (BTC)", min_value=0.0, value=2.0, step=0.5)
min_oi = st.sidebar.number_input("Мин. открытый интерес (BTC)", min_value=0.0, value=20.0, step=5.0)
exclude_perp = st.sidebar.checkbox("Исключить PERPETUAL", value=True)

st.sidebar.subheader("Дополнительные сканеры")
iv_threshold = st.sidebar.slider(
    "Порог IV Rank для High-IV скана (%)", min_value=50, max_value=100, value=85
)

st.sidebar.subheader("API")
api_key = st.sidebar.text_input(
    "API Key (опционально)", value=os.getenv("DERIBIT_API_KEY", ""), type="password"
)
api_secret = st.sidebar.text_input(
    "API Secret (опционально)", value=os.getenv("DERIBIT_API_SECRET", ""), type="password"
)

# ---------------------------------------------------------------------------
# Scanner initialisation (cached per session)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def init_scanner(key: str, secret: str) -> DeribitOptionsScanner:
    return DeribitOptionsScanner(
        api_key=key or None,
        api_secret=secret or None,
    )


scanner = init_scanner(api_key, api_secret)

# ---------------------------------------------------------------------------
# Helper — build OptionFilters from sidebar values
# ---------------------------------------------------------------------------


def build_filters() -> OptionFilters:
    return OptionFilters(
        currency=currency,
        min_volume=min_volume,
        min_open_interest=min_oi,
        iv_min=iv_min,
        iv_max=iv_max,
        delta_min=delta_min,
        delta_max=delta_max,
        dte_min=int(dte_min),
        dte_max=int(dte_max),
        exclude_perpetual=exclude_perp,
        instrument_type=pair_type,
    )


# ---------------------------------------------------------------------------
# Helper — render a results table + charts
# ---------------------------------------------------------------------------

DISPLAY_COLS = [
    "instrument_name", "option_type", "strike", "dte",
    "pair_type", "quote_currency",
    "iv", "delta", "gamma", "theta", "vega",
    "volume", "open_interest", "liquidity_score",
    "iv_rank", "moneyness", "spread_pct",
    "premium_quote", "long_max_loss", "long_max_profit",
    "short_max_profit", "short_max_loss",
]


def get_available_columns(df: pd.DataFrame, cols: list) -> list:
    return [c for c in cols if c in df.columns]


def _format_risk_reward_view(df: pd.DataFrame) -> pd.DataFrame:
    view = df.copy()
    for col in ["long_max_profit", "short_max_loss"]:
        if col in view.columns:
            view[col] = view[col].apply(lambda v: "∞" if pd.notna(v) and v == float("inf") else v)
    return view


def render_results(df: pd.DataFrame, scan_label: str, scanner_ref: DeribitOptionsScanner) -> None:
    if df.empty:
        st.warning(f"Нет данных по запросу «{scan_label}». Попробуйте изменить фильтры.")
        return

    st.success(f"✅ Найдено опционов: **{len(df)}**")

    # ---- Table ----
    show_cols = get_available_columns(df, DISPLAY_COLS)
    st.dataframe(
        _format_risk_reward_view(df[show_cols]).reset_index(drop=True),
        use_container_width=True,
        height=320,
    )

    # ---- Export ----
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Скачать CSV",
        data=csv_data,
        file_name=f"{scan_label.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )

    # ---- Charts ----
    col1, col2 = st.columns(2)

    if "iv" in df.columns and "liquidity_score" in df.columns:
        with col1:
            fig = px.scatter(
                df,
                x="iv",
                y="liquidity_score",
                color="option_type" if "option_type" in df.columns else None,
                hover_data=get_available_columns(df, ["instrument_name", "strike", "dte", "delta"]),
                title="IV vs Liquidity Score",
                labels={"iv": "Implied Volatility", "liquidity_score": "Liquidity Score"},
            )
            st.plotly_chart(fig, use_container_width=True)

    if "dte" in df.columns and "iv" in df.columns:
        with col2:
            fig2 = px.scatter(
                df,
                x="dte",
                y="iv",
                color="option_type" if "option_type" in df.columns else None,
                size="open_interest" if "open_interest" in df.columns else None,
                hover_data=get_available_columns(df, ["instrument_name", "strike", "delta"]),
                title="DTE vs IV (размер = Open Interest)",
                labels={"dte": "Days to Expiration", "iv": "Implied Volatility"},
            )
            st.plotly_chart(fig2, use_container_width=True)

    if "delta" in df.columns and "iv" in df.columns:
        fig3 = px.histogram(
            df,
            x="delta",
            color="option_type" if "option_type" in df.columns else None,
            nbins=40,
            title="Распределение Delta",
            labels={"delta": "Delta"},
        )
        st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_general, tab_high_iv, tab_ic, tab_arb = st.tabs(
    ["🔍 Основной скан", "🔥 Высокая IV", "🦅 Iron Condor", "⚖️ Арбитраж"]
)

# ===========================  GENERAL SCAN  =================================

with tab_general:
    st.subheader(f"Основной скан ликвидных опционов {currency}")
    st.caption("Для каждой найденной возможности показываются приблизительные максимальные прибыль/убыток для long и short позиции (на 1 контракт).")
    if st.button("▶ Запустить скан", key="btn_general"):
        with st.spinner("Сканирование... (может занять несколько секунд)"):
            filters = build_filters()
            df = scanner.scan(filters)
        render_results(df, f"general_{currency}", scanner)

# ===========================  HIGH IV  ======================================

with tab_high_iv:
    st.subheader(f"Высокая IV — продажа волатильности ({currency})")
    st.info(
        f"Поиск опционов с IV Rank ≥ **{iv_threshold}%** "
        "(7–60 DTE, IV ≥ 50%, минимальная ликвидность)."
    )
    if st.button("▶ Запустить скан High IV", key="btn_high_iv"):
        iv_threshold_decimal = iv_threshold / 100
        with st.spinner("Сканирование..."):
            df_hiv = scanner.scan_high_iv(currency=currency, iv_threshold=iv_threshold_decimal)
        render_results(df_hiv, f"high_iv_{currency}", scanner)

# ===========================  IRON CONDOR  ==================================

with tab_ic:
    st.subheader(f"Iron Condor сетапы ({currency})")
    st.info("Ищем опционы с 30–45 DTE и умеренной IV (30–70%).")
    if st.button("▶ Запустить скан Iron Condor", key="btn_ic"):
        with st.spinner("Сканирование..."):
            df_ic = scanner.scan_iron_condor_setup(currency=currency)
        render_results(df_ic, f"iron_condor_{currency}", scanner)

# ===========================  ARBITRAGE  ====================================

with tab_arb:
    st.subheader(f"Арбитражные возможности Put-Call Parity ({currency})")
    st.info(
        "Отклонение ≥ 0.5% между синтетическим фьючерсом и спотом — "
        "потенциальная арбитражная возможность."
    )
    if st.button("▶ Запустить скан Арбитраж", key="btn_arb"):
        with st.spinner("Сканирование..."):
            arb_list = scanner.scan_arbitrage_opportunities(currency=currency)

        if not arb_list:
            st.warning("Арбитражных возможностей не найдено.")
        else:
            st.success(f"✅ Найдено возможностей: **{len(arb_list)}**")
            df_arb = pd.DataFrame(arb_list)
            st.dataframe(df_arb.reset_index(drop=True), use_container_width=True)

            fig_arb = px.bar(
                df_arb.head(20),
                x="strike",
                y="arb_pct",
                color="dte",
                title="Топ-20 арбитражных возможностей (% отклонение)",
                labels={"arb_pct": "Отклонение (%)", "strike": "Strike", "dte": "DTE"},
            )
            st.plotly_chart(fig_arb, use_container_width=True)

            csv_arb = df_arb.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Скачать CSV",
                data=csv_arb,
                file_name=f"arbitrage_{currency}.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Данные предоставляются Deribit API. "
    "Публичные эндпоинты работают без API-ключей. "
    "Не является инвестиционной рекомендацией."
)

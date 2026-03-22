import streamlit as st


def apply_dark_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0e1117;
            --panel: #161b22;
            --muted: #8b949e;
            --text: #e6edf3;
            --accent: #3b82f6;
            --warn: #f59e0b;
            --danger: #ef4444;
        }
        .stApp {
            background-color: var(--bg);
            color: var(--text);
        }
        div[data-testid="stMetric"] {
            background: var(--panel);
            border-radius: 10px;
            padding: 0.7rem;
            border: 1px solid rgba(255,255,255,0.06);
        }
        .risk-chip {
            display: inline-block;
            padding: 0.25rem 0.55rem;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 600;
            background: rgba(59,130,246,0.16);
            color: #c7ddff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

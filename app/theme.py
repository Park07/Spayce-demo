"""
Crowd Stress Detector — Theme following DESIGN.md
Warm gray background, amber accent, operator-first.
"""
import streamlit as st


def apply_dark_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

        :root {
            --bg-primary: #1e1e22;
            --bg-surface: #26262b;
            --bg-elevated: #2e2e33;
            --bg-overlay: #36363b;
            --border-default: rgba(255,255,255,0.06);
            --border-emphasis: rgba(255,255,255,0.12);
            --border-strong: rgba(255,255,255,0.20);
            --text-primary: #ececef;
            --text-secondary: #a0a0a8;
            --text-muted: #5c5c66;
            --text-faint: #3a3a42;
            --accent: #e8a443;
            --accent-hover: #d4933a;
            --status-safe: #34d399;
            --status-safe-bg: rgba(52,211,153,0.08);
            --status-warning: #f59e0b;
            --status-warning-bg: rgba(245,158,11,0.08);
            --status-critical: #ef4444;
            --status-critical-bg: rgba(239,68,68,0.10);
            --status-info: #60a5fa;
            --status-info-bg: rgba(96,165,250,0.06);
        }

        /* ─── Global ─── */
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
            font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
        }

        /* ─── Sidebar ─── */
        [data-testid="stSidebar"] {
            background-color: var(--bg-surface) !important;
            border-right: 1px solid var(--border-default) !important;
        }
        [data-testid="stSidebar"] * {
            color: var(--text-primary) !important;
            font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
        }

        /* ─── Sidebar toggle: HUGE, glowing, unmissable ─── */
        [data-testid="collapsedControl"] {
            background: var(--bg-elevated) !important;
            border: 2px solid var(--accent) !important;
            border-radius: 8px !important;
            width: 56px !important;
            height: 56px !important;
            top: 16px !important;
            left: 16px !important;
            z-index: 999999 !important;
            box-shadow: 0 0 24px rgba(232, 164, 67, 0.3) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        [data-testid="collapsedControl"]:hover {
            box-shadow: 0 0 36px rgba(232, 164, 67, 0.5) !important;
            background: var(--bg-overlay) !important;
        }
        [data-testid="collapsedControl"] svg {
            color: var(--accent) !important;
            width: 32px !important;
            height: 32px !important;
        }

        /* ─── Typography ─── */
        h1, .stTitle {
            font-family: 'IBM Plex Sans', sans-serif !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em !important;
            color: var(--text-primary) !important;
            font-size: 20px !important;
        }
        h2 {
            font-family: 'IBM Plex Sans', sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: 0.08em !important;
            color: var(--text-secondary) !important;
            font-size: 12px !important;
            text-transform: uppercase !important;
            border-bottom: 1px solid var(--border-default) !important;
            padding-bottom: 8px !important;
            margin-top: 24px !important;
        }
        h3 {
            font-family: 'IBM Plex Sans', sans-serif !important;
            font-weight: 600 !important;
            color: var(--text-primary) !important;
            font-size: 16px !important;
        }

        /* ─── Metric cards ─── */
        [data-testid="stMetric"] {
            background: var(--bg-surface) !important;
            border: 1px solid var(--border-default) !important;
            border-radius: 8px !important;
            padding: 16px !important;
            transition: border-color 0.2s ease !important;
        }
        [data-testid="stMetric"]:hover {
            border-color: var(--border-emphasis) !important;
        }
        [data-testid="stMetric"] label {
            font-family: 'IBM Plex Sans', sans-serif !important;
            font-weight: 600 !important;
            font-size: 10px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
            color: var(--text-muted) !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-family: 'IBM Plex Mono', monospace !important;
            font-weight: 700 !important;
            font-size: 28px !important;
            color: var(--text-primary) !important;
        }

        /* ─── Alerts ─── */
        .stAlert {
            border-radius: 8px !important;
            border-left-width: 3px !important;
            font-family: 'IBM Plex Sans', sans-serif !important;
            font-size: 13px !important;
            font-weight: 500 !important;
        }

        /* ─── Data tables ─── */
        [data-testid="stDataFrame"] table {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 12px !important;
        }
        [data-testid="stDataFrame"] th {
            background: var(--bg-surface) !important;
            color: var(--text-muted) !important;
            font-family: 'IBM Plex Sans', sans-serif !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            font-size: 10px !important;
            letter-spacing: 0.08em !important;
            padding: 10px 12px !important;
        }
        [data-testid="stDataFrame"] td {
            background: var(--bg-primary) !important;
            color: var(--text-primary) !important;
            border-bottom: 1px solid var(--border-default) !important;
            padding: 8px 12px !important;
        }

        /* ─── Buttons ─── */
        .stButton > button {
            background: var(--bg-elevated) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-emphasis) !important;
            border-radius: 6px !important;
            font-family: 'IBM Plex Sans', sans-serif !important;
            font-weight: 700 !important;
            font-size: 13px !important;
            padding: 10px 20px !important;
            transition: all 0.15s ease !important;
        }
        .stButton > button:hover {
            background: var(--bg-overlay) !important;
            border-color: var(--accent) !important;
        }
        .stButton > button[kind="primary"] {
            background: var(--accent) !important;
            border: 1px solid var(--accent) !important;
            color: #16161a !important;
            font-weight: 700 !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: var(--accent-hover) !important;
            box-shadow: 0 0 20px rgba(232, 164, 67, 0.2) !important;
        }

        /* ─── Download ─── */
        .stDownloadButton > button {
            background: var(--bg-surface) !important;
            color: var(--accent) !important;
            border: 1px solid rgba(232, 164, 67, 0.3) !important;
            border-radius: 6px !important;
            font-family: 'IBM Plex Sans', sans-serif !important;
            font-weight: 600 !important;
        }

        /* ─── File uploader ─── */
        [data-testid="stFileUploader"] section {
            border: 1px dashed var(--border-emphasis) !important;
            border-radius: 8px !important;
            background: var(--bg-surface) !important;
        }

        /* ─── Video ─── */
        video {
            border-radius: 8px !important;
            border: 1px solid var(--border-default) !important;
            box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important;
        }

        /* ─── Expander ─── */
        [data-testid="stExpander"] {
            background: var(--bg-surface) !important;
            border: 1px solid var(--border-default) !important;
            border-radius: 8px !important;
        }
        [data-testid="stExpander"] summary {
            font-family: 'IBM Plex Sans', sans-serif !important;
            font-weight: 600 !important;
            font-size: 13px !important;
        }

        /* ─── Plotly ─── */
        .js-plotly-plot .plotly .main-svg {
            background: transparent !important;
        }

        /* ─── Progress bar ─── */
        [data-testid="stProgress"] > div > div {
            background: linear-gradient(90deg, var(--accent-hover), var(--accent)) !important;
            border-radius: 4px !important;
        }

        /* ─── Radio ─── */
        [data-testid="stRadio"] label {
            font-family: 'IBM Plex Sans', sans-serif !important;
            font-weight: 500 !important;
            font-size: 13px !important;
        }

        /* ─── Divider ─── */
        hr { border-color: var(--border-default) !important; }

        /* ─── Status badges ─── */
        .status-safe {
            color: var(--status-safe); background: var(--status-safe-bg);
            border: 1px solid rgba(52,211,153,0.20); padding: 4px 12px;
            border-radius: 6px; font-family: 'IBM Plex Mono', monospace;
            font-weight: 600; font-size: 11px; letter-spacing: 0.05em;
        }
        .status-caution {
            color: var(--status-warning); background: var(--status-warning-bg);
            border: 1px solid rgba(245,158,11,0.25); padding: 4px 12px;
            border-radius: 6px; font-family: 'IBM Plex Mono', monospace;
            font-weight: 600; font-size: 11px; letter-spacing: 0.05em;
        }
        .status-warning {
            color: var(--status-warning); background: var(--status-warning-bg);
            border: 1px solid rgba(245,158,11,0.25); padding: 4px 12px;
            border-radius: 6px; font-family: 'IBM Plex Mono', monospace;
            font-weight: 600; font-size: 11px; letter-spacing: 0.05em;
        }
        .status-critical {
            color: var(--status-critical); background: var(--status-critical-bg);
            border: 1px solid rgba(239,68,68,0.30); padding: 4px 12px;
            border-radius: 6px; font-family: 'IBM Plex Mono', monospace;
            font-weight: 700; font-size: 11px; letter-spacing: 0.05em;
            animation: pulse-glow 2s ease-in-out infinite;
        }
        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 0 rgba(239,68,68,0); }
            50% { box-shadow: 0 0 12px rgba(239,68,68,0.2); }
        }

        /* ─── Operator action cards ─── */
        .action-immediate {
            background: var(--status-critical-bg); border-left: 3px solid var(--status-critical);
            padding: 12px 16px; border-radius: 0 8px 8px 0; margin-bottom: 8px;
            font-family: 'IBM Plex Sans', sans-serif; font-weight: 700;
            font-size: 13px; color: var(--status-critical);
        }
        .action-preemptive {
            background: var(--status-warning-bg); border-left: 3px solid var(--status-warning);
            padding: 12px 16px; border-radius: 0 8px 8px 0; margin-bottom: 8px;
            font-family: 'IBM Plex Sans', sans-serif; font-weight: 600;
            font-size: 13px; color: var(--status-warning);
        }
        .action-watch {
            background: var(--status-info-bg); border-left: 3px solid var(--status-info);
            padding: 12px 16px; border-radius: 0 8px 8px 0; margin-bottom: 8px;
            font-family: 'IBM Plex Sans', sans-serif; font-weight: 500;
            font-size: 13px; color: var(--status-info);
        }
        .action-clear {
            background: var(--status-safe-bg); border-left: 3px solid var(--status-safe);
            padding: 12px 16px; border-radius: 0 8px 8px 0;
            font-family: 'IBM Plex Sans', sans-serif; font-weight: 500;
            font-size: 13px; color: var(--status-safe);
        }

        /* ─── Zone cards ─── */
        .zone-card {
            background: var(--bg-surface); border: 1px solid var(--border-default);
            border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;
            transition: border-color 0.2s ease;
        }
        .zone-card:hover { border-color: var(--border-emphasis); }
        .zone-card-header {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 10px;
        }
        .zone-card-name {
            font-family: 'IBM Plex Sans', sans-serif; font-weight: 700;
            font-size: 14px; color: var(--text-primary);
        }
        .zone-card-stat {
            font-family: 'IBM Plex Mono', monospace; font-size: 12px;
            color: var(--text-secondary);
        }
        .zone-card-stat strong {
            color: var(--text-primary); font-size: 18px;
        }

        /* ─── Scrollbar ─── */
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: var(--bg-primary); }
        ::-webkit-scrollbar-thumb { background: var(--border-emphasis); border-radius: 3px; }

        /* ─── Caption ─── */
        .stCaption, [data-testid="stCaptionContainer"] {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 11px !important; color: var(--text-muted) !important;
        }

        /* ─── Hide ONLY footer, keep header for sidebar toggle ─── */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* But keep the sidebar toggle visible */
        [data-testid="collapsedControl"] {
            visibility: visible !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
import streamlit as st
import os
import pandas as pd
from data.fetch_mlb import get_mlb_stats, get_ai_hr_predictions
from data.fetch_nba import get_nba_stats
from data.fetch_nfl import get_nfl_stats
from ai.hr_predictor import generate_daily_hr_projections
from utils.player_search import search_player_by_name

# Safe data directory setup
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)

st.set_page_config(page_title="Multi-Sport AI Prop Projection Bot", layout="wide")

st.title("🔮 Multi-Sport AI Prop Projection Dashboard")
sport = st.sidebar.selectbox("Select Sport", ["MLB", "NBA", "NFL"])
refresh = st.sidebar.button("🔄 Refresh Projections")

player_search = st.sidebar.text_input("Search Player")
team_filter = st.sidebar.text_input("Filter by Team (optional)")

def display_player_stats(df):
    st.dataframe(df, use_container_width=True)

if sport == "MLB":
    st.subheader("⚾ MLB Daily AI Projected Home Run Props")

    if refresh:
        generate_daily_hr_projections()

    all_projections = get_ai_hr_predictions()
    if player_search:
        all_projections = search_player_by_name(all_projections, player_search)
    if team_filter:
        all_projections = all_projections[all_projections['team'] == team_filter.upper()]

    display_player_stats(all_projections)

elif sport == "NBA":
    st.subheader("🏀 NBA Daily Player Props")
    df = get_nba_stats()
    if player_search:
        df = search_player_by_name(df, player_search)
    if team_filter:
        df = df[df['team'] == team_filter.upper()]
    display_player_stats(df)

elif sport == "NFL":
    st.subheader("🏈 NFL Daily Player Props")
    df = get_nfl_stats()
    if player_search:
        df = search_player_by_name(df, player_search)
    if team_filter:
        df = df[df['team'] == team_filter.upper()]
    display_player_stats(df)

st.markdown("---")
st.markdown("© 2025 Multi-Sport AI Prop Bot — Powered by Real-Time Data + ML")

import streamlit as st
import pandas as pd
import numpy as np
import random
import altair as alt
import requests
import base64
import json
from openai import OpenAI
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Football Talent Evaluator", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# === Styling ===
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
.viewerBadge_link__1S137, .css-164nlkn, .css-1dp5vir {display: none !important;}
.stApp { background: linear-gradient(to bottom, #5CC6FF, #F0F8FF); font-family: "Segoe UI Emoji"; }
.card {
    background: #FFF; border-radius: 12px; padding: 20px;
    margin: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# === GitHub Utilities ===
def fetch_verified_data_from_github():
    url = f"https://api.github.com/repos/{st.secrets['GITHUB_REPO']}/contents/{st.secrets['GITHUB_FILE']}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        content = base64.b64decode(r.json()["content"])
        return pd.read_csv(pd.compat.StringIO(content.decode()))
    return pd.DataFrame(columns=[
        'Player Name', 'Team Name', 'Age', 'Goals', 'Assists', 'Dribbles',
        'Interceptions', 'xG', 'Passing Accuracy', 'Minutes',
        'Player Asking Price (EUR)', 'Verified'
    ])

def push_verified_data_to_github(df):
    url = f"https://api.github.com/repos/{st.secrets['GITHUB_REPO']}/contents/{st.secrets['GITHUB_FILE']}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    existing = requests.get(url, headers=headers)
    sha = existing.json().get("sha") if existing.status_code == 200 else None
    content = base64.b64encode(df.to_csv(index=False).encode()).decode()
    data = {"message": "Update verified players DB", "content": content, "branch": "main"}
    if sha: data["sha"] = sha
    requests.put(url, headers=headers, data=json.dumps(data))

@st.cache_data(ttl=3600)
def get_cached_github_db():
    return fetch_verified_data_from_github()

@st.cache_resource
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

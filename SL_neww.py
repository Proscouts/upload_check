import streamlit as st
import pandas as pd
import numpy as np
import random
import requests
import base64
import json
from openai import OpenAI

# === Streamlit Config ===
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

# === GitHub Functions ===
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
        'Player Asking Price (EUR)', 'Verified', 'Source'
    ])

def push_verified_data_to_github(df):
    url = f"https://api.github.com/repos/{st.secrets['GITHUB_REPO']}/contents/{st.secrets['GITHUB_FILE']}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    r = requests.get(url, headers=headers)
    sha = r.json().get("sha") if r.status_code == 200 else None
    content = base64.b64encode(df.to_csv(index=False).encode()).decode()
    data = {"message": "Update verified players", "content": content, "branch": "main"}
    if sha: data["sha"] = sha
    requests.put(url, headers=headers, data=json.dumps(data))

@st.cache_data(ttl=3600)
def get_cached_github_db():
    return fetch_verified_data_from_github()

# === GPT Full Stats Verification ===
def verify_player_fully(player):
    prompt = f"""
You are a football scout evaluating whether the following player stats seem realistic based on Transfermarkt, FBref or similar public sources.

Player Info:
Name: {player['Player Name']}
Team: {player['Team Name']}
Age: {player['Age']}
Goals: {player['Goals']}
Assists: {player['Assists']}
xG: {player['xG']}
Interceptions: {player['Interceptions']}
Minutes: {player['Minutes']}
Passing Accuracy: {player['Passing Accuracy']}%
Asking Price (EUR): {player['Player Asking Price (EUR)']}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a strict and smart football scout. No disclaimers."},
                {"role": "user", "content": prompt}
            ]
        )

        result = response.choices[0].message.content.strip().lower()
        counts = sum([
            int(rating.split(":")[1].strip()) >= 7
            for line in result.splitlines()
            if any(stat in line for stat in ['goals', 'assists', 'xg', 'interceptions', 'minutes', 'passing accuracy'])
            for rating in [line]
        ])

        if counts >= 4:
            return "Verified"
        elif counts >= 2:
            return "Partially Verified"
        else:
            return "Unverified"
    except Exception as e:
        return "Unverified - AI Error"

# === Load GitHub DB
player_db = get_cached_github_db()
df = None
st.title("⚽ Football Talent Evaluator")

# === Manual Input Form ===
with st.form("manual_input"):
    name = st.text_input("Player Name")
    team = st.text_input("Team Name", value="Free Agent")
    age = st.number_input("Age", 16, 45)
    goals = st.number_input("Goals", 0, 1000)
    assists = st.number_input("Assists", 0, 1000)
    dribbles = st.number_input("Dribbles", 0, 100)
    interceptions = st.number_input("Interceptions", 0, 100)
    xg = st.number_input("xG", 0.0, 25.0)
    passing = st.number_input("Passing Accuracy (%)", 0.0, 100.0)
    minutes = st.number_input("Minutes Played", 0, 10000)
    price = st.number_input("Player Asking Price (EUR)", 10000, 10000000, step=10000)
    save_button = st.form_submit_button("💾 Save Player")

if save_button:
    df = pd.DataFrame([{
        'Player Name': name, 'Team Name': team, 'Age': age,
        'Goals': goals, 'Assists': assists, 'Dribbles': dribbles,
        'Interceptions': interceptions, 'xG': xg, 'Passing Accuracy': passing,
        'Minutes': minutes, 'Player Asking Price (EUR)': price
    }])
    df['Verified'] = verify_player_fully(df.iloc[0])
    df['Source'] = team
    player_db = pd.concat([player_db, df], ignore_index=True)
    push_verified_data_to_github(player_db)
    st.success("✅ Player verified and saved to GitHub.")

# === Upload CSV Block
uploaded = st.file_uploader("📁 Upload Player CSV", type=["csv"])
if uploaded:
    df_u = pd.read_csv(uploaded)
    df_u.columns = [col.strip().lower() for col in df_u.columns]
    column_map = {
        'player name': 'Player Name', 'team name': 'Team Name', 'age': 'Age',
        'goals': 'Goals', 'assists': 'Assists', 'dribbles': 'Dribbles',
        'interceptions': 'Interceptions', 'xg': 'xG',
        'passing accuracy': 'Passing Accuracy', 'minutes': 'Minutes',
        'player asking price (eur)': 'Player Asking Price (EUR)'
    }
    df_u.rename(columns=column_map, inplace=True)

    if 'Player Asking Price (EUR)' not in df_u.columns:
        st.error("❌ Missing 'Player Asking Price (EUR)' column.")
    else:
        updated = False
        for _, row in df_u.iterrows():
            pname = row['Player Name']
            if pname not in player_db['Player Name'].values:
                row['Verified'] = verify_player_fully(row)
                row['Source'] = row['Team Name']
                player_db = pd.concat([player_db, pd.DataFrame([row])], ignore_index=True)
                updated = True
        if updated:
            push_verified_data_to_github(player_db)
            st.success("✅ Upload complete with AI verification.")
        else:
            st.info("ℹ️ All players already exist in the database.")

# === Search & View
st.subheader("🔍 Search Player from Database")
search_name = st.text_input("Enter player name to search:")
if search_name:
    filtered = player_db[player_db['Player Name'].str.contains(search_name, case=False, na=False)]
    st.dataframe(filtered, height=200) if not filtered.empty else st.warning("No player found.")

# === Final GitHub-Only Database View
st.subheader("📊 All Verified Players from GitHub")
st.dataframe(player_db.sort_values("Player Name").reset_index(drop=True), height=300)

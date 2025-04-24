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

# ========== CONFIG ==========
st.set_page_config(page_title="Football Talent Evaluator", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ========== STYLE ==========
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
.viewerBadge_link__1S137, .css-164nlkn, .css-1dp5vir {display: none !important;}
.stApp {
    background: linear-gradient(to bottom, #5CC6FF, #F0F8FF);
    font-family: "Segoe UI Emoji", "Apple Color Emoji";
}
.card {
    background: #FFF; border-radius: 12px; padding: 20px;
    margin: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ========== GITHUB HELPERS ==========
def fetch_verified_data_from_github():
    url = f"https://api.github.com/repos/{st.secrets['GITHUB_REPO']}/contents/{st.secrets['GITHUB_FILE']}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = base64.b64decode(response.json()["content"])
        return pd.read_csv(pd.compat.StringIO(content.decode()))
    return pd.DataFrame(columns=[
        'Player Name', 'Team Name', 'Age', 'Goals', 'Assists', 'Dribbles',
        'Interceptions', 'xG', 'Passing Accuracy', 'Minutes', 'Market Value (EUR)',
        'Verified'
    ])

def push_verified_data_to_github(df):
    url = f"https://api.github.com/repos/{st.secrets['GITHUB_REPO']}/contents/{st.secrets['GITHUB_FILE']}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    get_resp = requests.get(url, headers=headers)
    sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None

    content = base64.b64encode(df.to_csv(index=False).encode()).decode()
    data = {
        "message": "Update verified players DB",
        "content": content,
        "branch": "main"
    }
    if sha:
        data["sha"] = sha

    push_resp = requests.put(url, headers=headers, data=json.dumps(data))
    return push_resp.status_code in [200, 201]

# ========== GPT VERIFICATION ==========
def verify_with_openai(player_name, market_value):
    prompt = (
        f"What is the approximate current market value of football player {player_name}? "
        f"Compare with {market_value} EUR and assess if it seems realistic."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a football transfer analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except:
        return "Verification unavailable"

# ========== PREPARE PLAYER DATA ==========
def prepare_data(df):
    df = df.rename(columns={
        'player_name': 'Player Name', 'team_name': 'Team Name',
        'player_match_goals': 'Goals', 'player_match_assists': 'Assists',
        'player_match_dribbles': 'Dribbles', 'player_match_interceptions': 'Interceptions',
        'player_match_np_xg': 'xG', 'player_match_passing_ratio': 'Passing Accuracy',
        'player_match_minutes': 'Minutes'
    })
    df['Market_Value_SAR'] = df['Market Value (EUR)'] * 3.75
    df['Asking_Price_SAR'] = df['Market_Value_SAR'] * np.random.uniform(1.05, 1.2, size=len(df))
    df['Image'] = df['Player Name'].apply(lambda n: f"https://robohash.org/{n.replace(' ', '')}.png?set=set2")
    df['Best_Fit_Club'] = df['Team Name'].apply(lambda _: random.choice(['Man United', 'PSG', 'Al Hilal']))
    df['Nationality'] = "Egyptian"
    df['Position'] = random.choices(['Forward', 'Midfielder', 'Defender'], k=len(df))
    df['Transfer_Chance'] = df['Market_Value_SAR'].apply(lambda x: random.uniform(0.6, 0.95))

    # Train quick regressor
    features = ['xG', 'Assists', 'Goals', 'Dribbles', 'Interceptions', 'Passing Accuracy', 'Market_Value_SAR']
    X = df[features]
    y = df['Market_Value_SAR'] * random.uniform(1.05, 1.15)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    df['Predicted_Year_1'] = model.predict(X)
    df['Predicted_Year_2'] = df['Predicted_Year_1'] * 1.05
    df['Predicted_Year_3'] = df['Predicted_Year_2'] * 1.05

    return df

# ========== MAIN ==========
st.title("⚽ Football Talent Evaluator")
st.markdown("Upload a player CSV or manually evaluate one:")

player_db = fetch_verified_data_from_github()
df = None

# === Manual Form
with st.form("manual_input"):
    name = st.text_input("Player Name")
    team = st.text_input("Team Name")
    age = st.number_input("Age", 16, 45)
    goals = st.number_input("Goals", 0, 10)
    assists = st.number_input("Assists", 0, 10)
    dribbles = st.number_input("Dribbles", 0, 20)
    interceptions = st.number_input("Interceptions", 0, 10)
    xg = st.number_input("xG", 0.0, 5.0)
    passing = st.number_input("Passing Accuracy (%)", 0.0, 100.0)
    minutes = st.number_input("Minutes Played", 0, 120)
    market_val = st.number_input("Market Value (EUR)", 10000, 10000000, step=10000)
    submitted = st.form_submit_button("Evaluate Player")

if submitted:
    manual_df = pd.DataFrame([{
        'player_name': name, 'team_name': team, 'age': age,
        'player_match_goals': goals, 'player_match_assists': assists,
        'player_match_dribbles': dribbles, 'player_match_interceptions': interceptions,
        'player_match_np_xg': xg, 'player_match_passing_ratio': passing,
        'player_match_minutes': minutes, 'Market Value (EUR)': market_val
    }])
    df = prepare_data(manual_df)
    verified = verify_with_openai(name, market_val)

    # Update GitHub DB
    row = {
        'Player Name': name, 'Team Name': team, 'Age': age, 'Goals': goals,
        'Assists': assists, 'Dribbles': dribbles, 'Interceptions': interceptions,
        'xG': xg, 'Passing Accuracy': passing, 'Minutes': minutes,
        'Market Value (EUR)': market_val, 'Verified': verified
    }
    player_db = player_db[player_db['Player Name'] != name]
    player_db = pd.concat([player_db, pd.DataFrame([row])], ignore_index=True)
    push_verified_data_to_github(player_db)

# === File Upload
uploaded_file = st.file_uploader("📁 Or upload CSV", type=["csv"])
if uploaded_file and not submitted:
    raw_df = pd.read_csv(uploaded_file)
    df = prepare_data(raw_df)

    for _, row in raw_df.iterrows():
        verified = verify_with_openai(row['player_name'], row['Market_Value_EUR'])
        entry = {
            'Player Name': row['player_name'], 'Team Name': row['team_name'], 'Age': row['age'],
            'Goals': row['player_match_goals'], 'Assists': row['player_match_assists'],
            'Dribbles': row['player_match_dribbles'], 'Interceptions': row['player_match_interceptions'],
            'xG': row['player_match_np_xg'], 'Passing Accuracy': row['player_match_passing_ratio'],
            'Minutes': row['player_match_minutes'], 'Market Value (EUR)': row['Market_Value_EUR'],
            'Verified': verified
        }
        player_db = player_db[player_db['Player Name'] != row['player_name']]
        player_db = pd.concat([player_db, pd.DataFrame([entry])], ignore_index=True)
    push_verified_data_to_github(player_db)
    st.success("✅ Uploaded and verified")

# === Display Player Card
if df is not None and not df.empty:
    player = df.iloc[0]
    st.markdown(f"""
    <div class='card'>
        <h3>{player['Player Name']} ({player['Position']})</h3>
        <img src="{player['Image']}" width="100">
        <p><strong>Club:</strong> {player['Team Name']}</p>
        <p><strong>Age:</strong> {player['age']}</p>
        <p><strong>Market Value:</strong> €{player['Market Value (EUR)']:,.0f}</p>
        <p><strong>Verified Info:</strong> {player_db[player_db['Player Name'] == player['Player Name']]['Verified'].values[0]}</p>
        <p><strong>Best Fit Club:</strong> {player['Best_Fit_Club']}</p>
    </div>
    """, unsafe_allow_html=True)

    forecast_df = pd.DataFrame({
        "Year": ["2024", "2025", "2026"],
        "Predicted Value (SAR)": [player['Predicted_Year_1'], player['Predicted_Year_2'], player['Predicted_Year_3']],
        "Club Asking Price (SAR)": [player['Asking_Price_SAR']] * 3
    })
    chart = alt.Chart(forecast_df).transform_fold(
        ['Predicted Value (SAR)', 'Club Asking Price (SAR)'],
        as_=['Metric', 'SAR Value']
    ).mark_line(point=True).encode(
        x='Year:N', y='SAR Value:Q', color='Metric:N'
    )
    st.altair_chart(chart, use_container_width=True)

    # === AI Summary
    st.header("🧠 AI Summary")
    comment = f"{player['Player Name']} is showing promise with consistent performance."
    try:
        summary = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": f"Summarize this: {comment}"}]
        ).choices[0].message.content
    except:
        summary = "AI summary unavailable"

    st.markdown(f"<div class='card'><p>{summary}</p></div>", unsafe_allow_html=True)

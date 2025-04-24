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
    data = {
        "message": "Update verified players DB",
        "content": content,
        "branch": "main"
    }
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

# === GPT Verification ===
def verify_with_openai(player_name, asking_price):
    prompt = (
        f"You are a football scout with access to Transfermarkt. "
        f"Evaluate player {player_name} with asking price {asking_price} EUR. "
        f"Cite source if possible. No disclaimers."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "Be concise. No disclaimers. Assume real-time data access."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except:
        return "Not Verified"

# === Prepare Data ===
def prepare_data(df):
    df['Player Asking Price (EUR)'] = df['Player Asking Price (EUR)'].astype(float)
    df['Market_Value_SAR'] = df['Player Asking Price (EUR)'] * 3.75
    df['Asking_Price_SAR'] = df['Market_Value_SAR'] * np.random.uniform(1.05, 1.2, size=len(df))
    df['Image'] = df['Player Name'].apply(lambda n: f"https://robohash.org/{n.replace(' ', '')}.png?set=set2")
    df['Best_Fit_Club'] = df['Team Name'].apply(lambda _: random.choice(['Man United', 'PSG', 'Al Hilal']))
    df['Position'] = random.choices(['Forward', 'Midfielder', 'Defender'], k=len(df))
    df['Transfer_Chance'] = df['Market_Value_SAR'].apply(lambda x: random.uniform(0.6, 0.95))

    features = ['xG', 'Assists', 'Goals', 'Dribbles', 'Interceptions', 'Passing Accuracy', 'Market_Value_SAR']
    X = df[features]
    y = df['Market_Value_SAR'] * random.uniform(1.05, 1.15)
    model = train_model(X, y)
    df['Predicted_Year_1'] = model.predict(X)
    df['Predicted_Year_2'] = df['Predicted_Year_1'] * 1.05
    df['Predicted_Year_3'] = df['Predicted_Year_2'] * 1.05
    return df

# === Load GitHub DB ===
player_db = get_cached_github_db()
df = None

st.title("⚽ Football Talent Evaluator")

# === Manual Form ===
with st.form("manual_input"):
    name = st.text_input("Player Name")
    team = st.text_input("Team Name", value="No Club")
    age = st.number_input("Age", 16, 45)
    goals = st.number_input("Goals", 0, 10)
    assists = st.number_input("Assists", 0, 10)
    dribbles = st.number_input("Dribbles", 0, 20)
    interceptions = st.number_input("Interceptions", 0, 10)
    xg = st.number_input("xG", 0.0, 5.0)
    passing = st.number_input("Passing Accuracy (%)", 0.0, 100.0)
    minutes = st.number_input("Minutes Played", 0, 120)
    asking_price = st.number_input("Player Asking Price (EUR)", 10000, 10000000, step=10000)
    submitted = st.form_submit_button("Evaluate Player")

if submitted:
    df = pd.DataFrame([{
        'Player Name': name, 'Team Name': team, 'Age': age,
        'Goals': goals, 'Assists': assists, 'Dribbles': dribbles,
        'Interceptions': interceptions, 'xG': xg, 'Passing Accuracy': passing,
        'Minutes': minutes, 'Player Asking Price (EUR)': asking_price
    }])
    df = prepare_data(df)

    if name not in player_db['Player Name'].values:
        verified = verify_with_openai(name, asking_price)
        df['Verified'] = verified
        player_db = pd.concat([player_db, df], ignore_index=True)
        push_verified_data_to_github(player_db)

# === Upload CSV ===
uploaded_file = st.file_uploader("📁 Upload Player CSV", type=["csv"])
if uploaded_file:
    uploaded_df = pd.read_csv(uploaded_file)
    uploaded_df.columns = uploaded_df.columns.str.strip().str.replace('_', ' ').str.title()

    if 'Market Value (Eur)' in uploaded_df.columns and 'Player Asking Price (Eur)' not in uploaded_df.columns:
        uploaded_df['Player Asking Price (EUR)'] = uploaded_df['Market Value (Eur)']

    if 'Player Asking Price (EUR)' not in uploaded_df.columns:
        st.error("❌ 'Player Asking Price (EUR)' column is missing.")
    else:
        st.write("🔍 Columns:", uploaded_df.columns.tolist())
        df = prepare_data(uploaded_df)
        updated = False
        for _, row in uploaded_df.iterrows():
            if row['Player Name'] not in player_db['Player Name'].values:
                verified = verify_with_openai(row['Player Name'], row['Player Asking Price (EUR)'])
                row['Verified'] = verified
                player_db = pd.concat([player_db, pd.DataFrame([row])], ignore_index=True)
                updated = True
        if updated:
            push_verified_data_to_github(player_db)
            st.success("✅ Players uploaded and added to database.")
        else:
            st.info("ℹ️ All players already exist in database.")

# === Display Profile ===
if df is not None and not df.empty:
    player = df.iloc[0]
    verified_info = "Not Verified"
    record = player_db[player_db['Player Name'] == player['Player Name']]
    if not record.empty:
        verified_info = record['Verified'].values[0]

    st.markdown(f"""
    <div class='card'>
        <h3>{player['Player Name']} ({player['Position']})</h3>
        <img src="{player['Image']}" width="100">
        <p><strong>Team:</strong> {player['Team Name']}</p>
        <p><strong>Age:</strong> {player['Age']}</p>
        <p><strong>Asking Price:</strong> €{player['Player Asking Price (EUR)']:,.0f}</p>
        <p><strong>Verified Info:</strong> {verified_info}</p>
        <p><strong>Best Fit Club:</strong> {player['Best_Fit_Club']}</p>
    </div>
    """, unsafe_allow_html=True)

    chart_df = pd.DataFrame({
        "Year": ["2024", "2025", "2026"],
        "Predicted Value (SAR)": [player['Predicted_Year_1'], player['Predicted_Year_2'], player['Predicted_Year_3']],
        "Club Asking Price (SAR)": [player['Asking_Price_SAR']] * 3
    })
    st.altair_chart(
        alt.Chart(chart_df).transform_fold(
            ['Predicted Value (SAR)', 'Club Asking Price (SAR)'],
            as_=['Metric', 'SAR Value']
        ).mark_line(point=True).encode(
            x='Year:N', y='SAR Value:Q', color='Metric:N'
        ), use_container_width=True
    )

    st.header("🧠 AI Summary")
    try:
        comment = f"{player['Player Name']} has strong development potential."
        summary = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": f"Summarize this profile: {comment}"}]
        ).choices[0].message.content
    except:
        summary = "AI summary unavailable."
    st.markdown(f"<div class='card'><p>{summary}</p></div>", unsafe_allow_html=True)

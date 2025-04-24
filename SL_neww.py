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

# === GitHub Utils ===
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
    data = {"message": "Update DB", "content": content, "branch": "main"}
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

def verify_with_openai(player_name, asking_price):
    prompt = (
        f"Evaluate {player_name}'s market value based on their asking price {asking_price} EUR. "
        f"Use known sources (e.g., Transfermarkt). Do not add disclaimers."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a confident football scout with data access."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except:
        return "Not Verified"

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

player_db = get_cached_github_db()
df = None
st.title("⚽ Football Talent Evaluator")

# === Manual Form ===
with st.form("manual_input"):
    name = st.text_input("Player Name")
    team = st.text_input("Team Name", value="Free Agent")
    age = st.number_input("Age", 16, 45)
    goals = st.number_input("Goals", 0, 10)
    assists = st.number_input("Assists", 0, 10)
    dribbles = st.number_input("Dribbles", 0, 20)
    interceptions = st.number_input("Interceptions", 0, 10)
    xg = st.number_input("xG", 0.0, 5.0)
    passing = st.number_input("Passing Accuracy (%)", 0.0, 100.0)
    minutes = st.number_input("Minutes Played", 0, 120)
    price = st.number_input("Player Asking Price (EUR)", 10000, 10000000, step=10000)
    submitted = st.form_submit_button("Evaluate Player")

if submitted:
    df = pd.DataFrame([{
        'Player Name': name, 'Team Name': team, 'Age': age,
        'Goals': goals, 'Assists': assists, 'Dribbles': dribbles,
        'Interceptions': interceptions, 'xG': xg, 'Passing Accuracy': passing,
        'Minutes': minutes, 'Player Asking Price (EUR)': price
    }])
    df = prepare_data(df)
    verified = verify_with_openai(name, price)
    df['Verified'] = verified
    player_db = pd.concat([player_db, df], ignore_index=True)
    push_verified_data_to_github(player_db)

# === Upload Section ===
uploaded = st.file_uploader("📁 Upload CSV", type=["csv"])
if uploaded:
    df_u = pd.read_csv(uploaded)
    df_u.columns = df_u.columns.str.strip().str.replace('_', ' ').str.title()
    if 'Player Asking Price (EUR)' not in df_u.columns:
        st.error("CSV must contain 'Player Asking Price (EUR)' column.")
    else:
        df_u = prepare_data(df_u)
        for _, row in df_u.iterrows():
            pname = row['Player Name']
            if pname not in player_db['Player Name'].values:
                verified = verify_with_openai(pname, row['Player Asking Price (EUR)'])
                row['Verified'] = verified
                player_db = pd.concat([player_db, pd.DataFrame([row])], ignore_index=True)
        push_verified_data_to_github(player_db)
        st.success("✅ Upload successful!")

# === Show Profile (if any) ===
if df is not None and not df.empty:
    player = df.iloc[0]
    st.markdown(f"""
    <div class='card'>
        <h3>{player['Player Name']} ({player['Position']})</h3>
        <img src="{player['Image']}" width="100">
        <p><strong>Club:</strong> {player['Team Name']}</p>
        <p><strong>Asking Price:</strong> €{player['Player Asking Price (EUR)']:,.0f}</p>
        <p><strong>Verified:</strong> {player['Verified']}</p>
        <p><strong>Best Fit:</strong> {player['Best_Fit_Club']}</p>
    </div>
    """, unsafe_allow_html=True)

    forecast = pd.DataFrame({
        "Year": ["2024", "2025", "2026"],
        "Predicted Value (SAR)": [player['Predicted_Year_1'], player['Predicted_Year_2'], player['Predicted_Year_3']],
        "Asking Price (SAR)": [player['Asking_Price_SAR']] * 3
    })
    st.altair_chart(
        alt.Chart(forecast).transform_fold(
            ['Predicted Value (SAR)', 'Asking Price (SAR)'],
            as_=['Metric', 'SAR Value']
        ).mark_line(point=True).encode(
            x='Year:N', y='SAR Value:Q', color='Metric:N'
        ), use_container_width=True
    )

# === Export + Search + Show All ===
st.subheader("📥 Download Verified Database")
st.download_button("Download CSV", player_db.to_csv(index=False), "Verified_Player_DB.csv", "text/csv")

st.subheader("🔍 Search Player")
q = st.text_input("Enter name")
if q:
    results = player_db[player_db['Player Name'].str.contains(q, case=False, na=False)]
    st.dataframe(results if not results.empty else "No match found.")

st.subheader("📊 All Verified Players")
st.dataframe(player_db.sort_values(by="Player Name").reset_index(drop=True))

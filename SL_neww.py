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

# === Streamlit config ===
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
        'Player Asking Price (EUR)', 'Verified'
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

@st.cache_resource
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# === GPT with Source + Fallback ===
def verify_with_openai(player_name, asking_price):
    prompt = (
        f"Evaluate the market value of {player_name}. The asking price is {asking_price} EUR. "
        f"Use known sources like Transfermarkt or FBref. If it can't be verified, reply 'Unverified'."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a confident scout. Never write disclaimers."},
                {"role": "user", "content": prompt}
            ]
        )
        result = response.choices[0].message.content.strip()
        return result  # Always return the full AI output instead of just "Unverified"
    except:
        return "Unverified"
# === Prepare Player Data ===
def prepare_data(df):
    df['Player Asking Price (EUR)'] = df['Player Asking Price (EUR)'].astype(float)
    df['Market_Value_SAR'] = df['Player Asking Price (EUR)'] * 3.75
    df['Asking_Price_SAR'] = df['Market_Value_SAR'] * np.random.uniform(1.05, 1.2, size=len(df))
    df['Image'] = df['Player Name'].apply(lambda n: f"https://robohash.org/{n.replace(' ', '')}.png?set=set2")
    df['Best_Fit_Club'] = df['Team Name'].apply(lambda _: random.choice(['Man United', 'PSG', 'Al Hilal']))
    df['Position'] = random.choices(['Forward', 'Midfielder', 'Defender'], k=len(df))
    df['Transfer_Chance'] = df['Market_Value_SAR'].apply(lambda x: random.uniform(0.6, 0.95))

    # Required columns for model
    features = ['xG', 'Assists', 'Goals', 'Dribbles', 'Interceptions', 'Passing Accuracy', 'Market_Value_SAR']
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.stop()

    X = df[features]
    y = df['Market_Value_SAR'] * random.uniform(1.05, 1.15)
    model = train_model(X, y)
    df['Predicted_Year_1'] = model.predict(X)
    df['Predicted_Year_2'] = df['Predicted_Year_1'] * 1.05
    df['Predicted_Year_3'] = df['Predicted_Year_2'] * 1.05
    return df


# === Load GitHub DB
player_db = get_cached_github_db()
df = None
st.title("⚽ Football Talent Evaluator")

# === Manual Entry Form ===
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

# === Upload CSV Block ===
# === Upload CSV Block ===
uploaded = st.file_uploader("📁 Upload Player CSV", type=["csv"])
if uploaded:
    df_u = pd.read_csv(uploaded)

    # ✅ FIX: Normalize and match expected column names
    column_map = {
        'player name': 'Player Name',
        'team name': 'Team Name',
        'age': 'Age',
        'goals': 'Goals',
        'assists': 'Assists',
        'dribbles': 'Dribbles',
        'interceptions': 'Interceptions',
        'xg': 'xG',
        'passing accuracy': 'Passing Accuracy',
        'minutes': 'Minutes',
        'player asking price (eur)': 'Player Asking Price (EUR)'
    }
    df_u.columns = [col.strip().lower() for col in df_u.columns]
    df_u.rename(columns=column_map, inplace=True)

    # Continue with price column check and prepare_data
    # Try to fix weird formats of "asking price"
    matches = [col for col in df_u.columns if 'asking price' in col.lower()]
    if matches:
        df_u.rename(columns={matches[0]: 'Player Asking Price (EUR)'}, inplace=True)

    # Final check
    if 'Player Asking Price (EUR)' not in df_u.columns:
        st.error("❌ Could not detect 'Player Asking Price (EUR)' column.")
        st.write("🧪 Columns found:", df_u.columns.tolist())
    else:
        df_u = prepare_data(df_u)
        updated = False
        for _, row in df_u.iterrows():
            pname = row['Player Name']
            if pname not in player_db['Player Name'].values:
                # GPT verification with fallback
                try:
                    verified = verify_with_openai(pname, row['Player Asking Price (EUR)'])
                    if "unverified" in verified.lower() or "no access" in verified.lower():
                        verified = "Unverified"
                except:
                    verified = "Unverified"

                row['Verified'] = verified
                player_db = pd.concat([player_db, pd.DataFrame([row])], ignore_index=True)
                updated = True
        if updated:
            push_verified_data_to_github(player_db)
            st.success("✅ Upload completed and players added.")
        else:
            st.info("ℹ️ No new players added — all are already in the database.")

# === Forecast & Card ===
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

    # === 🧠 AI Commentary Block ===
    st.markdown("### 🧠 Player Attitude Summary (AI-Generated)")
    comment = f"{player['Player Name']} has been showing impactful performances with key contributions in recent matches."

    sentiment_prompt = [
        {"role": "system", "content": "You are a football sentiment expert. Classify the tone of the comment as Positive, Neutral, or Negative."},
        {"role": "user", "content": comment}
    ]
    summary_prompt = [
        {"role": "system", "content": "You're a football scout. Write a one-line summary of this player's recent performance."},
        {"role": "user", "content": comment}
    ]

    def get_icon(text):
        return ("🔴", "Negative") if "negative" in text.lower() else (
               ("🟢", "Positive") if "positive" in text.lower() else
               ("🟡", "Neutral") if "neutral" in text.lower() else ("⚪", "Unclear"))

    try:
        sentiment_response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=sentiment_prompt
        )
        sentiment_text = sentiment_response.choices[0].message.content.strip()
        emoji, mood = get_icon(sentiment_text)

        summary_response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=summary_prompt
        )
        summary_text = summary_response.choices[0].message.content.strip()
    except Exception as e:
        emoji, mood, sentiment_text, summary_text = "⚪", "Unavailable", "Unable to classify sentiment.", "Summary not available."
        st.warning(f"⚠️ AI fallback triggered: {e}")

    st.markdown(f'''
    <div class='card'>
        <h4>Comment Context:</h4>
        <p><em>{comment}</em></p>
        <h4>Sentiment:</h4>
        <p><strong>{emoji} {mood}</strong> – {sentiment_text}</p>
        <h4>Scout Summary:</h4>
        <p>{summary_text}</p>
    </div>
    ''', unsafe_allow_html=True)
# === 📥 Download Verified Player DB
st.subheader("📥 Download Verified Player Database")
st.download_button(
    label="Download CSV",
    data=player_db.to_csv(index=False),
    file_name="Verified_Player_Database.csv",
    mime="text/csv"
)

# === 🔍 Search Player
st.subheader("🔍 Search Player from Database")
search_name = st.text_input("Enter player name to search:")
if search_name:
    filtered_players = player_db[player_db['Player Name'].str.contains(search_name, case=False, na=False)]
    if not filtered_players.empty:
        st.success(f"✅ Found {len(filtered_players)} result(s).")
        st.dataframe(filtered_players)
    else:
        st.warning("No player found with that name.")

# === 📊 Display All Verified Players
st.subheader("📊 All Verified Players from GitHub Database")
st.dataframe(player_db.sort_values("Player Name").reset_index(drop=True))

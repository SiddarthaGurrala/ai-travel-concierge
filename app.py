import datetime
import os
import sqlite3
import requests
import streamlit as st
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# ---------------------------------------------------------
# Page & UI Setup
# ---------------------------------------------------------
st.set_page_config(page_title="AI Travel Concierge", page_icon="✈️", layout="wide")
st.title("✈️ AI Travel Concierge")
st.caption("Track A: Week 6 Specialization — Tool-Calling, Itinerary Generation & SQLite Storage")

# ---------------------------------------------------------
# SQLite Database Setup (Persistent Storage Requirement)
# ---------------------------------------------------------
DB_FILE = "travel_concierge.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            destination TEXT,
            response TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_search(destination: str, response: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT INTO searches (timestamp, destination, response) VALUES (?, ?, ?)",
        (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), destination, response)
    )
    conn.commit()
    conn.close()

def get_recent_searches(limit: int = 5):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT timestamp, destination, response FROM searches ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

init_db()

# ---------------------------------------------------------
# Sidebar Configuration & Secure Key Handling
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Priority: Streamlit secrets/env first, manual input fallback
    default_key = os.getenv("GOOGLE_API_KEY", "")
    api_key = st.text_input("Gemini API Key", value=default_key, type="password", help="Stored securely in memory or secrets.")
    
    st.markdown("---")
    st.subheader("🛠️ Active Integrations")
    st.markdown("- ☀️ **Open-Meteo Weather API**")
    st.markdown("- 🔍 **DuckDuckGo Web Search**")
    st.markdown("- ✈️ **Travel & Flight Guidance Tool**")
    st.markdown("- 💾 **SQLite Persistent Database**")
    
    st.markdown("---")
    st.subheader("🕒 Saved Search History")
    history_entries = get_recent_searches()
    if history_entries:
        for t_stamp, dest, resp in history_entries:
            with st.expander(f"{dest} ({t_stamp[:10]})"):
                st.caption(f"Recorded at: {t_stamp}")
                st.write(resp[:250] + "..." if len(resp) > 250 else resp)
    else:
        st.info("No saved itineraries yet.")

# ---------------------------------------------------------
# Tools Definition
# ---------------------------------------------------------
@tool
def get_current_weather(city: str) -> str:
    """Fetch real-time weather and temperature for any city using the Open-Meteo API."""
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url, timeout=5).json()
        if not geo_res.get("results"):
            return f"Could not find coordinates for '{city}'."
        
        loc = geo_res["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        w_res = requests.get(weather_url, timeout=5).json()
        curr = w_res.get("current_weather", {})
        temp = curr.get("temperature")
        wind = curr.get("windspeed")
        return f"Current weather in {loc['name']}, {loc.get('country', '')}: {temp}°C, Wind Speed: {wind} km/h."
    except Exception as e:
        return f"Failed to retrieve weather data: {str(e)}"

@tool
def get_travel_transit_info(origin: str, destination: str) -> str:
    """Fetch transit, standard airline hubs, and route planning guidance between an origin and destination."""
    return (
        f"Route Analysis ({origin} -> {destination}): "
        f"Primary connectivity is served through major international hub connections. "
        f"Recommended travel window: Advance booking 4-6 weeks prior for optimal fares. "
        f"Standard transit includes direct or single-stop connecting routes depending on carrier alliances."
    )

search_tool = DuckDuckGoSearchRun()
tools = [get_current_weather, search_tool, get_travel_transit_info]

# ---------------------------------------------------------
# Chat Interface & Agent Execution
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am your AI Travel Concierge. "
                "I can build custom multi-day itineraries, check live weather, search web travel advice, "
                "and archive your travel plans automatically!"
            )
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_prompt := st.chat_input("Ask for a trip plan (e.g., 'Plan a 3-day budget itinerary for Rome with current weather')"):
    if not api_key:
        st.warning("Please provide a Gemini API Key in the sidebar to proceed.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=api_key,
            temperature=0.3
        )
        
        system_instruction = (
            "You are an expert AI Travel Concierge. When users ask for a travel plan, provide a clear, "
            "actionable, day-by-day itinerary. Use the weather tool to check destination climate, the search tool "
            "for current tips, and include lodging and dining recommendations."
        )
        
        agent_executor = create_react_agent(llm, tools, prompt=system_instruction)

        agent_inputs = [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in st.session_state.messages
        ]

        with st.chat_message("assistant"):
            with st.spinner("Compiling live data, itinerary, and travel specifics..."):
                response = agent_executor.invoke({"messages": agent_inputs})
                final_answer = response["messages"][-1].content
                st.markdown(final_answer)

        st.session_state.messages.append({"role": "assistant", "content": final_answer})

        # Save to SQLite database
        dest_summary = user_prompt[:30].replace("'", "")
        save_search(dest_summary, final_answer)
        st.rerun()

    except Exception as err:
        st.error(f"Error processing your request: {err}")
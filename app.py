import streamlit as st
import pandas as pd
import requests
import difflib

# -----------------------------
# API URLs
# -----------------------------
RAIN_URL = "https://api.data.gov.in/resource/6c05cd1b-ed59-40c2-bc31-e314f39c6971?api-key=579b464db66ec23bdd0000015941db76bcec4b074da1a1ec1cd8d90e&format=json&limit=10000"
CROP_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key=579b464db66ec23bdd0000015941db76bcec4b074da1a1ec1cd8d90e&format=json&limit=10000"
MSP_URL = "https://api.data.gov.in/resource/14389871-c2f4-4348-b4ca-b55391d4ea0b?api-key=579b464db66ec23bdd0000015941db76bcec4b074da1a1ec1cd8d90e&format=json&limit=10000"

# -----------------------------
# Fetch Data Helper
# -----------------------------
def fetch_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "records" in data:
            return pd.DataFrame(data["records"])
        else:
            st.error("No 'records' key found in API response.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# -----------------------------
# Cached Data Loaders
# -----------------------------
@st.cache_data
def load_crop_data():
    df = fetch_data(CROP_URL)
    df.columns = df.columns.str.strip().str.lower()
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.strip()
    return df

@st.cache_data
def load_rainfall_data():
    all_data = []
    base_url = RAIN_URL + "&offset="
    offset = 0
    while True:
        response = requests.get(base_url + str(offset))
        data = response.json()
        if "records" not in data or not data["records"]:
            break
        all_data.extend(data["records"])
        offset += len(data["records"])
        if len(data["records"]) < 1000:
            break
    df = pd.DataFrame(all_data)
    df.columns = df.columns.str.strip().str.lower()
    return df

@st.cache_data
def load_msp_data():
    df = fetch_data(MSP_URL)
    df.columns = df.columns.str.strip().str.lower()
    return df

# -----------------------------
# Load Datasets
# -----------------------------
rainfall_df = load_rainfall_data()
crop_df = load_crop_data()
msp_df = load_msp_data()

# -----------------------------
# Helper Functions
# -----------------------------
def extract_state_from_question(question):
    states = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana",
        "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
        "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
        "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands"
    ]
    matches = [state for state in states if state.lower() in question.lower()]
    return matches

def extract_crop_from_question(question):
    question = question.lower()
    known_crops = [
        "rice", "wheat", "maize", "cotton", "sugarcane", "barley", "jowar",
        "bajra", "ragi", "pulses", "tur", "urad", "moong", "gram",
        "mustard", "groundnut", "soybean", "potato", "onion", "banana",
        "mango", "apple", "coffee", "tea"
    ]
    crops_found = [crop for crop in known_crops if crop in question]
    return crops_found if crops_found else None

# -----------------------------
# Crop Production
# -----------------------------
def answer_crop_production_question(question):
    try:
        states = extract_state_from_question(question)
        crops = extract_crop_from_question(question)

        if not states:
            return {"success": False, "message": "Please specify a state to check crop market data."}

        df = crop_df.copy()
        if "state" not in df.columns or "commodity" not in df.columns:
            return {"success": False, "message": "Required columns (state or commodity) not found in dataset."}

        df["state"] = df["state"].astype(str).str.strip().str.lower()
        df["commodity"] = df["commodity"].astype(str).str.strip().str.lower()

        state_matches = [s.lower() for s in states]
        crop_matches = [c.lower() for c in crops] if crops else []

        df_state = df[df["state"].isin(state_matches)]
        if df_state.empty:
            return {"success": False, "message": f"No data found for {', '.join(states)}"}

        if crop_matches:
            df_state = df_state[df_state["commodity"].isin(crop_matches)]
            if df_state.empty:
                return {"success": False, "message": f"No data found for crop {', '.join(crops)} in {', '.join(states)}."}

        for col in ["min_price", "max_price", "modal_price"]:
            if col in df_state.columns:
                df_state[col] = pd.to_numeric(df_state[col], errors="coerce")

        df_state = df_state.dropna(subset=["min_price", "max_price", "modal_price"], how="all")

        summary = (
            df_state.groupby(["state", "commodity"], as_index=False)[["min_price", "max_price", "modal_price"]]
                .mean(numeric_only=True)
                .sort_values(by="modal_price", ascending=False)
        )

        message = f"🌾 **Average Market Prices for {', '.join(states)}:**"
        meta = {
            "source": "Agmarknet (via data.gov.in)",
            "source_url": CROP_URL,
            "rows_used": len(df_state)
        }
        return {"success": True, "message": message, "table": summary, "meta": meta}
    except Exception as e:
        return {"success": False, "message": f"Error processing request: {str(e)}"}

# -----------------------------
# Rainfall
# -----------------------------
def answer_rainfall_question(question):
    states = extract_state_from_question(question)
    if not states:
        return {"success": False, "message": "Please specify one or more states to compare rainfall."}

    df = rainfall_df.copy()
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    rainfall_columns = [c for c in df.columns if "rain" in c]
    if not rainfall_columns:
        return {"success": False, "message": "No rainfall-related column found in dataset."}

    rainfall_col = rainfall_columns[0]
    state_col = "state" if "state" in df.columns else ("state_name" if "state_name" in df.columns else None)

    if not state_col:
        return {"success": False, "message": "State column missing in dataset."}

    mask = df[state_col].str.lower().isin([s.lower() for s in states])
    df = df[mask]

    if df.empty:
        return {"success": False, "message": f"No rainfall data found for {', '.join(states)}"}

    summary = df.groupby(state_col)[rainfall_col].apply(lambda x: pd.to_numeric(x, errors='coerce').mean()).reset_index()
    message = f"🌧️ **Average Rainfall Comparison for {', '.join(states)}:**"
    meta = {"source": "IMD (via data.gov.in)", "source_url": RAIN_URL, "rows_used": len(df)}

    return {"success": True, "message": message, "table": summary, "meta": meta}

# -----------------------------
# MSP
# -----------------------------
def answer_msp_question(question):
    df = msp_df.copy()

    crop_col = next((col for col in df.columns if any(word in col.lower() for word in ["crop", "commodity", "product", "name"])), None)
    if not crop_col:
        return {"success": False, "message": "Could not detect a crop or commodity column in the MSP dataset."}

    words = question.lower().split()
    matches = []
    for w in words:
        close_matches = difflib.get_close_matches(w, df[crop_col].astype(str).str.lower().unique(), n=1, cutoff=0.6)
        if close_matches:
            matches.append(close_matches[0])

    if not matches:
        return {"success": False, "message": "Please specify a valid crop name to get MSP data (e.g., Wheat, Paddy, Cotton)."}

    mask = df[crop_col].astype(str).str.lower().isin(matches)
    result = df[mask]

    if result.empty:
        return {"success": False, "message": f"No MSP data found for {', '.join(matches)}"}

    message = f"💰 **MSP Data for {', '.join(matches)}:**"
    meta = {"source": "Department of Agriculture & Farmers Welfare", "source_url": MSP_URL, "rows_used": len(result)}
    return {"success": True, "message": message, "table": result.head(10), "meta": meta}

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Project Samarth", layout="wide")
st.title("Project Samarth")

st.markdown("""
### 💬 Ask natural language questions:
- Compare the rainfall in Maharashtra and Gujarat  
- Show MSP data for Wheat  
- What is the crop production in Punjab  
""")

question = st.text_input("💬 Ask your question here:")
if st.button("Ask"):
    if "rain" in question.lower():
        res = answer_rainfall_question(question)
    elif "msp" in question.lower():
        res = answer_msp_question(question)
    elif "crop" in question.lower() or "production" in question.lower():
        res = answer_crop_production_question(question)
    else:
        st.warning("Please ask a question related to rainfall, crop production, or MSP.")
        res = None

    if res:
        if not res["success"]:
            st.warning(res["message"])
        else:
            st.markdown(res["message"])
            st.dataframe(res["table"].reset_index(drop=True))
            with st.expander("🔍 Data Source Details"):
                st.markdown(f"- **Source:** {res['meta']['source']}")
                st.markdown(f"- **Dataset URL:** {res['meta']['source_url']}")
                st.markdown(f"- **Rows Used:** {res['meta']['rows_used']}")

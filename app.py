import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import shap
import uuid
from supabase import create_client, Client

# --- Database Setup ---
SUPABASE_URL = "https://lkvoedulqrordxjuqvgx.supabase.co"
SUPABASE_KEY = "sb_publishable_tbpKlNGTY0fVYe88PgUyTw_nS2OP5Ef"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Generate an anonymous ID for the user's session
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())

# --- UI Setup ---
st.set_page_config(page_title="ClearCart Research Prototype", layout="wide")
st.title("🛒 ClearCart: Transparent AI Engine")
st.markdown("### The Algorithmic Nutrition Label & Data Dividend Prototype")

# --- Load Data & Train Model ---
@st.cache_resource
def load_and_train():
    try:
        file_path = r"D:\Resarch work\Research paper 2026\Kaggel\archive\amazon_beauty_clean.csv"
        df = pd.read_csv(file_path)
        if len(df.columns) < 3: raise ValueError("CSV columns squished.")
        
        rename_mapping = {
            'user_id': 'UserID', 'parent_asin': 'ProductID', 
            'rating': 'Rating', 'product_name': 'ProductName', 'title': 'ProductName'
        }
        df = df.rename(columns=rename_mapping)
        df = df[['UserID', 'ProductID', 'Rating', 'ProductName']].dropna()
    except Exception:
        data = {
            'UserID': ['User_01', 'User_01', 'User_02', 'User_02', 'User_03', 'User_03', 'User_01'],
            'ProductID': ['B001', 'B002', 'B001', 'B003', 'B002', 'B003', 'B003'],
            'Rating': [5, 4, 3, 5, 4, 4, 5],
            'ProductName': ['Herbivore Sea Mist', 'Vegan Dry Shampoo', 'Herbivore Sea Mist', 'Creamsicle Lip Balm', 'Vegan Dry Shampoo', 'Creamsicle Lip Balm', 'Creamsicle Lip Balm']
        }
        df = pd.DataFrame(data)
        
    item_stats = df.groupby('ProductID').agg(Item_Avg_Rating=('Rating', 'mean'), Item_Total_Reviews=('Rating', 'count')).reset_index()
    user_stats = df.groupby('UserID').agg(User_Avg_Rating=('Rating', 'mean'), User_Total_Reviews=('Rating', 'count')).reset_index()
    df = df.merge(item_stats, on='ProductID').merge(user_stats, on='UserID')
    
    features = ['Item_Avg_Rating', 'Item_Total_Reviews', 'User_Avg_Rating', 'User_Total_Reviews']
    X = df[features]
    y = df['Rating']
    
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    return df, model, explainer, features, X

with st.spinner("Initializing AI Engine..."):
    df, model, explainer, features, X = load_and_train()

# --- Storefront UI ---
st.sidebar.header("👤 Shopper Profile")
st.sidebar.caption(f"Session ID: `{st.session_state['session_id'][:8]}...`")
sample_user = st.sidebar.selectbox("Select a User ID to simulate:", df['UserID'].unique())

user_data = df[df['UserID'] == sample_user].iloc[0]
product_name = user_data['ProductName']
base_price = 45.00 

st.subheader(f"✨ Recommended for You: **{product_name}**")

# --- The Data Dividend Feature ---
st.write("---")
st.markdown("### 🛡️ Privacy Controls & Data Dividend")
st.write("Retailers use your data to predict what you want. Turn off your data to increase privacy, but lose your personalized discount.")

col1, col2 = st.columns(2)
with col1:
    share_history = st.toggle("Share my Purchase History (Boosts AI Accuracy)", value=True)
    share_behavior = st.toggle("Share my Rating Behavior", value=True)

discount = 0
if share_history: discount += 5.00
if share_behavior: discount += 3.50
final_price = base_price - discount

with col2:
    st.metric(label="Current Price", value=f"${final_price:.2f}", delta=f"-${discount:.2f} Data Dividend" if discount > 0 else "No Discount")

# --- The Algorithmic Nutrition Label ---
st.write("---")
st.markdown("### 🧐 Algorithmic Nutrition Label")
st.write("Why exactly are we recommending this item to you?")

if st.button("Generate AI Explanation"):
    # 1. LOG THE DATA TO SUPABASE SILENTLY
    try:
        supabase.table('user_interactions').insert({
            "session_id": st.session_state['session_id'],
            "shared_history": share_history,
            "shared_behavior": share_behavior,
            "final_price": float(final_price),
            "viewed_explanation": True
        }).execute()
    except Exception as e:
        print(f"Database logging error: {e}") # Fails silently so user isn't disrupted
        
    # 2. RUN THE AI LOGIC
    user_item_features = X.iloc[[user_data.name]].copy()
    if not share_history: user_item_features['Item_Total_Reviews'] = 0 
    if not share_behavior: user_item_features['User_Avg_Rating'] = 3.0 
        
    predicted_score = model.predict(user_item_features)[0]
    shap_values = explainer.shap_values(user_item_features)
    
    st.success(f"**AI Match Score:** {predicted_score:.2f} out of 5.0 Stars")
    
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)): base_val = base_val[0]
    current_shap = shap_values[0] if len(np.shape(shap_values)) > 1 else shap_values
    
    st.write(f"**Starting AI Baseline:** {base_val:.2f} Stars")
    for feature, shap_val in zip(features, current_shap):
        if shap_val > 0.05:
            st.info(f"⬆️ **{feature}** increased your match score by {abs(shap_val):.2f}")
        elif shap_val < -0.05:
            st.warning(f"⬇️ **{feature}** decreased your match score by {abs(shap_val):.2f}")

st.write("---")
st.caption("*Note: This is a research prototype. Anonymous interaction data (toggles clicked, features viewed) is collected for academic research purposes. No personal identifiable information (PII) is stored.*")
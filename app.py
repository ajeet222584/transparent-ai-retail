import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import shap
import uuid
import time
from supabase import create_client, Client

# --- Database Setup ---
SUPABASE_URL = "https://lkvoedulqrordxjuqvgx.supabase.co"
SUPABASE_KEY = "sb_publishable_tbpKlNGTY0fVYe88PgUyTw_nS2OP5Ef"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Session State (Memory) ---
if 'session_id' not in st.session_state: st.session_state['session_id'] = str(uuid.uuid4())
if 'start_time' not in st.session_state: st.session_state['start_time'] = time.time()
if 'purchased' not in st.session_state: st.session_state['purchased'] = False

# --- UI Setup ---
st.set_page_config(page_title="ClearCart India Prototype", layout="wide")
st.title("🛒 ClearCart: Transparent AI Engine")
st.markdown("### Context-Aware Retail with Algorithmic Nutrition Labels")

# --- Load Indian Market Data ---
@st.cache_resource
def load_data():
    # Built-in India-specific dataset
    data = {
        'UserID': ['User_01', 'User_01', 'User_02', 'User_02', 'User_03', 'User_03', 'User_01'],
        'ProductID': ['P1', 'P2', 'P1', 'P3', 'P2', 'P3', 'P3'],
        'Rating': [5, 4, 3, 5, 4, 4, 5],
        'ProductName': ['Lakmé Lumi Face Cream', 'Mamaearth Ubtan Face Wash', 'Lakmé Lumi Face Cream', 'Biotique Morning Nectar', 'Mamaearth Ubtan Face Wash', 'Biotique Morning Nectar', 'Biotique Morning Nectar'],
        'ImageURL': [
            'https://m.media-amazon.com/images/I/51wUqFhQWjL._SL1080_.jpg', 
            'https://m.media-amazon.com/images/I/51n2xK+2cTL._SL1200_.jpg',
            'https://m.media-amazon.com/images/I/51wUqFhQWjL._SL1080_.jpg',
            'https://m.media-amazon.com/images/I/51nO60K3jYL._SL1000_.jpg',
            'https://m.media-amazon.com/images/I/51n2xK+2cTL._SL1200_.jpg',
            'https://m.media-amazon.com/images/I/51nO60K3jYL._SL1000_.jpg',
            'https://m.media-amazon.com/images/I/51nO60K3jYL._SL1000_.jpg'
        ],
        'Description': [
            'A lightweight face cream with a hint of highlighter for a 3D glow.',
            'Tan removal face wash with Turmeric & Saffron for all skin types.',
            'A lightweight face cream with a hint of highlighter for a 3D glow.',
            'Nourishing face lotion blended with pure honey, wheatgerm and seaweed.',
            'Tan removal face wash with Turmeric & Saffron for all skin types.',
            'Nourishing face lotion blended with pure honey, wheatgerm and seaweed.',
            'Nourishing face lotion blended with pure honey, wheatgerm and seaweed.'
        ]
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

df, model, explainer, features, X = load_data()

# --- Sidebar: User Context ---
st.sidebar.header("👤 Shopper Profile")
sample_user = st.sidebar.selectbox("Select User Persona:", df['UserID'].unique(), help="Simulates different user rating histories.")

st.sidebar.markdown("### Contextual Input")
skin_type = st.sidebar.selectbox("Skin Type:", ["Oily", "Dry", "Combination", "Sensitive"])
purpose = st.sidebar.radio("Purpose:", ["For Myself", "As a Gift"])
season = st.sidebar.selectbox("Current Season:", ["Summer", "Monsoon", "Winter"])

# --- Main Storefront ---
user_data = df[df['UserID'] == sample_user].iloc[0]
base_price_inr = 499.00 

st.subheader("✨ Context-Aware Recommendation")
st.write(f"Based on your **{skin_type}** skin during the **{season}**, we recommend:")

colA, colB = st.columns([1, 2])
with colA:
    st.image(user_data['ImageURL'], width=200)
with colB:
    st.markdown(f"### {user_data['ProductName']}")
    st.write(user_data['Description'])
    
# --- The Data Dividend Feature ---
st.write("---")
st.markdown("### 🛡️ Privacy Controls & Data Dividend")
st.write("Turn off your data to increase privacy, but lose your personalized discount.")

col1, col2 = st.columns(2)
with col1:
    share_history = st.toggle("Share my Purchase History", value=True)
    share_behavior = st.toggle("Share my Rating Behavior", value=True)

discount_inr = 0
if share_history: discount_inr += 50.00
if share_behavior: discount_inr += 35.00
final_price_inr = base_price_inr - discount_inr

with col2:
    st.metric(label="Final Price", value=f"₹{final_price_inr:.2f}", delta=f"-₹{discount_inr:.2f} Data Dividend" if discount_inr > 0 else "No Discount")

# --- XAI Explanation ---
if st.button("🧐 Generate AI Explanation (Nutrition Label)"):
    user_item_features = X.iloc[[user_data.name]].copy()
    if not share_history: user_item_features['Item_Total_Reviews'] = 0 
    if not share_behavior: user_item_features['User_Avg_Rating'] = 3.0 
        
    predicted_score = model.predict(user_item_features)[0]
    shap_values = explainer.shap_values(user_item_features)
    
    st.success(f"**AI Match Score:** {predicted_score:.2f} / 5.0 Stars")
    
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)): base_val = base_val[0]
    current_shap = shap_values[0] if len(np.shape(shap_values)) > 1 else shap_values
    
    for feature, shap_val in zip(features, current_shap):
        if shap_val > 0.05: st.info(f"⬆️ **{feature}** increased match by {abs(shap_val):.2f}")
        elif shap_val < -0.05: st.warning(f"⬇️ **{feature}** decreased match by {abs(shap_val):.2f}")

# --- Purchase & Research Survey ---
st.write("---")
if not st.session_state['purchased']:
    if st.button("🛒 Buy Now", type="primary"):
        # Calculate time taken
        time_taken = round(time.time() - st.session_state['start_time'], 2)
        st.session_state['time_taken'] = time_taken
        st.session_state['purchased'] = True
        st.rerun()

if st.session_state['purchased']:
    st.success("🎉 Purchase Successful! Please help our research by answering 5 quick questions.")
    
    with st.form("research_survey"):
        st.write("*(1 = Strongly Disagree, 7 = Strongly Agree)*")
        q1 = st.slider("1. I understood why the AI recommended this product.", 1, 7, 4)
        q2 = st.slider("2. I felt in control of my personal data.", 1, 7, 4)
        q3 = st.slider("3. The 'Data Dividend' discount was a fair trade for my data.", 1, 7, 4)
        q4 = st.slider("4. The AI Explanation increased my trust in the retailer.", 1, 7, 4)
        q5 = st.slider("5. I would prefer shopping at stores that offer this transparency.", 1, 7, 4)
        
        if st.form_submit_button("Submit Survey & Save Data"):
            try:
                # Push ALL data to Supabase
                supabase.table('user_interactions').insert({
                    "session_id": st.session_state['session_id'],
                    "shared_history": share_history,
                    "shared_behavior": share_behavior,
                    "final_price": float(final_price_inr),
                    "time_taken_seconds": st.session_state['time_taken'],
                    "purchased": True,
                    "skin_type": skin_type,
                    "purpose": purpose,
                    "survey_q1": q1, "survey_q2": q2, "survey_q3": q3, "survey_q4": q4, "survey_q5": q5
                }).execute()
                st.balloons()
                st.info("✅ Data successfully saved to Supabase! Thank you for participating.")
            except Exception as e:
                st.error(f"Error saving to database: {e}")
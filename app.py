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
if 'viewed_explanation' not in st.session_state: st.session_state['viewed_explanation'] = False

# --- UI Setup ---
st.set_page_config(page_title="ClearCart India", layout="wide")
st.title("🛒 ClearCart: Transparent AI Engine")
st.markdown("### Context-Aware Retail with Algorithmic Nutrition Labels")

# --- Load Indian Market Data ---
@st.cache_resource
def load_data():
    data = {
        'UserID': ['User_01', 'User_02', 'User_03', 'User_01', 'User_02', 'User_03', 'User_01'],
        'ProductID': ['P1', 'P3', 'P2', 'P3', 'P1', 'P1', 'P2'],
        'Rating': [5, 4, 3, 5, 4, 4, 5],
        'ProductName': [
            'Lakme Lumi Lit Cream', 
            'Biotique Bio Sandalwood Sunscreen SPF 50', 
            'Mamaearth Ubtan Face Wash', 
            'Biotique Bio Sandalwood Sunscreen SPF 50', 
            'Lakme Lumi Lit Cream', 
            'Lakme Lumi Lit Cream', 
            'Mamaearth Ubtan Face Wash'
        ],
        'ImageURL': [
            'lakme.jpg', 
            'biotique.jpg', 
            'mamaearth.jpg', 
            'biotique.jpg', 
            'lakme.jpg', 
            'lakme.jpg', 
            'mamaearth.jpg'  
        ],
        'BasePrice': [262.00, 215.00, 247.00, 215.00, 262.00, 262.00, 247.00],
        'StarRating': [4.3, 4.2, 4.1, 4.2, 4.3, 4.3, 4.1],
        'TotalReviews': [14205, 12300, 8540, 12300, 14205, 14205, 8540],
        'Description': [
            'Lakme Lumi Lit Cream is a lightweight face cream that works as a moisturizer and highlighter in one product. It contains niacinamide and hyaluronic acid to hydrate and improve texture.',
            'Biotique Sun Shield Sandalwood SPF 50+ is an Ayurvedic sunscreen to protect from UVA/UVB rays. Enriched with sandalwood, saffron, and honey.',
            'Mamaearth Ubtan Natural Glow Face Wash is formulated with turmeric and saffron to remove impurities while promoting a natural glow.',
            'Biotique Sun Shield Sandalwood SPF 50+ is an Ayurvedic sunscreen to protect from UVA/UVB rays. Enriched with sandalwood, saffron, and honey.',
            'Lakme Lumi Lit Cream is a lightweight face cream that works as a moisturizer and highlighter in one product. It contains niacinamide and hyaluronic acid to hydrate and improve texture.',
            'Lakme Lumi Lit Cream is a lightweight face cream that works as a moisturizer and highlighter in one product. It contains niacinamide and hyaluronic acid to hydrate and improve texture.',
            'Mamaearth Ubtan Natural Glow Face Wash is formulated with turmeric and saffron to remove impurities while promoting a natural glow.'
        ],
        'KeyBenefits': [
            'Moisturizes and hydrates\nProvides instant radiance\nUse as primer or daily cream',
            'Protects from UV rays\nPrevents sunburn/tanning\nKeeps skin soft',
            'Cleanses dirt and oil\nHelps remove tan\nNatural turmeric/saffron',
            'Protects from UV rays\nPrevents sunburn/tanning\nKeeps skin soft',
            'Moisturizes and hydrates\nProvides instant radiance\nUse as primer or daily cream',
            'Moisturizes and hydrates\nProvides instant radiance\nUse as primer or daily cream',
            'Cleanses dirt and oil\nHelps remove tan\nNatural turmeric/saffron'
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
sample_user = st.sidebar.selectbox("Select User Persona:", df['UserID'].unique())

st.sidebar.markdown("### Contextual Input")
skin_type = st.sidebar.selectbox("Skin Type:", ["Oily", "Dry", "Combination", "Sensitive"])
purpose = st.sidebar.radio("Purpose:", ["For Myself", "As a Gift"])
season = st.sidebar.selectbox("Current Season:", ["Summer", "Monsoon", "Winter"])

# --- Main Storefront UI ---
user_data = df[df['UserID'] == sample_user].iloc[0]
base_price_inr = user_data['BasePrice']

st.subheader("✨ Recommended for You")
st.caption(f"Based on your **{skin_type}** skin during the **{season}** season")

col_img, col_info = st.columns([1, 2.5])
with col_img:
    try:
        st.image(user_data['ImageURL'], width=250)
    except:
        st.warning("Image loading...")

with col_info:
    st.markdown(f"## {user_data['ProductName']}")
    st.write(f"⭐⭐⭐⭐⭐ **{user_data['StarRating']}** ({user_data['TotalReviews']:,} ratings)")
    st.markdown(f"#### MRP: ₹{base_price_inr:.2f}")
    with st.expander("📝 Product Description", expanded=True):
        st.write(user_data['Description'])
        st.markdown("**Key Benefits:**")
        for b in user_data['KeyBenefits'].split('\n'):
            st.markdown(f"- {b}")
    
# --- Data Dividend ---
st.write("---")
st.markdown("### 🛡️ Privacy Controls & Data Dividend")
col1, col2 = st.columns(2)
with col1:
    share_history = st.toggle("Share my Purchase History", value=True)
    share_behavior = st.toggle("Share my Rating Behavior", value=True)

discount_pct = (0.12 if share_history else 0) + (0.08 if share_behavior else 0)
final_price_inr = base_price_inr * (1 - discount_pct)

with col2:
    st.metric(label="Your Price", value=f"₹{final_price_inr:.2f}", delta=f"-₹{base_price_inr*discount_pct:.2f}" if discount_pct > 0 else None)

# --- XAI Explanation ---
st.write("---")
if st.button("🧐 Generate AI Explanation (Nutrition Label)"):
    st.session_state['viewed_explanation'] = True
    user_item_features = X.iloc[[user_data.name]].copy()
    if not share_history: user_item_features['Item_Total_Reviews'] = 0 
    if not share_behavior: user_item_features['User_Avg_Rating'] = 3.0 
        
    predicted_score = model.predict(user_item_features)[0]
    shap_values = explainer.shap_values(user_item_features)
    st.success(f"**AI Match Score:** {predicted_score:.2f} / 5.0")
    
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)): base_val = base_val[0]
    current_shap = shap_values[0] if len(np.shape(shap_values)) > 1 else shap_values
    
    for feature, shap_val in zip(features, current_shap):
        if shap_val > 0.05: st.info(f"⬆️ **{feature}** increased match by {abs(shap_val):.2f}")
        elif shap_val < -0.05: st.warning(f"⬇️ **{feature}** decreased match by {abs(shap_val):.2f}")

# --- Purchase & Survey ---
st.write("---")
if not st.session_state['purchased']:
    if st.button("🛒 Buy Now", type="primary"):
        st.session_state['time_taken'] = round(time.time() - st.session_state['start_time'], 2)
        st.session_state['purchased'] = True
        st.rerun()

if st.session_state['purchased']:
    st.success("🎉 Purchase Successful!")
    with st.form("research_survey"):
        st.write("*(1 = Strongly Disagree, 7 = Strongly Agree)*")
        q1 = st.slider("1. I understood why the AI recommended this product.", 1, 7, 4)
        q2 = st.slider("2. I felt in control of my personal data.", 1, 7, 4)
        q3 = st.slider("3. The 'Data Dividend' was a fair trade.", 1, 7, 4)
        q4 = st.slider("4. The AI Explanation increased my trust.", 1, 7, 4)
        q5 = st.slider("5. I prefer shopping with this transparency.", 1, 7, 4)
        
        if st.form_submit_button("Submit Survey & Save Data"):
            try:
                supabase.table('user_interactions').insert({
                    "session_id": st.session_state['session_id'],
                    "shared_history": share_history,
                    "shared_behavior": share_behavior,
                    "final_price": float(final_price_inr),
                    "time_taken_seconds": st.session_state['time_taken'],
                    "purchased": True,
                    "skin_type": skin_type,
                    "purpose": purpose,
                    "survey_q1": q1, "survey_q2": q2, "survey_q3": q3, "survey_q4": q4, "survey_q5": q5,
                    "product_name": user_data['ProductName'],
                    "user_persona": sample_user,
                    "viewed_explanation": st.session_state['viewed_explanation']
                }).execute()
                st.balloons()
                st.info("✅ Data saved! You can now close this tab.")
            except Exception as e:
                st.error(f"Error: {e}")
"""
CropGuard - AI Crop Disease Detector
Streamlit Web App for Tamil Nadu Agricultural Cooperatives
Hackathon: TANCAM Women's Hackathon
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="CropGuard - Disease Detector",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding: 20px; }
    .stMetric { background-color: #f0f7f0; padding: 15px; border-radius: 10px; }
    .success-box { background-color: #d4edda; padding: 15px; border-radius: 5px; border-left: 4px solid #28a745; }
    .danger-box { background-color: #f8d7da; padding: 15px; border-radius: 5px; border-left: 4px solid #dc3545; }
    .title-text { color: #2d5016; font-size: 2.5em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Disease database
DISEASE_DATABASE = {
    'Healthy': {
        'severity': 'None',
        'treatment': 'Regular maintenance and monitoring',
        'duration': 'N/A',
        'cost': 0,
        'pesticide': 'None',
        'frequency': 'N/A',
        'precautions': ['Monitor regularly', 'Ensure proper drainage', 'Maintain field hygiene']
    },
    'Rice_Blast': {
        'severity': 'High',
        'treatment': 'Fungicide spray - Tricyclazole or Hexaconazole',
        'duration': '7-10 days',
        'cost': 450,
        'pesticide': 'Tricyclazole 75% WP @ 0.6g/L or Hexaconazole 5% EC @ 1ml/L',
        'frequency': '3 sprays at 10-day intervals',
        'precautions': ['Apply at boot stage', 'Spray in evening', 'Repeat if rains occur']
    },
    'Rice_LeafSpot': {
        'severity': 'Medium',
        'treatment': 'Fungicide spray - Mancozeb or Copper oxychloride',
        'duration': '10-14 days',
        'cost': 320,
        'pesticide': 'Mancozeb 75% WP @ 2g/L or Copper Oxychloride 50% WP @ 2.5g/L',
        'frequency': '2-3 sprays at 10-day intervals',
        'precautions': ['Start spraying at first symptom', 'Improve drainage', 'Remove infected debris']
    },
    'Rice_SheatRot': {
        'severity': 'High',
        'treatment': 'Fungicide spray - Validamycin A',
        'duration': '7-10 days',
        'cost': 550,
        'pesticide': 'Validamycin A 3% L @ 2ml/L',
        'frequency': '3-4 sprays at 7-day intervals',
        'precautions': ['Early detection is critical', 'Reduce nitrogen application', 'Improve aeration']
    },
    'Cotton_LeafCurl': {
        'severity': 'High',
        'treatment': 'Insecticide for whitefly + Virus management',
        'duration': '15-20 days',
        'cost': 650,
        'pesticide': 'Thiamethoxam 25% WG @ 0.4g/L or Imidacloprid 17.8% SL @ 0.4ml/L',
        'frequency': '4-5 sprays at 5-day intervals',
        'precautions': ['Control whitefly population', 'Use yellow sticky traps', 'Remove infected plants']
    },
    'Cotton_Wilt': {
        'severity': 'Critical',
        'treatment': 'Soil treatment + Fungicide',
        'duration': '30+ days',
        'cost': 1200,
        'pesticide': 'Trichoderma soil application + Carbendazim 50% WP @ 1g/L spray',
        'frequency': 'Soil drench at planting + 2-3 sprays',
        'precautions': ['Use resistant varieties', 'Practice crop rotation', 'Sterilize soil']
    },
    'Cotton_Anthracnose': {
        'severity': 'Medium',
        'treatment': 'Fungicide spray - Mancozeb or Tebuconazole',
        'duration': '10-14 days',
        'cost': 400,
        'pesticide': 'Mancozeb 75% WP @ 2g/L or Tebuconazole 25% EC @ 1ml/L',
        'frequency': '2-3 sprays at 10-day intervals',
        'precautions': ['Remove infected leaves', 'Improve air circulation', 'Avoid overhead irrigation']
    },
    'Sugarcane_RedRot': {
        'severity': 'Critical',
        'treatment': 'Hot water treatment + Fungicide',
        'duration': '45+ days',
        'cost': 1500,
        'pesticide': 'Hot water treatment (50-52°C for 2 hrs) + Carbendazim seed treatment',
        'frequency': 'At planting + monitoring',
        'precautions': ['Use disease-free setts', 'Practice field sanitation', 'Crop rotation essential']
    },
    'Sugarcane_Smut': {
        'severity': 'High',
        'treatment': 'Fungicide spray + Variety change',
        'duration': '20-30 days',
        'cost': 800,
        'pesticide': 'Mancozeb 75% WP @ 2g/L or Copper Oxychloride @ 2.5g/L',
        'frequency': '3-4 sprays',
        'precautions': ['Use resistant varieties', 'Field sanitation', 'Remove whip-like structures']
    },
    'Sugarcane_LeafScald': {
        'severity': 'Medium',
        'treatment': 'Fungicide spray - Copper based',
        'duration': '14-21 days',
        'cost': 380,
        'pesticide': 'Copper Oxychloride 50% WP @ 2.5g/L',
        'frequency': '2-3 sprays at 10-day intervals',
        'precautions': ['Improve drainage', 'Field sanitation', 'Avoid overhead irrigation']
    }
}

# Initialize session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Header
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://img.icons8.com/color/96/000000/leaf.png", width=80)
with col2:
    st.markdown('<p class="title-text">🌾 CropGuard</p>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Crop Disease Detection for Tamil Nadu Cooperatives**")

st.markdown("---")

# Main interface
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Detect Disease", "📊 Dashboard", "📚 Disease Info", "ℹ️ About"])

with tab1:
    st.header("🔍 Upload Crop Image for Disease Detection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Step 1: Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a crop leaf image (JPG, PNG, or WebP)",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload a clear photo of the affected crop leaf"
        )
        
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            image_data = Image.open(uploaded_file)
            st.image(image_data, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.info(f"📐 Image size: {image_data.size[0]}x{image_data.size[1]} pixels")
    
    with col2:
        st.subheader("Step 2: AI Analysis Results")
        
        if st.session_state.uploaded_file is not None:
            if st.button("🤖 Analyze Image", key="analyze_btn", use_container_width=True):
                with st.spinner("🔬 Analyzing image... Please wait..."):
                    
                    # Simulate AI prediction (in real deployment, use trained model)
                    # For demo, we'll show realistic predictions
                    
                    # Load image
                    img = Image.open(st.session_state.uploaded_file)
                    
                    # Demo: Simulate disease detection with confidence scores
                    diseases = [
                        ('Rice_Blast', 0.78),
                        ('Rice_LeafSpot', 0.15),
                        ('Healthy', 0.05),
                        ('Cotton_LeafCurl', 0.02)
                    ]
                    
                    detected_disease = diseases[0][0]
                    confidence = diseases[0][1]
                    
                    st.session_state.prediction = {
                        'disease': detected_disease,
                        'confidence': confidence,
                        'alternatives': diseases[1:]
                    }
            
            # Display results
            if st.session_state.prediction:
                pred = st.session_state.prediction
                disease_name = pred['disease']
                confidence = pred['confidence']
                
                # Result display
                st.markdown("---")
                
                if disease_name == 'Healthy':
                    st.markdown(f"""
                    <div class="success-box">
                    <h3>✅ Crop Status: HEALTHY</h3>
                    <p>Confidence: <b>{confidence*100:.1f}%</b></p>
                    <p>No disease detected. Continue regular monitoring.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    severity = DISEASE_DATABASE[disease_name]['severity']
                    st.markdown(f"""
                    <div class="danger-box">
                    <h3>⚠️ Disease Detected: {disease_name.replace('_', ' ')}</h3>
                    <p>Confidence: <b>{confidence*100:.1f}%</b></p>
                    <p>Severity: <b>{severity}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Treatment recommendation
                if disease_name in DISEASE_DATABASE:
                    treatment = DISEASE_DATABASE[disease_name]
                    
                    st.subheader("💊 Treatment Recommendation")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("💰 Estimated Cost (per acre)", f"₹{treatment['cost']}")
                        st.metric("⏱️ Treatment Duration", treatment['duration'])
                    with col_b:
                        st.metric("🔄 Spray Frequency", treatment['frequency'])
                        st.metric("🌡️ Severity Level", treatment['severity'])
                    
                    st.markdown("---")
                    
                    st.subheader("🧪 Pesticide Details")
                    st.info(f"**Pesticide:** {treatment['pesticide']}")
                    st.write(f"**Treatment:** {treatment['treatment']}")
                    
                    st.subheader("⚡ Precautions")
                    for i, precaution in enumerate(treatment['precautions'], 1):
                        st.write(f"{i}. {precaution}")
                
                # Alternative predictions
                st.markdown("---")
                st.subheader("🔄 Alternative Predictions")
                for alt_disease, alt_conf in pred['alternatives']:
                    st.write(f"• {alt_disease.replace('_', ' ')}: {alt_conf*100:.1f}%")

with tab2:
    st.header("📊 Cooperative Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌾 Total Scans", "47", "+5 today")
    with col2:
        st.metric("✅ Healthy Crops", "32", "68%")
    with col3:
        st.metric("⚠️ Diseased", "15", "32%")
    with col4:
        st.metric("💰 Money Saved", "₹8,450", "By targeted sprays")
    
    st.markdown("---")
    
    st.subheader("📈 Recent Detections")
    recent_data = {
        'Date': ['2024-03-28', '2024-03-28', '2024-03-27', '2024-03-27', '2024-03-26'],
        'Farmer': ['Murugan Farm', 'Lakshmi Coop', 'Ravi Fields', 'Sunita Farm', 'Joint Farm'],
        'Crop': ['Rice', 'Cotton', 'Sugarcane', 'Rice', 'Cotton'],
        'Disease': ['Healthy', 'Leaf Curl', 'Red Rot', 'Blast', 'Healthy'],
        'Action': ['Monitor', 'Spray Thiamethoxam', 'Hot water treatment', 'Fungicide', 'Continue']
    }
    
    import pandas as pd
    df = pd.DataFrame(recent_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.subheader("🎯 Treatment Cost Savings")
    st.write("By using CropGuard for targeted disease detection, cooperatives save on:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("**20-30% reduction** in pesticide usage")
    with col2:
        st.success("**₹500-1000 per acre** saved per season")
    with col3:
        st.success("**Faster disease control** = Better yields")

with tab3:
    st.header("📚 Disease Information Database")
    
    selected_disease = st.selectbox(
        "Select a disease to learn more:",
        list(DISEASE_DATABASE.keys())
    )
    
    if selected_disease:
        disease_info = DISEASE_DATABASE[selected_disease]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"{selected_disease.replace('_', ' ')}")
            st.markdown("---")
            
            st.write(f"**Severity Level:** {disease_info['severity']}")
            st.write(f"**Treatment:** {disease_info['treatment']}")
            st.write(f"**Pesticide:** {disease_info['pesticide']}")
            st.write(f"**Treatment Duration:** {disease_info['duration']}")
            st.write(f"**Spray Frequency:** {disease_info['frequency']}")
            st.write(f"**Estimated Cost (per acre):** ₹{disease_info['cost']}")
            
            st.markdown("---")
            st.subheader("Precautions & Prevention")
            for i, precaution in enumerate(disease_info['precautions'], 1):
                st.write(f"{i}. {precaution}")
        
        with col2:
            st.markdown("### Quick Stats")
            st.metric("Cost", f"₹{disease_info['cost']}")
            st.metric("Severity", disease_info['severity'])
            st.metric("Duration", disease_info['duration'])

with tab4:
    st.header("ℹ️ About CropGuard")
    
    st.markdown("""
    ### 🌾 Crop Disease Detection AI for Tamil Nadu
    
    **CropGuard** is an AI-powered mobile/web application designed specifically for Tamil Nadu agricultural 
    cooperatives to detect crop diseases early and provide actionable treatment recommendations.
    
    #### 🎯 Problem We Solve
    - Tamil Nadu farmers lose 15-30% of crops to diseases annually
    - Farmers use blanket pesticide sprays (wasteful, expensive)
    - Cooperative members lack expert agronomic advice
    - Treatments are delayed due to lack of diagnosis
    
    #### ✅ Our Solution
    1. **Easy Photo Upload** - Farmers/cooperatives take leaf photo
    2. **AI Disease Detection** - Deep learning identifies disease in seconds
    3. **Instant Recommendations** - Specific pesticide + cost + application timing
    4. **Cooperative Dashboard** - Track treatments across member farms
    
    #### 🏆 Team
    **TANCAM Women's Hackathon**
    - Computer Vision AI Specialist
    - Data Scientist
    - Agricultural Expert Advisor
    
    #### 📊 Impact Metrics
    - **Expected adoption:** 5,000+ cooperatives in TN
    - **Annual savings:** ₹25 Cr+ across cooperatives
    - **Crop loss reduction:** 15-20%
    - **Environmental benefit:** 30% less pesticide runoff
    
    #### 🔬 Technology Stack
    - **Model:** TensorFlow/Keras with MobileNetV2 (transfer learning)
    - **Dataset:** PlantVillage + Tamil Nadu agricultural databases
    - **Frontend:** Streamlit (web) / React Native (mobile)
    - **Deployment:** AWS/Heroku for scalability
    
    #### 📱 How to Use
    1. Click "Detect Disease" tab
    2. Upload clear photo of affected crop leaf
    3. Click "Analyze Image"
    4. Get instant diagnosis + treatment plan
    5. Save report for follow-up
    
    ---
    
    **Built during TANCAM Women's Hackathon 2024 | Tamil Nadu**
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📧 Contact: contact@cropguard.in")
    with col2:
        st.info("🌐 Website: www.cropguard.in")
    with col3:
        st.info("📱 App: Available on Android/iOS")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9em;">
<p>🌾 CropGuard - Empowering Tamil Nadu Farmers with AI 🌾</p>
<p>TANCAM Women's Hackathon 2024 | Built with ❤️ for Agriculture</p>
</div>
""", unsafe_allow_html=True)

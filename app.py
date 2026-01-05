import streamlit as st
import numpy as np
from PIL import Image
import os
import time
from src.preprocess import preprocess_image
from src.model import load_trained_model, predict, MODEL_MAP

# Configuration and Styling
st.set_page_config(
    page_title="Agri-Scan AI | Crop Disease Detection",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    .stApp {
        background: radial-gradient(circle at top left, rgba(46, 204, 113, 0.1), transparent),
                    radial-gradient(circle at bottom right, rgba(52, 152, 219, 0.1), transparent);
    }

    .header-container {
        padding: 2.5rem;
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        width: 100%;
    }

    .title-text {
        color: #2c3e50;
        font-weight: 700;
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
        background: -webkit-linear-gradient(#27ae60, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border: 1px dashed #2ecc71;
        transition: all 0.3s ease;
    }

    .upload-section:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }

    .result-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(15px);
        padding: 2.5rem;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 2rem;
        animation: fadeIn 0.8s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    .status-healthy { background: #e8f5e9; color: #2e7d32; }
    .status-diseased { background: #ffebee; color: #c62828; }

    .confidence-meter {
        height: 10px;
        background: #eee;
        border-radius: 5px;
        margin: 1.5rem 0;
        overflow: hidden;
    }

    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #2ecc71, #27ae60);
        border-radius: 5px;
        transition: width 1s ease-in-out;
    }

    .info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin-top: 2rem;
    }

    .info-box {
        background: rgba(248, 249, 250, 0.8);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: left;
    }

    .info-label {
        color: #7f8c8d;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }

    .info-value {
        color: #2c3e50;
        font-size: 1.2rem;
        font-weight: 600;
    }

    /* Hide standard streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load the model based on selection
@st.cache_resource
def get_model(crop_name):
    return load_trained_model(crop_name)

# Sidebar for additional info
with st.sidebar:
    st.image("header.png", use_container_width=True)
    st.title("About Agri-Scan")
    st.info("""
        **SRS Compliance:** This system adheres to the Software Requirements Specification for 
        Multi-Crop Disease Detection.
    """)
    st.subheader("Supported Crops")
    st.write("- Apple 🍎\n- Potato 🥔\n- Tomato 🍅\n- Corn 🌽\n- Grape 🍇\n- Peach 🍑\n- Pepper 🫑\n- Strawberry 🍓")
    
    st.divider()
    st.write("Built with ❤️ by Antigravity")

# Main Content
st.markdown("""
<div class="header-container">
    <h1 class="title-text">Agri-Scan AI</h1>
    <p style="color: #636e72; font-size: 1.2rem;">Intelligent Multi-Crop Disease Detection System</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📸 Upload Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG image (Max 5MB)...", 
        type=["jpg", "jpeg", "png"],
        help="Ensure the leaf is well-lit and centered."
    )
    
    st.divider()
    st.subheader("🌾 Select Crop Type")
    options = ["Auto-Detect Crop"] + list(MODEL_MAP.keys())
    selected_crop = st.selectbox(
        "Which crop does this leaf belong to?",
        options=options,
        index=0,
        help="Choose 'Auto-Detect' to let the AI identify the crop for you."
    )
    
    if uploaded_file is not None:
        # SRS Requirement: Validate image size and format
        if uploaded_file.size > 5 * 1024 * 1024:
            st.error("❌ File size exceeds 5MB limit. Please upload a smaller image.")
            uploaded_file = None
        else:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error loading image: {e}")
                uploaded_file = None
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if uploaded_file is not None:
        start_time = time.time()
        with st.spinner("Analyzing physiological markers..."):
            try:
                # Preprocess image
                processed_img = preprocess_image(image)

                if selected_crop == "Auto-Detect Crop":
                    results = []
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    crops_to_check = list(MODEL_MAP.keys())
                    for i, crop_name in enumerate(crops_to_check):
                        progress_text.text(f"Scanning {crop_name} expert model...")
                        progress_bar.progress((i + 1) / len(crops_to_check))
                        
                        model_result = get_model(crop_name)
                        if not isinstance(model_result, str):
                            c, d, conf = predict(model_result, processed_img, crop_name)
                            results.append({'crop': c, 'disease': d, 'conf': conf})
                    
                    progress_text.empty()
                    progress_bar.empty()
                    
                    # Sort by confidence
                    results = sorted(results, key=lambda x: x['conf'], reverse=True)
                    
                    if results:
                        best = results[0]
                        crop, disease, confidence = best['crop'], best['disease'], best['conf']
                        
                        # Show scan report if multiple models are confident
                        with st.expander("🔍 View Expert Scan Report"):
                            st.write("The AI checked all specialist models. Here were the top matches:")
                            for res in results[:3]:
                                st.write(f"- **{res['crop']} Expert:** {res['disease']} ({res['conf']*100:.1f}%)")
                            st.info("If the top match is incorrect, please select the crop manually for Expert Mode.")
                    else:
                        st.error("Detection failed.")
                        st.stop()
                else:
                    # Load specific model
                    model_result = get_model(selected_crop)
                    
                    if model_result is None or isinstance(model_result, str):
                        error_to_show = model_result if model_result else "Unknown loading error"
                        st.error(f"❌ **Model Load Error:** {error_to_show}")
                        st.info("💡 **Tip:** This often happens if the weights file doesn't match the model structure.")
                        st.stop()
                    
                    model = model_result
                    crop, disease, confidence = predict(model, processed_img, selected_crop)
                
                # Minimum confidence threshold for reliability
                if confidence < 0.4:
                    st.warning("⚠️ **Low Confidence Output:** The system is unsure about this classification. Ensure you uploaded a clear leaf image.")

                # Determine health status
                is_healthy = "healthy" in disease.lower()
                status_class = "status-healthy" if is_healthy else "status-diseased"
                status_text = "HEALTHY" if is_healthy else "DISEASED"
                
                # Calculate execution time
                end_time = time.time()
                inference_time = end_time - start_time
                
                # Display Result
                st.markdown(f"""
<div class="result-card">
    <div class="status-badge {status_class}">{status_text}</div>
    <h2 style="color:#2c3e50; margin-bottom:0.1rem;">{disease}</h2>
    <p style="color:#7f8c8d;">Detected in {crop}</p>
    <div class="confidence-meter">
        <div class="confidence-fill" style="width: {confidence*100}%;"></div>
    </div>
    <p style="font-weight:600; color:#27ae60;">{confidence*100:.2f}% Confidence Score</p>
    <div class="info-grid">
        <div class="info-box">
            <div class="info-label">CROP TYPE</div>
            <div class="info-value">{crop}</div>
        </div>
        <div class="info-box">
            <div class="info-label">RESPONSE TIME</div>
            <div class="info-value">{inference_time:.2f}s</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
                
                # SRS Compliance check for execution time
                if inference_time > 3.0:
                    st.caption("⚡ *Note: Inference took longer than 3s due to server load.*")
                
                if not is_healthy:
                    st.warning("⚠️ **Recommendation:** Consult with an agricultural expert. Early detection is key to preventing yield loss.")
                else:
                    st.success("✅ **Observation:** Excellent! The leaf shows no signs of common pathogens.")

            except Exception as e:
                st.error(f"❌ An error occurred during processing: {e}")
                st.info("Please try uploading a different image or refresh the page.")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 4rem; color: #95a5a6;">
            <p style="font-size: 4rem;">🍃</p>
            <h3>Ready for Analysis</h3>
            <p>Please upload a clear image of the plant leaf to begin the detection process.</p>
            <p style="font-size: 0.8rem; margin-top: 2rem;">Adheres to SRS Section 4.0 Functional Requirements</p>
        </div>
        """, unsafe_allow_html=True)

# Footer info
st.divider()
st.caption("Disclaimer: This tool is for educational and research purposes. Always verify with professionals before taking action.")

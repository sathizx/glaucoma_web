import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from PIL import Image
import os
import joblib
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import asyncio

# Fix for "RuntimeError: no running event loop" on Windows
try:
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except Exception:
    pass

# Set up the Streamlit app
st.set_page_config(page_title="Glaucoma Detection System", layout="wide")
st.title("Glaucoma Detection from Retinal Images")

# Initialize session state for model persistence
if 'model_trained_and_loaded' not in st.session_state:
    st.session_state.model_trained_and_loaded = False
    st.session_state.model = None
    st.session_state.scaler = None
    st.session_state.feature_size = None  # To track feature size consistency

# Paths for dataset and saved models
DATASET_PATH = r"dataset"
MODEL_PATH = r"saved_models/glaucoma_model.pkl"
SCALER_PATH = r"saved_models/scaler.pkl"
FEATURE_SIZE_PATH = r"saved_models/feature_size.pkl"

# Consistent feature extraction function
def extract_features(image):
    """Extract LBP, HOG, and GLCM features from an image."""
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Extract LBP features
    lbp = local_binary_pattern(img_gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist / np.sum(lbp_hist)

    # Extract HOG features
    hog_features, _ = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

    # Extract GLCM features
    glcm = graycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    glcm_props = [graycoprops(glcm, prop).ravel()[0] for prop in ['contrast', 'homogeneity', 'energy', 'correlation']]

    # Combine all features
    combined_features = np.concatenate((lbp_hist, hog_features, glcm_props))
    return combined_features

# Automatically load pre-trained model on app start
def load_trained_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        st.session_state.model = joblib.load(MODEL_PATH)
        st.session_state.scaler = joblib.load(SCALER_PATH)
        if os.path.exists(FEATURE_SIZE_PATH):
            st.session_state.feature_size = joblib.load(FEATURE_SIZE_PATH)
            st.sidebar.success("Feature size loaded successfully!")
        else:
            st.sidebar.warning("Feature size file not found. Please retrain the model.")
        st.session_state.model_trained_and_loaded = True
        st.sidebar.success("Pre-trained model loaded successfully!")
    else:
        st.sidebar.warning("No pre-trained model found. Please train the model.")

# Load model on app start
load_trained_model()

# Image testing section
st.header("Glaucoma Detection")
uploaded_file = st.file_uploader("Upload a retinal image for analysis:", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    if not st.session_state.model_trained_and_loaded:
        st.warning("Please train or load the model first.")
    else:
        try:
            # Process uploaded image
            image = Image.open(uploaded_file).convert("RGB")

            # Reduce image size for display
            resized_image = image.resize((150, 150))  # Resize to 150x150 for display
            st.image(resized_image, caption="Uploaded Retinal Image (Resized)", use_container_width=False)

            # Resize image for processing
            img_np = np.array(image.resize((224, 224)))
            features = extract_features(img_np)

            # Ensure feature size consistency
            if st.session_state.feature_size and len(features) != st.session_state.feature_size:
                raise ValueError(f"Feature size mismatch: Expected {st.session_state.feature_size}, got {len(features)}")

            scaled_features = st.session_state.scaler.transform([features])

            # Prediction
            prediction = st.session_state.model.predict(scaled_features)[0]
            confidence = st.session_state.model.predict_proba(scaled_features)[0]

            # Display results
            if prediction == 0:
                st.error("Glaucoma Detected!")
            else:
                st.success("No Glaucoma Detected.")
            
            # Plot prediction confidence
            st.subheader("Prediction Confidence")
            labels = ["Glaucoma", "Normal"]
            colors = ["red", "green"]
            plt.bar(labels, confidence, color=colors)
            plt.ylim(0, 1)
            plt.ylabel("Probability")
            plt.title("Prediction Confidence")
            st.pyplot(plt)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
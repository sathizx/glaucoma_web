import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
import cv2
from PIL import Image

# Paths for dataset and saved models
DATASET_PATH = r"dataset"
MODEL_PATH = r"saved_models/glaucoma_model.pkl"
SCALER_PATH = r"saved_models/scaler.pkl"
FEATURE_SIZE_PATH = r"saved_models/feature_size.pkl"

# Feature extraction function
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

def train_model():
    try:
        # Preprocessing transformations for dataset
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Load dataset
        dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
        dataset.samples = [s for s in dataset.samples if os.path.isfile(s[0])]

        if len(dataset.samples) == 0:
            print("No valid images found in the dataset path!")
            return

        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        features, labels = [], []
        for img, label in data_loader:
            img_np = np.transpose(img.squeeze().numpy(), (1, 2, 0)) * 255
            img_np = img_np.astype(np.uint8)
            features.append(extract_features(img_np))
            labels.append(label.item())

        features = np.array(features)
        labels = np.array(labels)

        # Handle class imbalance
        smote = SMOTE()
        features, labels = smote.fit_resample(features, labels)

        # Save feature size for consistency
        feature_size = features.shape[1]
        joblib.dump(feature_size, FEATURE_SIZE_PATH)
        print(f"Feature size saved: {feature_size}")

        # Split and preprocess
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Train classifiers
        rf = RandomForestClassifier(random_state=42)
        ada = AdaBoostClassifier(random_state=42)
        svm = SVC(probability=True, random_state=42)

        # Combine classifiers using VotingClassifier
        model = VotingClassifier(estimators=[('rf', rf), ('ada', ada), ('svm', svm)], voting='soft')
        model.fit(X_train, y_train)

        # Save model and scaler
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

        # Evaluate model
        accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"Training completed! Model accuracy: {accuracy:.2f}")
        print(f"Model saved at: {MODEL_PATH}")
        print(f"Scaler saved at: {SCALER_PATH}")

    except Exception as e:
        print(f"Error during training: {str(e)}")

if __name__ == "__main__":
    train_model()
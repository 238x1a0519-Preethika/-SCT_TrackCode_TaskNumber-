import os
import cv2
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

images = []
lables = []

cat_path = "cats"
dog_path = "Dogs"

# Optional: For advanced feature extraction (highly recommended for raw images)
try:
    from skimage.feature import hog
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def extract_features(img, use_hog=True):
    """
    Extracts features from an image.
    Supports either raw flattened pixel intensity or HOG features.
    """
    # Resize to a consistent small size
    img_resized = cv2.resize(img, (64, 64))
    
    if use_hog and HAS_SKIMAGE:
        # Convert to grayscale for HOG
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        # Extract Histogram of Oriented Gradients (HOG)
        features = hog(
            gray, 
            orientations=9, 
            pixels_per_cell=(8, 8), 
            cells_per_block=(2, 2), 
            visualize=False
        )
    else:
        # Fallback to normalized raw pixel flattening
        # Scaling to [0, 1] helps SVM optimization converge faster and better
        features = img_resized.flatten() / 255.0
        
    return features


def load_dataset(dataset_path, use_hog=True):
    """
    Safely scans the dataset directory, filters image formats, 
    and returns features (A) and labels (B).
    """
    images = []
    labels = []
    
    categories = {
        "cats": 0,  # Class 0
        "dogs": 1   # Class 1
    }
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset directory '{dataset_path}' not found. "
            "Please ensure your folders are structured correctly."
        )
        
    for category_name, label in categories.items():
        category_path = os.path.join(dataset_path, category_name)
        
        if not os.path.exists(category_path):
            print(f"Warning: Directory '{category_path}' not found. Skipping...")
            continue
            
        print(f"Loading {category_name}...")
        count = 0
        for filename in os.listdir(category_path):
            if filename.lower().endswith(valid_extensions):
                img_path = os.path.join(category_path, filename)
                img = cv2.imread(img_path)
                
                if img is not None:
                    features = extract_features(img, use_hog=use_hog)
                    images.append(features)
                    labels.append(label)
                    count += 1
                    
        print(f"Successfully loaded {count} images from '{category_name}'.")
        
    return np.array(images), np.array(labels)


def train_and_evaluate(dataset_path="dataset", use_hog=True):
    """
    Runs the full machine learning pipeline: Load -> Split -> Train -> Evaluate.
    """
    # 1. Load dataset
    try:
        A, B = load_dataset(dataset_path, use_hog=use_hog)
    except FileNotFoundError as e:
        print(e)
        return None
        
    if len(A) == 0:
        print("No images were loaded. Check your paths and try again.")
        return None
        
    # 2. Split into training and testing sets
    A_train, A_test, B_train, B_test = train_test_split(
        A, B, test_size=0.2, random_state=42, stratify=B
    )
    
    print(f"\nDataset loaded. Training shape: {A_train.shape}, Test shape: {A_test.shape}")
    
    # 3. Initialize and train SVM model
    # Using 'rbf' often yields better results for complex structures, 
    # but 'linear' is kept here as a reliable baseline.
    print("Training SVM classifier (this may take a moment depending on dataset size)...")
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    svm_model.fit(A_train, B_train)
    
    # 4. Predict and evaluate
    B_pred = svm_model.predict(A_test)
    accuracy = accuracy_score(B_test, B_pred)
    
    print("\n" + "="*40)
    print(f"Training Complete! Accuracy: {accuracy:.4f}")
    print("="*40)
    
    print("\nClassification Report:")
    print(classification_report(
        B_test,
        B_pred,
        target_names=["Cats", "Dogs"]
    ))
    
    return svm_model


def save_model(model, filename="svm_image_classifier.pkl"):
    """
    Saves the trained model to disk for future inference.
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel successfully saved to '{filename}'")


if __name__ == "__main__":
    # If scikit-image is installed, HOG features will be extracted automatically.
    # Otherwise, it falls back to normalized raw pixel flattening.
    if not HAS_SKIMAGE:
        print("Notice: 'scikit-image' library not found. Falling back to normalized raw pixel flattening.")
        print("To enable higher-accuracy HOG features, run: pip install scikit-image\n")
        
    # Train and evaluate the model
    model = train_and_evaluate(dataset_path="dataset", use_hog=True)
    
    # Save the model if training succeeded
    if model is not None:
        save_model(model)
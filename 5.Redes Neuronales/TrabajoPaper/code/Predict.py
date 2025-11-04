import librosa, warnings
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model

# Configure TensorFlow to run on CPU to avoid GPU issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

warnings.filterwarnings("ignore")

print("🚀 Starting Predict.py script...")
print(f"TensorFlow version: {tf.__version__}")
print("TensorFlow configured to use CPU only")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")

# Define file paths relative to the script location
extracted_df_path = os.path.join(script_dir, "extracted_df.pkl")
model1_path = os.path.join(script_dir, "Model1.h5")
model2_path = os.path.join(script_dir, "Model2.h5")
model3_path = os.path.join(script_dir, "Model3.h5")

# Check if files exist
print("Checking for required files...")
for file_path, name in [(extracted_df_path, "extracted_df.pkl"), 
                       (model1_path, "Model1.h5"), 
                       (model2_path, "Model2.h5"), 
                       (model3_path, "Model3.h5")]:
    if os.path.exists(file_path):
        print(f"✅ Found: {name}")
    else:
        print(f"❌ Missing: {name} at {file_path}")

# Load the files
print("Loading data and models...")
final = pd.read_pickle(extracted_df_path)
print(f"✅ Loaded extracted data: {final.shape}")

y = np.array(final["class"].tolist())
le = LabelEncoder()
le.fit_transform(y)
print(f"✅ Label encoder fitted with {len(np.unique(y))} classes")

print("Loading models...")
Model1_ANN = load_model(model1_path)
print("✅ ANN Model loaded")

Model2_CNN1D = load_model(model2_path)
print("✅ CNN1D Model loaded")

Model3_CNN2D = load_model(model3_path)
print("✅ CNN2D Model loaded")


def extract_feature(audio_path):
    audio_data, sample_rate = librosa.load(audio_path, res_type="kaiser_fast")
    feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=128)
    feature_scaled = np.mean(feature.T, axis=0)
    return np.array([feature_scaled])


def ANN_print_prediction(audio_path):
    prediction_feature = extract_feature(audio_path)
    predicted_vector = np.argmax(Model1_ANN.predict(prediction_feature), axis=-1)
    predicted_class = le.inverse_transform(predicted_vector)
    return predicted_class[0]


def CNN1D_print_prediction(audio_path):
    tmp = extract_feature(audio_path)
    prediction_feature = np.expand_dims(tmp, axis=2)
    predicted_vector = np.argmax(Model2_CNN1D.predict(prediction_feature), axis=-1)
    predicted_class = le.inverse_transform(predicted_vector)
    return predicted_class[0]


def CNN2D_print_prediction(audio_path):
    tmp2 = extract_feature(audio_path)
    prediction_feature = tmp2.reshape(tmp2.shape[0], 16, 8, 1)
    predicted_vector = np.argmax(Model3_CNN2D.predict(prediction_feature), axis=-1)
    predicted_class = le.inverse_transform(predicted_vector)
    return predicted_class[0]


audio_dataset_path = "C:\\Users\\Rodri\\Documents\\VS Workspace\\UrbanSound8K\\"
path = audio_dataset_path + "fold8/103076-3-0-0.wav"

print("🎵 Testing audio predictions...")
print(f"Audio file path: {path}")
print(f"Audio file exists: {os.path.exists(path)}")

if os.path.exists(path):
    print("\nANN Model Output --> ", ANN_print_prediction(path))
    print("\nCNN1D Model Output --> ", CNN1D_print_prediction(path))
    print("\nCNN2D Model Output --> ", CNN2D_print_prediction(path))
else:
    print("❌ Audio file not found. Please check the path.")
    print("Available alternatives:")
    
    # Try to find any audio files in common locations
    test_paths = [
        "C:\\Users\\Rodri\\Documents\\VS Workspace\\UrbanSound8K\\fold1\\",
        "C:\\Users\\Rodri\\Documents\\VS Workspace\\UrbanSound8K\\audio\\",
        os.path.join(script_dir, "test_audio.wav"),
        os.path.join(script_dir, "sample.wav")
    ]
    
    for test_path in test_paths:
        if os.path.exists(test_path):
            print(f"✅ Found directory: {test_path}")
            if os.path.isdir(test_path):
                files = [f for f in os.listdir(test_path) if f.endswith('.wav')][:3]
                print(f"   Sample files: {files}")
        else:
            print(f"❌ Not found: {test_path}")
    
    print("\n💡 To test the models, place an audio file in the assets directory or update the path.")

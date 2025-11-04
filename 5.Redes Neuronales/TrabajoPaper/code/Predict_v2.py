#!/usr/bin/env python3
"""
Optimized Audio Prediction Script
Loads models and makes predictions on audio files
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Configure environment before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

import tensorflow as tf
from tensorflow.keras.models import load_model

# Try to import librosa, with fallback if not available
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️  librosa not available. Audio processing will be limited.")

warnings.filterwarnings("ignore")

class AudioPredictor:
    """Class to handle audio prediction using trained models"""
    
    def __init__(self, assets_dir=None):
        """Initialize the predictor with model paths"""
        if assets_dir is None:
            assets_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.assets_dir = assets_dir
        self.models_loaded = False
        self.le = None
        
        # File paths
        self.paths = {
            'data': os.path.join(assets_dir, 'extracted_df.pkl'),
            'ann': os.path.join(assets_dir, 'Model1.h5'),
            'cnn1d': os.path.join(assets_dir, 'Model2.h5'),
            'cnn2d': os.path.join(assets_dir, 'Model3.h5')
        }
        
        print(f"🤖 AudioPredictor initialized")
        print(f"📁 Assets directory: {assets_dir}")
    
    def load_models(self):
        """Load all models and data"""
        try:
            print("📊 Loading data...")
            final = pd.read_pickle(self.paths['data'])
            print(f"✅ Data loaded: {final.shape}")
            
            # Setup label encoder
            y = np.array(final["class"].tolist())
            self.le = LabelEncoder()
            self.le.fit_transform(y)
            print(f"✅ Label encoder fitted with {len(np.unique(y))} classes")
            
            print("🧠 Loading models...")
            self.model_ann = load_model(self.paths['ann'])
            print("✅ ANN Model loaded")
            
            self.model_cnn1d = load_model(self.paths['cnn1d'])
            print("✅ CNN1D Model loaded")
            
            self.model_cnn2d = load_model(self.paths['cnn2d'])
            print("✅ CNN2D Model loaded")
            
            self.models_loaded = True
            print("🎉 All models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
        
        return True
    
    def extract_feature(self, audio_path):
        """Extract MFCC features from audio file"""
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required for audio processing")
        
        try:
            audio_data, sample_rate = librosa.load(audio_path, res_type="kaiser_fast")
            feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=128)
            feature_scaled = np.mean(feature.T, axis=0)
            return np.array([feature_scaled])
        except Exception as e:
            print(f"❌ Error extracting features from {audio_path}: {e}")
            return None
    
    def predict_ann(self, audio_path):
        """Make prediction using ANN model"""
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        features = self.extract_feature(audio_path)
        if features is None:
            return None
        
        predicted_vector = np.argmax(self.model_ann.predict(features, verbose=0), axis=-1)
        predicted_class = self.le.inverse_transform(predicted_vector)
        return predicted_class[0]
    
    def predict_cnn1d(self, audio_path):
        """Make prediction using CNN1D model"""
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        features = self.extract_feature(audio_path)
        if features is None:
            return None
        
        features_3d = np.expand_dims(features, axis=2)
        predicted_vector = np.argmax(self.model_cnn1d.predict(features_3d, verbose=0), axis=-1)
        predicted_class = self.le.inverse_transform(predicted_vector)
        return predicted_class[0]
    
    def predict_cnn2d(self, audio_path):
        """Make prediction using CNN2D model"""
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        features = self.extract_feature(audio_path)
        if features is None:
            return None
        
        features_2d = features.reshape(features.shape[0], 16, 8, 1)
        predicted_vector = np.argmax(self.model_cnn2d.predict(features_2d, verbose=0), axis=-1)
        predicted_class = self.le.inverse_transform(predicted_vector)
        return predicted_class[0]
    
    def predict_all(self, audio_path):
        """Make predictions using all models"""
        if not os.path.exists(audio_path):
            print(f"❌ Audio file not found: {audio_path}")
            return None
        
        print(f"🎵 Analyzing: {os.path.basename(audio_path)}")
        
        results = {}
        try:
            results['ANN'] = self.predict_ann(audio_path)
            results['CNN1D'] = self.predict_cnn1d(audio_path)
            results['CNN2D'] = self.predict_cnn2d(audio_path)
        except Exception as e:
            print(f"❌ Error making predictions: {e}")
            return None
        
        return results

def main():
    """Main function to run predictions"""
    print("=" * 60)
    print("AUDIO PREDICTION SYSTEM")
    print("=" * 60)
    
    # Initialize predictor
    predictor = AudioPredictor()
    
    # Load models
    if not predictor.load_models():
        print("❌ Failed to load models. Exiting.")
        return
    
    # Test audio file path
    audio_dataset_path = "C:\\Users\\Rodri\\Documents\\VS Workspace\\UrbanSound8K\\"
    test_audio_path = audio_dataset_path + "fold8/103076-3-0-0.wav"
    
    print(f"\n🎵 Testing with audio file:")
    print(f"Path: {test_audio_path}")
    print(f"Exists: {os.path.exists(test_audio_path)}")
    
    if os.path.exists(test_audio_path):
        print("\n🚀 Making predictions...")
        results = predictor.predict_all(test_audio_path)
        
        if results:
            print("\n📊 PREDICTION RESULTS:")
            print("-" * 40)
            for model_name, prediction in results.items():
                print(f"{model_name:>8} Model: {prediction}")
        else:
            print("❌ Prediction failed")
    else:
        print("\n⚠️  Test audio file not found.")
        print("💡 To test predictions:")
        print("   1. Place an audio file in the assets directory, or")
        print("   2. Update the audio path in the script")
        
        # Look for any WAV files in the assets directory
        assets_files = [f for f in os.listdir(predictor.assets_dir) if f.lower().endswith('.wav')]
        if assets_files:
            print(f"\n🎵 Found audio files in assets directory: {assets_files}")
            test_file = os.path.join(predictor.assets_dir, assets_files[0])
            print(f"Testing with: {test_file}")
            
            results = predictor.predict_all(test_file)
            if results:
                print("\n📊 PREDICTION RESULTS:")
                print("-" * 40)
                for model_name, prediction in results.items():
                    print(f"{model_name:>8} Model: {prediction}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
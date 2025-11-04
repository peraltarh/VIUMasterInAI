#!/usr/bin/env python3
"""
Simple test script to check if required files can be loaded
"""

import os
import pandas as pd

def test_file_loading():
    """Test if all required files can be found and loaded"""
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")
    
    # Define file paths
    files_to_check = {
        "extracted_df.pkl": "Data file",
        "Model1.h5": "ANN Model",
        "Model2.h5": "CNN1D Model", 
        "Model3.h5": "CNN2D Model"
    }
    
    print("\n🔍 Checking for required files...")
    all_files_found = True
    
    for filename, description in files_to_check.items():
        file_path = os.path.join(script_dir, filename)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"✅ {description}: {filename} ({file_size:.1f} MB)")
        else:
            print(f"❌ {description}: {filename} - NOT FOUND")
            all_files_found = False
    
    if all_files_found:
        print("\n🎉 All files found! Testing pickle loading...")
        try:
            extracted_df_path = os.path.join(script_dir, "extracted_df.pkl")
            final = pd.read_pickle(extracted_df_path)
            print(f"✅ Pickle file loaded successfully!")
            print(f"   Data shape: {final.shape}")
            print(f"   Columns: {list(final.columns)}")
            if 'class' in final.columns:
                unique_classes = final['class'].unique()
                print(f"   Classes: {len(unique_classes)} unique values")
                print(f"   Sample classes: {list(unique_classes[:5])}")
        except Exception as e:
            print(f"❌ Error loading pickle file: {e}")
    else:
        print("\n❌ Some files are missing. Please ensure all model files are in the assets directory.")
    
    return all_files_found

if __name__ == "__main__":
    print("=" * 60)
    print("AUDIO PREDICTION FILES TEST")
    print("=" * 60)
    
    success = test_file_loading()
    
    if success:
        print("\n✅ File loading test PASSED!")
        print("The Predict.py script should work correctly.")
    else:
        print("\n❌ File loading test FAILED!")
        print("Please fix the missing files before running Predict.py")
    
    print("=" * 60)
from real_time_pipeline import RealTimeVerificationSystem
import os

def main():
    """Main function"""
    # Configuration
    MODEL_WEIGHTS_PATH = './mobilenetv2_final_model.weights.h5'
    REFERENCE_FILE = './real_time_path.txt'  # Single image per person
    CAMERA_ID = 0
    SIMILARITY_THRESHOLD = 0.8
    
    # Check if model exists
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Model weights not found at {MODEL_WEIGHTS_PATH}")
        print("Please ensure you have the trained model weights file")
        return
    
    # Check if reference file exists
    if not os.path.exists(REFERENCE_FILE):
        print(f"Error: Reference file not found at {REFERENCE_FILE}")
        return
    
    # Create and run system
    system = RealTimeVerificationSystem(
        model_weights_path=MODEL_WEIGHTS_PATH,
        reference_file=REFERENCE_FILE,
        camera_id=CAMERA_ID,
        similarity_threshold=SIMILARITY_THRESHOLD
    )
    
    system.run()

if __name__ == "__main__":
    main()
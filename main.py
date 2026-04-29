import argparse
import os
from audio_handler import AudioInputHandler
from feature_extractor import FeatureExtractor
from model_inference import ModelInference
from output_processing import OutputProcessor

def run_pipeline(audio_mode, file_path=None, record_duration=5):
    print("\n==================================================")
    print("       SPEECH-TO-TEXT PIPELINE INITIATED       ")
    print("==================================================")
    
    # -------------------------------------------------------------
    # Step 1: Audio Input
    # -------------------------------------------------------------
    print("\n[Step 1] Loading Audio...")
    audio_handler = AudioInputHandler()
    if audio_mode == 'file':
        if not file_path or not os.path.exists(file_path):
            print(f"[ERROR] Error: File not found: {file_path}")
            return
        audio, sr = audio_handler.load_from_file(file_path)
    else:
        audio, sr = audio_handler.record_from_microphone(duration=record_duration)
        
    if audio is None:
        print("[ERROR] Pipeline aborted at Step 1: Audio handling failed.")
        return
        
    # -------------------------------------------------------------
    # Step 2: Feature Extraction
    # -------------------------------------------------------------
    print("\n[Step 2] Extracting Features...")
    extractor = FeatureExtractor()
    features = extractor.extract(audio, sampling_rate=sr)
    
    if features is None:
        print("[ERROR] Pipeline aborted at Step 2: Feature extraction failed.")
        return
        
    # -------------------------------------------------------------
    # Step 3: Model Inference (Generate Token IDs)
    # -------------------------------------------------------------
    print("\n[Step 3] Running Model Inference...")
    inference_engine = ModelInference()
    
    # We will directly run generation to pass to Output processing
    generated_ids = inference_engine.generate_tokens(features)
    
    if generated_ids is None:
        print("[ERROR] Pipeline aborted at Step 3: Model inference failed.")
        return
        
    # -------------------------------------------------------------
    # Step 4: Output Processing
    # -------------------------------------------------------------
    print("\n[Step 4] Processing Model Output...")
    output_engine = OutputProcessor(processor=inference_engine.processor)
    
    final_text = output_engine.decode_and_clean(
        generated_ids, 
        save_to_file=True, 
        output_filename="final_transcription.txt"
    )
    
    # -------------------------------------------------------------
    # Results
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("               FINAL TRANSCRIPTION")
    print("="*50)
    print(f"\n--> {final_text}\n")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full Speech-to-Text Engine Pipeline.")
    parser.add_argument('--mode', choices=['mic', 'file'], default='mic', 
                        help="Input mode: 'mic' to record from microphone, 'file' to load an audio file.")
    parser.add_argument('--file', type=str, default=None, 
                        help="Path to audio file (required if mode is 'file').")
    parser.add_argument('--duration', type=int, default=5, 
                        help="Recording duration in seconds (if mode is 'mic').")
    
    args = parser.parse_args()
    
    run_pipeline(audio_mode=args.mode, file_path=args.file, record_duration=args.duration)

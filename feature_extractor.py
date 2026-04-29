import torch
from transformers import Speech2TextProcessor
import numpy as np
import warnings

# Suppressing extraneous Hugging Face future warnings to keep terminal readable
warnings.simplefilter('ignore')

class FeatureExtractor:
    def __init__(self, model_id="facebook/s2t-small-librispeech-asr"):
        """Initializes the processor capable of logging mel-filter matrices."""
        print(f"Loading Processor mapping for: {model_id}...")
        
        # Loads the feature extracting parameters directly tied to this exact model!
        self.processor = Speech2TextProcessor.from_pretrained(model_id)
        print("[OK] Processor successfully loaded and attached.")
        
    def extract(self, audio_array, sampling_rate=16000):
        """Converts raw audio numpy array into model-ready log-mel feature dictionary."""
        print("\nExtracting log-mel features mathematically...")
        
        # Enforce explicitly that the input parameter is indeed a Numpy array (Module 2 output)
        if not isinstance(audio_array, np.ndarray):
             print(f"[ERROR] Error: Expected numpy array passed from Mod 2, but got: {type(audio_array)}")
             return None
        
        try:
            # We strictly configure `return_tensors="pt"` telling hugging face to convert array features 
            # squarely into PyTorch Tensors natively instead of generic python lists!
            inputs = self.processor(audio_array, 
                                    sampling_rate=sampling_rate, 
                                    return_tensors="pt")
            
            print("[OK] Feature Extraction Successful!")
            print(f"   Input Features Tensor Shape: {inputs.input_features.shape}")
            if 'attention_mask' in inputs:
                 print(f"   Attention Mask Shape: {inputs.attention_mask.shape}")
            
            return inputs
        except Exception as e:
            print(f"[ERROR] Core feature extraction failed mathematically: {e}")
            return None

# Local Debug Testing System
if __name__ == "__main__":
    print("Welcome to Module 3 Engine Component System!")
    
    # 1. We artificially generate exactly 2 seconds of purely fake, static audio mapped exactly at 16000Hz 
    # (32000 points represent 2 seconds worth of data)
    print("\nGenerating simulated static audio data (2 seconds)...")
    dummy_audio = np.random.randn(32000).astype(np.float32) 
    
    # 2. We initialize our brand new Extractor class architecture and map features
    extractor = FeatureExtractor()
    model_inputs = extractor.extract(dummy_audio, sampling_rate=16000)
    
    # 3. Present outputs verifying structural integrity
    if model_inputs is not None:
        print(f"\n[OUTPUT] Final Generated Model Dictionary Keys: ", " | ".join(model_inputs.keys()))
        print("These parameters are precisely configured and ready to be loaded by the AI Model for Module 4!")

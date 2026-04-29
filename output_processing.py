import warnings

# Suppressing extraneous Hugging Face future warnings to keep terminal readable
warnings.simplefilter('ignore')

class OutputProcessor:
    def __init__(self, processor):
        """
        Initializes the Output Processor.
        Takes an already loaded Hugging Face Processor object to handle decoding.
        """
        print("Initializing the Output Processing Module...")
        self.processor = processor
        print("[OK] Output Processor initialized.")

    def decode_and_clean(self, generated_ids, save_to_file=True, output_filename="output.txt"):
        """
        Takes mathematical generated_ids array outputted by the model inference phase
        and precisely decodes them back to a raw, human readable English string.
        """
        print("\nDecoding Model Outputs back to Text...")
        
        # Validate that the generated IDs are present
        if generated_ids is None:
            print("[ERROR] Error: Missing 'generated_ids' parameter.")
            return None
            
        try:
            # 1. Decoding Step: Convert resulting Token ID sequence backward into English string
            # while filtering out internal structural tokens (e.g. padding and eos tokens).
            # The result is technically a list.
            transcription_list = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # 2. Extract final literal string from the returned python list wrapper
            raw_text = transcription_list[0]
            
            # --- 3. Output Cleaning Enhancements ---
            
            # Strip unnecessary extra whitespace from the trailing edges
            clean_text = raw_text.strip()
            
            # Ensure the first character is fully capitalized
            if len(clean_text) > 0:
                clean_text = clean_text.capitalize()
            
            print("[OK] Decoding Complete!")
            
            # 4. Optional: Save to local filesystem
            if save_to_file:
                try:
                    with open(output_filename, "w", encoding='utf-8') as my_file:
                        my_file.write(clean_text)
                    print(f"[OK] Text properly saved to: {output_filename}")
                except Exception as file_e:
                     print(f"[WARN] Warning: Could not save to filesystem. {file_e}")
                     
            return clean_text
            
        except Exception as e:
            print(f"[ERROR] Output processing and decoding failed: {e}")
            return None

# Local Debug Testing System
if __name__ == "__main__":
    print("Welcome to Module 5 Engine Component System!")
    
    try:
        from transformers import Speech2TextProcessor
        import torch
        
        print("\nSimulating the environment from prior Modules...")
        
        # Initialize a basic processor purely so we have a tokenizer dictionary to use
        # (Usually this carries over transparently from Module 3)
        dummy_processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
        
        # 1. Simulate the Token IDs arriving cleanly from the Model (Module 4 limit)
        # We mathematically spoof an array mapping out token values:
        # 2: </s> (Should be skipped!)
        # 452, 120, 290 : random IDs evaluating essentially to 'hello world' depending on dictionary!
        print("\nGenerating simulated Token IDs...")
        fake_generated_ids = torch.tensor([[2, 452, 120, 230, 2]])
        
        # 2. Initialize our clean Output structure mapped cleanly against our architecture
        output_engine = OutputProcessor(processor=dummy_processor)
        
        # 3. Process the raw IDs structurally back into an active text string
        final_text_output = output_engine.decode_and_clean(
            generated_ids=fake_generated_ids, 
            save_to_file=True,
            output_filename="output.txt"
        )
        
        # 4. Verify Final Output Output Validation Format
        if final_text_output is not None:
             print("\n[OUTPUT] Final Transcribed Text: ")
             print(f"--> '{final_text_output}'")
             
    except Exception as general_e:
        print(f"[ERROR] System evaluation crashed internally: {general_e}")

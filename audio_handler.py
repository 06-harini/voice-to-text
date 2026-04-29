import numpy as np
import librosa
import sounddevice as sd

import subprocess
import tempfile
import imageio_ffmpeg
import os

class AudioInputHandler:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr

    def load_from_file(self, file_path):
        """Loads an audio file and resamples it to 16000 Hz Mono."""
        try:
            # First attempt standard librosa loading
            audio, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
            self._validate_audio(audio, sr, "File Input")
            return audio, sr
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return None, None
        except Exception as e:
            print(f"Native librosa fallback skipped (Error: {e}).")
            print("Attempting robust FFMPEG decoding mapping to fix corrupted headers...")
            
            try:
                # 1. Grab our dynamically installed ffmpeg executable
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                
                # 2. Build a local safe temp file
                temp_wav = tempfile.mktemp(suffix=".wav")
                
                # 3. Use raw ffmpeg to forcefully rip the audio stream to a perfectly clean WAV mapping
                subprocess.run(
                    [
                        ffmpeg_exe, "-y", "-i", file_path, 
                        "-vn",                  # Ignore any video streams
                        "-acodec", "pcm_s16le", # Safe generic WAV mapping
                        "-ar", str(self.target_sr), # Match our target SR
                        "-ac", "1",             # Force Mono
                        temp_wav
                    ], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL,
                    check=True
                )
                
                # 4. Read the perfectly clean file using librosa
                audio, sr = librosa.load(temp_wav, sr=self.target_sr, mono=True)
                os.remove(temp_wav) # Ensure cleanup
                
                self._validate_audio(audio, sr, "FFMPEG Cleaned File Input")
                return audio, sr
                
            except Exception as ffmpeg_e:
                print(f"Error: Final robust loading pipeline structurally failed: {ffmpeg_e}")
                print("The file may be entirely broken or contain no real media.")
                return None, None

    def record_from_microphone(self, duration=5):
        """Records audio from the microphone for a given duration (seconds)."""
        print(f"\n🎤 Recording for {duration} seconds... Please speak now.")
        try:
            # Record using float32 numpy typing at 1 channel and targeted sampling rate
            audio = sd.rec(int(duration * self.target_sr), 
                           samplerate=self.target_sr, 
                           channels=1, 
                           dtype='float32')
            sd.wait() # Block python script until user stops speaking
            print("[OK] Recording finished.")
            
            # sounddevice inherently outputs a nested 2D array, flatten is for standard 1D Mono
            audio = audio.flatten()
            
            self._validate_audio(audio, self.target_sr, "Microphone Input")
            return audio, self.target_sr
        except Exception as e:
            print(f"[ERROR] Microphone recording failed: {e}")
            return None, None

    def _validate_audio(self, audio, sr, source_name):
        """Validates the audio array shape and sampling rate strictly."""
        print(f"\n--- {source_name} Validation ---")
        print(f"[OK] Sampling Rate: {sr} Hz")
        print(f"[OK] Audio Shape: {audio.shape}")
        
        if sr != 16000:
            print("[WARN] Warning: Sampling rate is not explicitly set to 16000 Hz!")
        
        if len(audio.shape) != 1:
             print("[WARN] Warning: Audio structure is not a flattened 1D Mono array!")
        else:
             print("[OK] Audio formatted correctly! Returning (audio, 16000) tuple for STT Model.")
        print("-" * 35)


if __name__ == "__main__":
    handler = AudioInputHandler()
    
    print("Welcome to Module 2 Runtime Check!")
    
    # 1. Test live Microphone (uncomment below to test)
    audio_data, sampling_rate = handler.record_from_microphone(duration=5)
    
    # 2. Test File loading (ensure test_audio.wav exists first beforehand)
    # audio_data, sampling_rate = handler.load_from_file("test_audio.wav")

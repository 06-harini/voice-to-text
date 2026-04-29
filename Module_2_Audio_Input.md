# Module 2: Audio Input Handling

Welcome to the second module! Before we can perform Speech-to-Text (STT) inference, our model needs to "listen" to the audio. However, neural networks cannot process raw `.wav` or `.mp3` files directly. We must first load and convert these sounds into numerical arrays.

In this module, you'll learn how to load audio from files and record real-time audio through your microphone.

## 1. How Audio Becomes Numbers

Sound is primarily air pressure changes over time. A microphone measures these continuous changes and converts them into an electrical signal. 
*   **Sampling:** A computer grabs a specific number of snapshots (samples) of this continuous signal every second. This count is known as the **Sampling Rate**.
*   **Digital Form:** Every snapshot is stored as a floating-point number, representing the amplitude (loudness) at that exact moment in time. By the end, a 1-second audio clip simply becomes an array (list) of these numerical float values!

## 2. Why 16kHz?

Most modern pre-trained STT models (like Hugging Face's Whisper or Wav2Vec2) strictly expect the incoming audio to have a sampling rate of **16,000 Hertz (16 kHz)**.
*   **The Science:** Human speech primarily falls within the 0 to 8kHz frequency range. According to the Nyquist-Shannon sampling theorem, systematically capturing frequencies up to 8kHz requires a sampling rate exactly twice as high: 16kHz.
*   **The Practicality:** Sampling above 16kHz provides very little extra speech clarity while significantly increasing the compute processing time and memory required.

## 3. Strict Audio Requirements
For smooth integration with our future Hugging Face models, your audio configuration **must** be:
*   **Sampling Rate:** 16,000 Hz
*   **Channels:** Mono (Single-channel)
*   **Format:** We prefer WAV as it's an uncompressed, lossless format, avoiding encoding artifacts.

## 4. Install Additional Dependencies
To record directly from your microphone over the audio pipeline, you need the `sounddevice` library. Ensure your virtual environment is activated, then run:

```bash
pip install sounddevice numpy
```

---

## 5. Audio Handling Implementations

We will build an `AudioInputHandler` class. It successfully manages two workflows: Loading a File, and Recording from a Microphone.

### Method A: Loading from an Audio File
We use the `librosa` library. `librosa.load` is incredibly useful because it automatically handles resampling (changing the rate to 16kHz) and downmixing (converting Stereo to Mono) in one simple go!

```python
import librosa

def load_from_file(file_path, target_sr=16000):
    # librosa automatically converts to mono if we use `mono=True`
    # It also dynamically resamples to exactly `target_sr`
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr
```

### Method B: Recording Real-time from Microphone
We use `sounddevice` coupled with `numpy` to capture a live array stream.

```python
import sounddevice as sd

def record_from_microphone(duration=5, target_sr=16000):
    print(f"🎤 Recording for {duration} seconds... Please speak now.")
    
    # shape will initially be (samples, channels)
    audio = sd.rec(int(duration * target_sr), 
                   samplerate=target_sr, 
                   channels=1, 
                   dtype='float32')
    
    sd.wait() # Block execution until recording completely finishes
    
    # Flatten the 2D array output into a 1D sequence suitable for Mono STT audio
    audio = audio.flatten()
    print("✅ Recording finished.")
    return audio, target_sr
```

---

## 6. Audio Validation Script & Expected Output

The final part of Module 2 is validating our array structure before passing it onto our model. 
*   **Shape Output:** It should precisely look like `(80000,)` for a 5-second 16kHz recording `(16000 samples * 5 seconds)`.
*   **Sampling Rate Output:** Let's verify it is explicitly maintained at 16000.

Here is the full working script consolidating everything, you can refer to `audio_handler.py` in your folder!

```python
import numpy as np
import librosa
import sounddevice as sd

class AudioInputHandler:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr

    def load_from_file(self, file_path):
        """Loads an audio file and resamples it to 16000 Hz Mono."""
        try:
            audio, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
            self._validate_audio(audio, sr, "File Input")
            return audio, sr
        except FileNotFoundError:
            print(f"❌ Error: File '{file_path}' not found.")
            return None, None
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return None, None

    def record_from_microphone(self, duration=5):
        """Records audio from the microphone for a given duration (seconds)."""
        print(f"🎤 Recording for {duration} seconds... Please speak now.")
        try:
            audio = sd.rec(int(duration * self.target_sr), 
                           samplerate=self.target_sr, 
                           channels=1, 
                           dtype='float32')
            sd.wait()
            print("✅ Recording finished.")
            
            # sounddevice returns 2D array, we flatten to 1D mono
            audio = audio.flatten()
            
            self._validate_audio(audio, self.target_sr, "Microphone Input")
            return audio, self.target_sr
        except Exception as e:
            print(f"❌ Microphone recording failed: {e}")
            return None, None

    def _validate_audio(self, audio, sr, source_name):
        """Validates the audio array shape and sampling rate."""
        print(f"\n--- {source_name} Validation ---")
        print(f"✅ Sampling Rate: {sr} Hz")
        print(f"✅ Audio Shape: {audio.shape}")
        
        if sr != 16000:
            print("⚠️ Warning: Sampling rate is not exactly 16000 Hz!")
        
        if len(audio.shape) != 1:
             print("⚠️ Warning: Audio is not a 1D Mono array!")
        else:
             print("✅ Audio formatting is correct and ready for the model.")
        print("-" * 35 + "\n")


if __name__ == "__main__":
    handler = AudioInputHandler()
    
    # Test Microphone Action
    audio_mic, sr_mic = handler.record_from_microphone(duration=5)
    
    # Expected final runtime outputs completely readied for Module 3 Feature Extraction:
    # `audio_mic` -> represents our output (numpy array)
    # `sr_mic`    -> represents our explicit (16000) integer sampling rate
```

---

## 7. Common Errors and Fixes

### Wrong Sampling Rate
*   **Symptom:** AI output is gibberish, hallucinates text, or you get model-level shape length mismatches.
*   **Fix:** Ensure you are using `librosa.load(..., sr=16000)` properly or specifying `samplerate=16000` in sounddevice. **Never assume** an unknown file is 16kHz normally. Let python explicitly compress or expand it statically.

### Stereo to Mono Conversion 
*   **Symptom:** Arrays have an extra dimension `(samples, 2)` instead of `(samples,)`. The pipeline will crash during inference.
*   **Fix:** `librosa` will auto-convert stereo if you put `mono=True`. For microphone input, always set `channels=1`, then `.flatten()` the array into a native 1D python list.

### File Not Found
*   **Symptom:** `FileNotFoundError: No such file or directory`.
*   **Fix:** Ensure your WAV file is physically placed inside the exact directory your script environment is running from, or provide an absolute path string via `"C:/absolute/file/folder/audio.wav"`.

### Unsupported Formats
*   **Symptom:** Decoding warnings or empty array loads (often seen with older `.m4a` files or missing lower system libraries).
*   **Fix:** Prioritize standardizing `.wav` formats using tools like `ffmpeg` or `audacity` to clean convert your data beforehand if `librosa` gets stuck loading them!

---
**Next Steps:** You are now correctly capturing speech waves and safely outputting standardized `(audio, sampling_rate)` tuples. You are fully geared up and ready to safely pass this tuple onto the model tokenizer in **Module 3 (Feature Extraction)**!

# Module 1: Setup Environment

Welcome to the Speech-to-Text project using Python and Hugging Face Transformers. This module will guide you through setting up your development environment, ensuring all necessary tools and libraries are installed for robust audio processing and deep learning.

## 1. Prerequisites
Ensure you have Python installed on your system. This project requires **Python 3.9 to 3.11**.
You can verify your Python version by running:
```bash
python --version
```

## 2. Virtual Environment Setup
It is highly recommended to use a virtual environment to isolate the project dependencies from other Python projects.

### Create the Virtual Environment
Navigate to your project directory in the terminal and run:
```bash
python -m venv venv
```

### Activate the Virtual Environment
Activate the environment to start using it.

**For Windows:**
```powershell
venv\Scripts\activate
```

**For Linux/Mac:**
```bash
source venv/bin/activate
```
*(You should now see `(venv)` at the beginning of your terminal prompt.)*

## 3. Install Required Libraries

We need several powerful deep learning and audio processing libraries. Run the following command in your activated virtual environment:

```bash
pip install torch torchaudio transformers datasets sentencepiece librosa
```

> **Note for Linux / Ubuntu users:** `librosa` requires an underlying system dependency for reading and decoding audio files. Install `libsndfile` by running:
> ```bash
> sudo apt-get update
> sudo apt-get install libsndfile1
> ```

## 4. Understanding the Libraries

Here is what each installed library does:

*   **`torch` (PyTorch)**: The core deep learning framework. It powers the underlying neural network computations, allowing us to train and run models efficiently.
*   **`transformers`**: Hugging Face's library that provides state-of-the-art pre-trained models (like Whisper or Wav2Vec2) for Speech-to-Text processing.
*   **`torchaudio`**: PyTorch's audio processing library. It helps load audio data as tensors and perform audio-specific transformations.
*   **`sentencepiece`**: A fast tokenizer library used by modern Transformer models to convert transcribed text into vocabulary tokens and vice-versa.
*   **`librosa`**: An easy-to-use library for audio analysis. It's frequently used for loading various audio formats, sample rate conversions, and extracting features.
*   **`datasets`**: Provides easy access to vast audio and text datasets, as well as efficient loading and data preprocessing mechanisms.

## 5. System Verification

Let's ensure everything is correctly installed, including checking if your system has a GPU available to accelerate model inference.

Run the provided verification script `verify_setup.py`:

```bash
python verify_setup.py
```

*If you do not have the script downloaded, you can create a file named `verify_setup.py` with the following code:*

```python
import sys

def verify_installation():
    missing_libs = []
    
    try:
        import torch
        print("✅ torch installed successfully.")
        print(f"   GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        missing_libs.append("torch")
        print("❌ torch is NOT installed.")

    try:
        import transformers
        print("✅ transformers installed successfully.")
    except ImportError:
        missing_libs.append("transformers")
        print("❌ transformers is NOT installed.")

    try:
        import datasets
        print("✅ datasets installed successfully.")
    except ImportError:
        missing_libs.append("datasets")
        print("❌ datasets is NOT installed.")

    try:
        import torchaudio
        print("✅ torchaudio installed successfully.")
    except ImportError:
        missing_libs.append("torchaudio")
        print("❌ torchaudio is NOT installed.")

    try:
        import sentencepiece
        print("✅ sentencepiece installed successfully.")
    except ImportError:
        missing_libs.append("sentencepiece")
        print("❌ sentencepiece is NOT installed.")

    try:
        import librosa
        print("✅ librosa installed successfully.")
    except ImportError:
        missing_libs.append("librosa")
        print("❌ librosa is NOT installed.")

    print("-" * 50)
    if not missing_libs:
        print("🎉 Success! All required libraries are installed and working.")
    else:
        print(f"⚠️ Missing libraries: {', '.join(missing_libs)}")
        print("Please review the installation instructions to fix these issues.")

if __name__ == "__main__":
    verify_installation()
```

## 6. Common Errors & Fixes

Here are a few common issues you might encounter during setup:

### `ModuleNotFoundError: No module named '...'`
**What it means:** The library isn't installed in your active environment.
**Fix:** Ensure your virtual environment is activated BEFORE installing packages (you should see `(venv)` in the terminal). Re-run the `pip install` command.

### `ImportError: sentencepiece is not installed`
**What it means:** Some transformer models require this explicitly, but it occasionally fails to auto-install as a dependency.
**Fix:** Explicitly run `pip install sentencepiece`. If you are on Windows, you may encounter build errors requiring the Microsoft C++ Build Tools to be installed.

### Audio Loading Issues (e.g., `UserWarning: PySoundFile failed. Trying audioread instead.`)
**What it means:** You are missing the system-level libraries needed to decode audio files. 
**Fix:** This is most common on Linux. Make sure you installed `libsndfile1` (as noted in step 3). You can also try installing the additional Python wrapper: `pip install soundfile`.

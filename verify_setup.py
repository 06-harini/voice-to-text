import sys

def verify_installation():
    missing_libs = []
    
    print("Beginning Environment Setup Verification...\n")
    print("-" * 50)
    
    try:
        import torch
        print("[OK] torch installed successfully.")
        print(f"   GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        missing_libs.append("torch")
        print("[ERROR] torch is NOT installed.")

    try:
        import transformers
        print("[OK] transformers installed successfully.")
    except ImportError:
        missing_libs.append("transformers")
        print("[ERROR] transformers is NOT installed.")

    try:
        import datasets
        print("[OK] datasets installed successfully.")
    except ImportError:
        missing_libs.append("datasets")
        print("[ERROR] datasets is NOT installed.")

    try:
        import torchaudio
        print("[OK] torchaudio installed successfully.")
    except ImportError:
        missing_libs.append("torchaudio")
        print("[ERROR] torchaudio is NOT installed.")

    try:
        import sentencepiece
        print("[OK] sentencepiece installed successfully.")
    except ImportError:
        missing_libs.append("sentencepiece")
        print("[ERROR] sentencepiece is NOT installed.")

    try:
        import librosa
        print("[OK] librosa installed successfully.")
    except ImportError:
        missing_libs.append("librosa")
        print("[ERROR] librosa is NOT installed.")

    print("-" * 50)
    if not missing_libs:
        print("🎉 Success! All required libraries are installed and working.")
    else:
        print(f"[WARN] Missing libraries: {', '.join(missing_libs)}")
        print("Please review the Module 1 setup guide to fix these issues.")

if __name__ == "__main__":
    verify_installation()

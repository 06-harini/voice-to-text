# Speech-to-Text with PyTorch and Flask

## 1. Project Title & Description
**Speech-to-Text Pipeline** is a machine learning project designed to convert speech audio into text. It leverages a powerful transformer-based model (`facebook/s2t-small-librispeech-asr`) powered by PyTorch(deepa learning library) and Hugging Face Transformers. The project also features a fully-functional REST API built with **Flask**, allowing easy deployment and integration with web or mobile frontends.

## 2. Features
- **Speech-to-text conversion**: High accuracy transcription using state-of-the-art transformer models.
- **File upload support**: Easily transcribe existing `.wav` audio files.
- **Microphone input support**: Record and transcribe live audio locally from the terminal.
- **REST API using Flask**: A lightweight web server to serve model predictions.
- **JSON responses**: Clean JSON outputs with the transcribed text for easy frontend parsing.

## 3. Requirements / Installation

### Prerequisites
- Python version: **3.9 – 3.11**

### Installation Steps

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **Mac/Linux:**
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies using pip:**
   ```bash
   pip install torch transformers datasets torchaudio sentencepiece librosa flask flask-cors sounddevice
   ```

### Required Libraries
- `torch`, `transformers`, `datasets`, `torchaudio`, `sentencepiece`, `librosa` (Core ML & Audio)
- `flask`, `flask-cors` (API deployment)
- `sounddevice` (Optional, required for microphone input via CLI)

## 4. Project Structure

Here is a breakdown of the key files in the repository:
- **`main.py`**: The main point of entry for the CLI. Handles parsing arguments, loading audio, and triggering transcription.
- **`audio_handler.py`**: Handles loading audio files from disk or capturing live audio from the microphone. Ensures audio is formatted correctly.
- **`feature_extractor.py`**: Extracts essential features (spectrograms) from raw audio waveforms to feed into the transformer model.
- **`model_inference.py`**: Loads the pre-trained Hugging Face model and runs the inference logic to generate token IDs.
- **`output_processing.py`**: Decodes the output token IDs back into human-readable text.
- **`app.py`**: The Flask backend application that initializes the REST API, handles file uploads, and returns JSON responses.
- **`templates/index.html`** (optional): A frontend UI layout to interact with the API via the browser.

## 5. How the Project Works (Flow Explanation)

### General Pipeline
**Audio Input → Feature Extraction → Model Inference → Output Processing → Text Output**

### API Flow
**User → Frontend UI** → Uploads Audio → **Flask API** → Passes audio to **Model** → Returns **Response (JSON)**

## 6. Working Steps

You can run the project in two primary modes:
**a) CLI mode (`main.py`)**: Best for local testing and debugging. Evaluate a file directly or record from your microphone.
**b) Flask API mode (`app.py`)**: Runs a web server locally. Use the browser UI or an HTTP client (e.g., Postman) to upload files and receive transcriptions.

## 7. Commands to Run

### a) Activate virtual environment:
- **Windows**: `venv\Scripts\activate`
- **Mac/Linux**: `source venv/bin/activate`

### b) Run CLI version:
```bash
# Using an audio file
python main.py --mode file --file "sample.wav"

# Using your microphone
python main.py --mode mic --duration 5
```

### c) Run Flask server:
```bash
python app.py
```

### d) Access API:
Once the server is running, the endpoint will be available at:
`http://127.0.0.1:5000/predict`

## 8. API Usage

You can interact with the API by sending basic HTTP requests:
- **Method**: `POST`
- **Endpoint**: `/predict`
- **Input**: Audio file using `multipart/form-data`.
- **Output**: JSON response containing the transcription.

**Example Response**:
```json
{
  "success": true,
  "transcription": "Hello world"
}
```

## 9. Sample Input
- **Supported Format**: `.wav`
- **Recommended**: **16kHz mono audio**. (Note: the backend automatically attempts to resample if the audio diverges from 16kHz).

## 10. Output
- **CLI Mode**: Prints transcription in the terminal.
- **API Mode**: Returns a JSON response.

## 11. Important Notes
- **Sampling Rate**: Always use `sampling_rate=16000`, as strictly required by the underlying model parameters.
- **Model Loading**: The model is instantiated once at the startup of `app.py` or internally cached to avoid reloading on every request, vastly improving speeds.
- **Error Types**: Missing file uploads or unreadable formats are handled carefully before hitting the inference pipeline.

## 12. Common Errors and Fixes
- **File not found**: Confirm the path to your `--file` is precise when running locally.
- **CORS issues**: If accessing the API from a different domain (like a React frontend), make sure `flask-cors` is managing the origins.
- **Slow response**: Model inference initialization can take a moment during the very first run. Consequent requests run significantly faster.
- **Wrong audio format**: Be sure your file isn't corrupted or in an incompatible format structure trying to mimic `.wav`.

## 13. Future Improvements
- **React frontend integration**: Swap the basic HTML view for a dynamic React component layout.
- **Real-time streaming transcription**: Utilize WebSockets for streaming live audio chunks for immediate processing.
- **Multi-language support**: Switch to alternative Hugging Face models tailored for other languages.
- **Deployment on cloud platforms**: Export as a Docker container to host on cloud platforms like Render, AWS, or GCP.

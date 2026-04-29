# Module 6: Deployment and UI Integration

## 1. Objective

Congratulations on successfully building the core Speech-to-Text inference pipeline! The final step in making your model truly useful is **Deployment**—making it accessible to end users who don't know how to run Python scripts or use the command line. 

In this module, we will learn how to:
- Convert the existing offline script (audio → features → model → text) into a live, usable web application.
- Understand the role of a Backend API (which hosts our model) and a Frontend UI (where users upload audio).

## 2. Architecture: How the Application Works

To serve our AI model over the internet, we transition from a simple script to a **Client-Server Architecture**.

**Flow of Data:**
1. **User (Frontend):** The user opens a web page and uploads an audio file (or captures microphone input).
2. **Frontend (Client):** The interface packages the audio file and sends an HTTP POST request to our backend server.
3. **Backend API (Server):** The server receives the file via an endpoint (e.g., `/predict`).
4. **Model Interface:** The server loads the audio (ensuring `sampling_rate=16000`), runs the Hugging Face pipeline (Processor → Model → Decoder), and generates the text.
5. **Response:** The server sends the transcribed text back as a structured JSON response to the Frontend.
6. **Result:** The Frontend displays the text to the User.

## 3. Backend Implementation (Flask API)

We will use **Flask**, a lightweight Python web framework, to create our backend API.

**Key concepts:**
- **Load Model Once:** We must load the Hugging Face model and processor when the server *starts*, not during every request. Loading it per request will cause massive latency and memory issues.
- **Endpoint (`/predict`):** We create a specific URL path that securely listens for `POST` requests.
- **Audio Processing:** We use `flask.request.files` to get the uploaded file, read it into memory, and use `librosa.load(..., sr=16000)` to ingest it seamlessly for our pipeline.

## 4. Code Example: Flask Application (`app.py`)

Here is a complete, beginner-friendly backend script for our Speech-to-Text model.

```python
import io
import torch
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration

app = Flask(__name__)
# Enable CORS so our frontend can communicate with this API from a browser
CORS(app) 

# 1. LOAD MODEL ONCE AT STARTUP (Not per request)
print("Loading Model and Processor...")
MODEL_ID = "facebook/s2t-small-librispeech-asr"
processor = Speech2TextProcessor.from_pretrained(MODEL_ID)
model = Speech2TextForConditionalGeneration.from_pretrained(MODEL_ID)
model.eval()  # Set model to evaluation mode for faster inference
print("Model loaded successfully!")

@app.route('/predict', methods=['POST'])
def predict():
    # 2. HANDLE FILE UPLOAD
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file part in the request"}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # 3. LOAD AUDIO INTO MEMORY
        audio_bytes = file.read()
        # librosa expects a file-like object or path; we use io.BytesIO for memory mapping
        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        
        # 4. RUN PIPELINE
        # Extract features
        inputs = processor(
            audio_data, 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        
        # Generate token IDs (no_grad reduces memory consumption)
        with torch.no_grad():
            generated_ids = model.generate(
                inputs["input_features"], 
                attention_mask=inputs.get("attention_mask")
            )
            
        # Decode to text
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 5. RETURN JSON RESPONSE
        return jsonify({
            "success": True,
            "transcription": transcription
        }), 200

    except Exception as e:
        # Catch faulty files or runtime errors
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask development server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
```

## 5. Frontend Example (Simple HTML UI)

To test our API, we can create a simple `index.html` file. The user selects a file, clicks a button, and JavaScript sends it to our Flask API.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Speech-to-Text</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, sans-serif; padding: 50px; background-color: #f4f4f9; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); max-width: 500px; margin: auto; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        #result { margin-top: 20px; padding: 15px; background: #e9ecef; border-radius: 5px; min-height: 50px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Upload Audio for Transcription</h2>
        <input type="file" id="audioFile" accept="audio/*">
        <br><br>
        <button onclick="uploadAudio()">Transcribe</button>
        
        <h3>Result:</h3>
        <div id="result">Transcription will appear here...</div>
    </div>

    <script>
        async function uploadAudio() {
            const fileInput = document.getElementById('audioFile');
            const resultDiv = document.getElementById('result');
            
            if (fileInput.files.length === 0) {
                alert("Please select a file first.");
                return;
            }

            resultDiv.innerText = "Processing... Please wait.";

            const formData = new FormData();
            formData.append("audio", fileInput.files[0]);

            try {
                // Send the file to our Flask backend
                const response = await fetch("http://127.0.0.1:5000/predict", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();
                
                if (data.success) {
                    resultDiv.innerText = data.transcription;
                } else {
                    resultDiv.innerText = "Error: " + data.error;
                }
            } catch (error) {
                resultDiv.innerText = "Error connecting to the server.";
                console.error(error);
            }
        }
    </script>
</body>
</html>
```

## 6. Deployment Options

Once your app works locally, here is how you can host it on the internet:

1. **Hugging Face Spaces (Recommended for ML):**
   - The absolute easiest way to deploy ML web apps.
   - You can use libraries like Streamlit or Gradio which eliminates the need to write separate front/backend code.
   - Provides free CPU or paid GPU instances specifically tailored for AI.
2. **Render or Railway (Backend APIs):**
   - Great for deploying the Flask API (`app.py`).
   - You configure a `requirements.txt` and they build it natively.
   - *Note:* Free tiers have limited RAM (usually 512MB), which might crash when loading large Transformer models.
3. **Vercel or Netlify (Frontend UI):**
   - Best for hosting your HTML/JS frontend (or React applications).
   - *Note:* They run "Serverless Functions" which have strict time limits and no active filesystem, so they **cannot** run persistent Python ML models directly. They must talk to your Backend API hosted elsewhere.

## 7. Important Notes

- **Always use `sampling_rate=16000`:** Most HF Speech models (like `facebook/s2t` and `whisper`) are trained strictly on 16kHz audio. Feeding the model 44.1kHz or 48kHz audio without resampling will result in random, confused text tracking.
- **Handle File Upload Errors Robustly:** Users might upload unsupported files (like PDFs). Wrapping the `librosa.load` block in a `try/except` ensures your server informs the user instead of outright crashing.
- **Global Model Loading:** The model parameters are large (100MB+). Loading it inside the route will stall the web server and destroy RAM. Initialization should always happen at the top of your script.

## 8. Common Errors and Fixes

| Symptom / Error | Cause | Fix |
| :--- | :--- | :--- |
| **"No audio file part" in response** | Frontend sending wrong payload key. | Ensure frontend `FormData.append("audio", file)` matches backend `request.files['audio']`. |
| **CORS Blocked in Browser Console** | Frontend and Backend are on different ports/domains. | Install `flask-cors` and use `CORS(app)` in your Flask script to bypass security blocking. |
| **API takes 15+ seconds per request** | Loading model dynamically per request. | Move `from_pretrained()` to global level, outside of the route handler. |
| **Memory Leak / Server Crash** | Accumulating gradients during text generation. | Ensure you are wrapping your generation loop in `with torch.no_grad():`. |

## 9. Output of Module

By configuring `app.py` and writing a simple user interface, the complex Speech-to-Text inference logic is now cleanly abstracted behind a REST API. Users can trivially upload audio files through their web browser and immediately receive transcribed text generated by your AI system! You now have the blueprint for a production-ready application.

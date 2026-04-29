import os
import tempfile
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from audio_handler import AudioInputHandler
from feature_extractor import FeatureExtractor
from model_inference import ModelInference
from output_processing import OutputProcessor

app = Flask(__name__)
CORS(app)

print("Loading Speech-to-Text Pipeline...")
inference_engine = ModelInference()
output_engine = OutputProcessor(processor=inference_engine.processor)
audio_handler = AudioInputHandler()
extractor = FeatureExtractor()
print("Pipeline loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file part in the request"}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_path = None
    try:
        # Create a temp file to store uploaded audio since our audio_handler works with paths
        # This allows us to use the robust FFMPEG fallback built in the previous module!
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        file.save(temp_path)
        
        # 1. Load Audio
        audio, sr = audio_handler.load_from_file(temp_path)
        if audio is None:
             return jsonify({"error": "Failed to load or process audio file. Check format."}), 500

        # 2. Extract Features
        features = extractor.extract(audio, sampling_rate=sr)
        if features is None:
             return jsonify({"error": "Feature extraction failed."}), 500

        # 3. Predict Tokens
        generated_ids = inference_engine.generate_tokens(features)
        if generated_ids is None:
             return jsonify({"error": "Model inference failed."}), 500

        # 4. Process Output
        transcription = output_engine.decode_and_clean(generated_ids, save_to_file=False)

        return jsonify({"success": True, "transcription": transcription}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

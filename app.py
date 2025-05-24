import os
import tempfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from tensorflow import keras
import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf
import io
import tensorflow as tf

# Configurations
MODEL_PATH = os.getenv('MODEL_PATH', 'models/heart_sound_model1_v5_0001.h5')  # Use .keras or .h5
NUM_SECONDS = 10
SAMPLE_RATE = 22050
NUM_SAMPLES = SAMPLE_RATE * NUM_SECONDS
N_MFCC = 64
N_FFT = 2048
HOP_LENGTH = 512
MAPPINGS = ['abnormal', 'artifact', 'normal']
INPUT_SHAPE = (128, 128, 1)

# Flask App
app = Flask(__name__)
CORS(app)

# Load Model
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}, expecting input shape {model.input_shape}")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None


# --- Audio Preprocessing ---
def preprocess_audio(file_path, sr=SAMPLE_RATE, duration=NUM_SECONDS):
    signal, sr = librosa.load(file_path, sr=sr, duration=duration)
    if len(signal) < NUM_SAMPLES:
        signal = np.pad(signal, (0, NUM_SAMPLES - len(signal)), mode='constant')

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=2048,
        hop_length=512
    )

    # Normalize MFCCs
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)

    # Resize to match INPUT_SHAPE
    mfccs_resized = tf.image.resize(mfccs[..., np.newaxis], INPUT_SHAPE[:2])

    # mfccs_resized is a tf.Tensor
    arr = mfccs_resized.numpy()  # now a NumPy array
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if 'audio_file' not in request.files:
        return jsonify({"error": "Missing 'audio_file' in request"}), 400

    audio_file = request.files['audio_file']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        audio_file.save(tmp.name)
        file_path = tmp.name

    try:
        input_tensor = preprocess_audio(file_path)
        preds = model.predict(input_tensor)
        idx = int(np.argmax(preds, axis=1)[0])
        label = MAPPINGS[idx]
        confidence = float(preds[0][idx])

        return jsonify({
            "label": label,
            "confidence": round(confidence, 4),
            "raw_scores": preds[0].tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(file_path)


# --- Noise Reduction Endpoint ---
@app.route('/noise-reduction', methods=['POST'])
def noise_reduction():
    if 'noise_only' not in request.files or 'heart_noisy' not in request.files:
        return jsonify({"error": "Both 'noise_only' and 'heart_noisy' are required"}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_noise:
            noise_path = tmp_noise.name
            request.files['noise_only'].save(noise_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_heart:
            heart_path = tmp_heart.name
            request.files['heart_noisy'].save(heart_path)

        noisy, sr = librosa.load(heart_path, sr=None)
        noise, _ = librosa.load(noise_path, sr=sr)
        cleaned = nr.reduce_noise(y=noisy, y_noise=noise, sr=sr)

        buffer = io.BytesIO()
        sf.write(buffer, cleaned, sr, format='WAV')
        buffer.seek(0)

    except Exception as e:
        return jsonify({"error": f"Noise reduction failed: {e}"}), 500
    finally:
        for path in [locals().get('noise_path'), locals().get('heart_path')]:
            if path and os.path.exists(path):
                os.remove(path)

    return send_file(buffer, as_attachment=True, download_name="cleaned.wav", mimetype="audio/wav")


@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "Healthy"}), 200


# --- Run Server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))

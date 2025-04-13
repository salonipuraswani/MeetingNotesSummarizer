from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from faster_whisper import WhisperModel
from transformers import pipeline

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model (use 'tiny', 'base', 'small', etc.)
model = WhisperModel("small", compute_type="int8", device="cpu")  # good balance for CPU

# Load the Hugging Face Summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio = request.files['audio']
    file_path = os.path.join(UPLOAD_FOLDER, audio.filename)
    audio.save(file_path)

    try:
        print("🟢 Starting transcription using faster-whisper...")
        segments, info = model.transcribe(file_path, beam_size=5)
        transcription = " ".join(segment.text for segment in segments)

        print("✅ Transcription complete. Summarizing...")

        # Summarize the transcription
        summary = summarizer(transcription, max_length=1000, min_length=20, do_sample=False)

        print("✅ Summary complete")
        return jsonify({
            'transcription': transcription,
            'summary': summary[0]['summary_text']
        })

    except Exception as e:
        print("🔥 Exception occurred during transcription or summarization:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

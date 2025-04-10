from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from flask import render_template


app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

    # Simulate processing (Replace this with actual ML/audio summarization logic)
    summary = f"This is a generated summary for: {audio.filename}. You can replace this with real logic."

    return jsonify({'summary': summary})


if __name__ == '__main__':
    app.run(debug=True, port=5000)

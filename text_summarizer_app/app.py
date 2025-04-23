from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import re
from faster_whisper import WhisperModel
from transformers import pipeline
import spacy
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name__)
CORS(app)

# Folder to temporarily store uploaded audio files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load transcription and NLP models
model = WhisperModel("small", compute_type="int8", device="cpu")  # Whisper for transcription
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # BART for summarization
nlp = spacy.load("en_core_web_sm")  # spaCy for sentence and entity parsing

# Keywords to detect important sentences in the transcript
KEY_TERMS = {
    "agenda", "decision", "deadline", "issue", "task",
    "assign", "responsible", "timeline", "progress", "meeting"
}

# Regular expressions to identify various date formats
DATE_PATTERNS = [
   r'\b\d{1,2}/\d{1,2}/\d{4}\b',                                 # 12/04/2025
   r'\b\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}\b',                 # 21st March 2025
   r'\b\w+\s+\d{1,2},\s+\d{4}\b',                                # March 21, 2025
   r'\b\d{1,2}(?:st|nd|rd|th)?\s+of\s+\w+\b',                    # 5th of April
   r'\b\d{1,2}(?:st|nd|rd|th)?\s+of\s+this\s+month\b',           # 15th of this month
   r'\b\d{1,2}(?:st|nd|rd|th)?\s+of\s+next\s+month\b',           # 3rd of next month
   r'\b\d{1,2}(?:st|nd|rd|th)?\s+\w+\b',                         # 7th March
   r'\b\w+\s+\d{1,2}(?:st|nd|rd|th)?\b',                         # March 7th
   r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',                               # 5/9/23
   r'\b\d{1,2}(?:st|nd|rd|th)?\s+of\s+(?:this|next)?\s*\w+\b',   # 2nd of this May/ 5th of next June
   r'\b(?:this\s+)?(?:month|year)\b',                            # this year
  # r'\b\d{1,2}(?:st|nd|rd|th)\b',                               # 21st
]

# Function to extract specific dates using regex patterns
def extract_specific_dates(text):
    matched_dates = set()
    date_dict = {}

    for pattern in DATE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            date = match.group(0)
            if re.match(r'\b\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}\b', date):
                day = date.split()[0]
                if day not in date_dict:
                    date_dict[day] = date
                    matched_dates.add(date)
            else:
                matched_dates.add(date)
    return matched_dates

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')  # Optional, used if front-end is connected

@app.route('/summarize', methods=['POST'])
def summarize_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    # Save uploaded audio file
    audio = request.files['audio']
    file_path = os.path.join(UPLOAD_FOLDER, audio.filename)
    audio.save(file_path)

    try:
        # Transcribe audio using Whisper
        segments, _ = model.transcribe(file_path, beam_size=5)
        transcription = " ".join(segment.text for segment in segments)

        # Use spaCy to parse sentences and entities
        doc = nlp(transcription)
        informative_sentences = []
        speakers = set()
        specific_dates = extract_specific_dates(transcription)

        # Extract informative sentences and named speakers
        for sent in doc.sents:
            pos_tags = [token.pos_ for token in sent]
            has_keywords = any(word.lower() in KEY_TERMS for word in sent.text.split())

            for ent in sent.ents:
                if ent.label_ == "PERSON":
                    speakers.add(ent.text)

            pos_score = sum(pos in ["NOUN", "PROPN", "VERB"] for pos in pos_tags)
            if pos_score >= 3 or has_keywords or len(sent.text.split()) >= 10:
                informative_sentences.append(sent.text)

        # Reduce transcription content to half to feed into summarizer
        total_words = len(transcription.split())
        max_words = total_words // 2
        reduced_text = ""
        word_count = 0
        for sent in informative_sentences:
            words = sent.split()
            if word_count + len(words) > max_words:
                break
            reduced_text += sent + " "
            word_count += len(words)

        # Calculate length for summarizer input
        max_len = max(50, int(total_words * 1.3) // 4)
        min_len = max(30, max_len // 2)

        # Generate abstractive summary using Hugging Face model
        summary = summarizer(
            reduced_text.strip(),
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )[0]['summary_text']

        return jsonify({
            'transcription': transcription,
            'summary': summary,
            'dates': list(specific_dates) if specific_dates else ["Not mentioned"],
            'speakers': list(speakers) if speakers else ["Not mentioned"]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    transcription = data.get('transcription', 'Not available')
    summary = data.get('summary', 'Not available')
    dates = data.get('dates', [])
    speakers = data.get('speakers', [])

    # Create a PDF in memory using ReportLab
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50

    # Add PDF content: title
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(30, y, "Meeting Summary Report")
    y -= 30

    # Summary Section
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(30, y, "Summary:")
    y -= 20
    pdf.setFont("Helvetica", 11)
    for line in summary.split('. '):
        if y < 60:
            pdf.showPage()
            y = height - 50
            pdf.setFont("Helvetica", 11)
        pdf.drawString(40, y, line.strip() + ".")
        y -= 15

    # Speakers Section
    y -= 20
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(30, y, "Speakers Mentioned:")
    y -= 20
    pdf.setFont("Helvetica", 11)
    for speaker in speakers:
        if y < 60:
            pdf.showPage()
            y = height - 50
        pdf.drawString(40, y, f"- {speaker}")
        y -= 15

    # Dates Section
    y -= 20
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(30, y, "Dates Mentioned:")
    y -= 20
    pdf.setFont("Helvetica", 11)
    for date in dates:
        if y < 60:
            pdf.showPage()
            y = height - 50
        pdf.drawString(40, y, f"- {date}")
        y -= 15

    # Full Transcription Section
    y -= 20
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(30, y, "Full Transcription:")
    y -= 20
    pdf.setFont("Helvetica", 10)
    for para in transcription.split('. '):
        if y < 60:
            pdf.showPage()
            y = height - 50
            pdf.setFont("Helvetica", 10)
        pdf.drawString(40, y, para.strip() + ".")
        y -= 12

    # Finalize and send the PDF
    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="meeting_summary.pdf",
        mimetype='application/pdf'
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)
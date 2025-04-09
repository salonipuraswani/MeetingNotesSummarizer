from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)
summarizer = pipeline("summarization", device=-1)

@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    if request.method == "POST":
        text = request.form["text"]
        result = summarizer(text, max_length=60, min_length=20, do_sample=False)
        summary = result[0]['summary_text']
    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)

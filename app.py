from flask import Flask, request, jsonify, render_template
from AI_Resume_Rank import evaluate_resumes
import tempfile
import os

app = Flask(__name__)
UPLOAD_FOLDER = './resumes'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    """
    Home page to display the form for uploading resumes and job description.
    """
    return render_template('AI_Resume_Rank.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """
    Flask route to handle resume file upload and ranking.
    """
    try:
        # Get job description and keywords from form
        job_description = request.form['job_description']
        job_keywords = set(request.form.getlist('keywords'))  # Get job-related keywords
        files = request.files.getlist('resumes')

        if not files or not job_description:
            return jsonify({"error": "Missing files or job description"}), 400

        # Save uploaded resumes temporarily
        resume_paths = []
        for file in files:
            # Using a context manager to safely save and close the file
            with tempfile.NamedTemporaryFile(delete=False, dir=UPLOAD_FOLDER) as temp_file:
                file.save(temp_file.name)  # Save the file to the temp file path
                resume_paths.append(temp_file.name)  # Add to list for processing later

        # Process resumes and get scores
        scores = evaluate_resumes(resume_paths, job_description, job_keywords)

        # Clean up temporary files
        for resume_path in resume_paths:
            os.remove(resume_path)

        # Format the results into a readable format
        result = [{"resume_index": score[0], "score": round(score[1], 2)} for score in scores]
        
        return render_template('results.html', scores=result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

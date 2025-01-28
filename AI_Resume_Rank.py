import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import tempfile

UPLOAD_FOLDER = './resumes'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def parse_resume(file_path):
    """
    Extracts text content from a resume file (PDF).
    """
    try:
        text = ''
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def preprocess_and_vectorize(resume_texts, job_description):
    """
    Converts the text of resumes and job descriptions into TF-IDF vectors.
    """
    if not job_description.strip():
        print("Job description is empty. Please provide valid input.")
        return None, None

    if not any(resume_texts):
        print("No valid resume texts to process. Ensure the resumes are properly formatted.")
        return None, None

    documents = resume_texts + [job_description]  # Combine resumes and JD
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
        return tfidf_matrix, vectorizer
    except Exception as e:
        print(f"Error during TF-IDF vectorization: {e}")
        return None, None

def keyword_count_normalized(resume_text, job_keywords):
    """
    Counts the occurrences of job-related keywords in the resume text and normalizes by the total word count.
    """
    resume_words = resume_text.lower().split()
    keyword_counts = Counter([word for word in resume_words if word in job_keywords])
    total_keywords = sum(keyword_counts.values())
    total_words = len(resume_words)
    return total_keywords / total_words if total_words > 0 else 0

def calculate_scores(tfidf_matrix, num_resumes, job_keywords, resume_texts):
    """
    Calculates cosine similarity and normalized keyword frequency for resumes.
    """
    if tfidf_matrix is None:
        print("TF-IDF matrix is None. Skipping scoring.")
        return []

    jd_vector = tfidf_matrix[-1]  # Last vector corresponds to the job description
    resume_vectors = tfidf_matrix[:-1]  # All other vectors are resumes
    try:
        similarity_scores = cosine_similarity(resume_vectors, jd_vector).flatten()
        scores = []
        for i, resume_text in enumerate(resume_texts):
            keyword_score = keyword_count_normalized(resume_text, job_keywords)
            total_score = (similarity_scores[i] * 0.7) + (keyword_score * 0.3)  # Weighted combination
            scores.append((i, total_score * 100))  # Scale to percentage
        return sorted(scores, key=lambda x: x[1], reverse=True)
    except Exception as e:
        print(f"Error during score calculation: {e}")
        return []

def evaluate_resumes(resume_files, job_description, job_keywords):
    """
    Main function to evaluate resumes and return scores.
    """
    resume_texts = [parse_resume(file_path) for file_path in resume_files]
    tfidf_matrix, vectorizer = preprocess_and_vectorize(resume_texts, job_description)
    scores = calculate_scores(tfidf_matrix, len(resume_texts), job_keywords, resume_texts)

    return scores

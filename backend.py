from flask import Flask, render_template, request, jsonify
import json

app = Flask(__name__)

# Example Python functions that return JSON
def process_cv(cv_file):
    # Process the CV file
    get_cv_data()
    return {"experiences": ["Experience 1", "Experience 2"]}

def generate_full_report(cv_data, job_offer_data):
    # Generate the full report
    return {"text": "Full report text", "figure": "path/to/figure.png"}

def process_job_offer(job_offer_file):
    # Process the job offer file
    return {"summary": "Job offer summary"}

@app.route('/')
def index():
    return render_template('index.html')  # Your HTML file

@app.route('/upload-cv', methods=['POST'])
def upload_cv():
    cv_file = request.files['cv-upload']
    cv_data = process_cv(cv_file)
    # Save CV data in session or database for later use
    return jsonify(cv_data)

@app.route('/generate-report', methods=['POST'])
def report():
    cv_data = request.json['cv_data']
    job_offer_data = request.json['job_offer_data']
    report_data = generate_full_report(cv_data, job_offer_data)
    return jsonify(report_data)

@app.route('/upload-job-offer', methods=['POST'])
def upload_job_offer():
    job_offer_file = request.files['job-offer-upload']
    job_offer_data = process_job_offer(job_offer_file)
    # Save job offer data in session or database for later use
    return jsonify(job_offer_data)

# This is your sample JSON data that the PDF detection library might return
sample_cv_data = {
    "personal_info": {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "123-456-7890"
    },
    "experiences": [
        {
            "title": "Software Engineer",
            "company": "Tech Corp",
            "duration": "June 2018 - Present",
            "details": "Developed and maintained several high-traffic web applications."
        },
        {
            "title": "Junior Developer",
            "company": "Dev Startup",
            "duration": "July 2016 - May 2018",
            "details": "Assisted in the development of web applications and provided technical support."
        }
    ],
    "education": [
        {
            "degree": "B.Sc. in Computer Science",
            "institution": "State University",
            "year": "2012 - 2016"
        }
    ],
    "skills": ["Python", "JavaScript", "SQL", "HTML/CSS"]
}

@app.route('/get-cv-data', methods=['GET'])
def get_cv_data():
    """
    Route to get the CV data. This simulates retrieving data processed by a PDF detection library.
    """
    return jsonify(sample_cv_data)


if __name__ == '__main__':
    app.run(debug=True)
    
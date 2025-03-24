import os
import email
from email import policy
import re
import json
import hashlib
import logging
from flask import Flask, request, render_template, redirect, url_for, flash
import jaydebeapi
from transformers import pipeline
from multiprocessing import Pool
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("email_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__, template_folder='templates')
app.secret_key = "your-secret-key"  # Required for flash messages
UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Add error handlers
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return "Internal Server Error", 500


@app.errorhandler(405)
def method_not_allowed(e):
    logger.warning(f"Method not allowed: {str(e)}")
    flash("Please use the upload form to submit files.", "error")
    return redirect(url_for('upload_page'))


# Define request types, sub-request types, and routing rules
REQUEST_TYPES = ["Payment Inquiry", "Loan Request", "Account Update"]
SUB_REQUEST_TYPES = {
    "Payment Inquiry": ["Payment Confirmation", "Payment Delay"],
    "Loan Request": ["Loan Status", "Loan Approval"],
    "Account Update": ["Account Details", "Account Closure"]
}
ROUTING_RULES = {
    "Payment Inquiry": "Finance Team",
    "Loan Request": "Loan Processing Team",
    "Account Update": "Customer Support Team"
}
PRIORITY_RULES = {
    "Payment Inquiry": 3,
    "Loan Request": 2,
    "Account Update": 1
}

# Load Hugging Face models in the main process
logger.info("Loading Hugging Face models...")
CLASSIFIER = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
NER = pipeline("ner", model="dslim/bert-base-NER")
logger.info("Models loaded successfully.")


# Step 1: Extract email components
def extract_email_components(filepath):
    logger.info(f"Extracting components from email: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            msg = email.message_from_file(f, policy=policy.default)
            subject = msg['subject'] or ""
            body = ""
            attachments = []

            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    body += part.get_payload(decode=True).decode()
                elif not part.get_content_type().startswith('multipart'):
                    attachment_name = part.get_filename()
                    if attachment_name:
                        attachments.append((attachment_name, part.get_payload(decode=True)))
                        logger.info(f"Attachment found: {attachment_name}")

            email_data = {
                'filename': os.path.basename(filepath),
                'subject': subject,
                'body': body,
                'attachments': attachments
            }
            logger.info(f"Email components extracted: {email_data['filename']}")
            return email_data
    except Exception as e:
        logger.error(f"Error extracting email components from {filepath}: {str(e)}")
        return None


# Step 2: Classify email intent using Hugging Face (in main process)
def classify_email_intent(email_data):
    logger.info(f"Classifying intent for email: {email_data['filename']}")
    try:
        text = email_data['subject'] + " " + email_data['body']
        result = CLASSIFIER(text, candidate_labels=REQUEST_TYPES)
        request_type = result['labels'][0]
        confidence = result['scores'][0]

        sub_types = SUB_REQUEST_TYPES.get(request_type, [])
        sub_request_type, sub_confidence = None, 0.0
        if sub_types:
            sub_result = CLASSIFIER(text, candidate_labels=sub_types)
            sub_request_type = sub_result['labels'][0]
            sub_confidence = sub_result['scores'][0]

        intent = {
            'request_type': request_type,
            'request_confidence': confidence,
            'sub_request_type': sub_request_type,
            'sub_request_confidence': sub_confidence
        }
        logger.info(f"Intent classified: {intent}")
        return intent
    except Exception as e:
        logger.error(f"Error classifying intent for {email_data['filename']}: {str(e)}")
        return None


# Step 3: Extract context of email using Hugging Face (in main process)
def extract_context(email_data):
    logger.info(f"Extracting context for email: {email_data['filename']}")
    try:
        text = email_data['subject'] + " " + email_data['body']
        entities = NER(text)
        context = {'entities': []}
        for entity in entities:
            context['entities'].append({
                'text': entity['word'],
                'label': entity['entity']
            })

        amount_pattern = r"USD\s*[\d,]+\.\d{2}"
        date_pattern = r"\d{1,2}-[A-Z]{3}-\d{4}"
        amounts = re.findall(amount_pattern, text)
        dates = re.findall(date_pattern, text)

        context['amounts'] = amounts
        context['dates'] = dates
        logger.info(f"Context extracted: {context}")
        return context
    except Exception as e:
        logger.error(f"Error extracting context for {email_data['filename']}: {str(e)}")
        return None


# Step 4: Handle multi-request emails (in main process)
def handle_multi_request(email_data):
    logger.info(f"Handling multi-request for email: {email_data['filename']}")
    try:
        text = email_data['body']
        segments = text.split("\n\n")
        intents = []
        for segment in segments:
            if segment.strip():
                result = CLASSIFIER(segment, candidate_labels=REQUEST_TYPES)
                intents.append({
                    'segment': segment,
                    'request_type': result['labels'][0],
                    'confidence': result['scores'][0]
                })

        primary_intent = max(intents, key=lambda x: x['confidence']) if intents else None
        logger.info(f"Multi-request handled: Primary intent - {primary_intent}")
        return primary_intent, intents
    except Exception as e:
        logger.error(f"Error handling multi-request for {email_data['filename']}: {str(e)}")
        return None, []


# Step 5: Assign priority and confidence
def assign_priority_and_confidence(intent, email_data):
    logger.info(f"Assigning priority for email: {email_data['filename']}")
    try:
        request_type = intent['request_type']
        confidence = intent['request_confidence']
        priority = PRIORITY_RULES.get(request_type, 1)

        text = email_data['body'].lower()
        if "urgent" in text or "immediate" in text:
            priority += 1

        scored_intent = {
            'request_type': request_type,
            'confidence': confidence,
            'priority': priority
        }
        logger.info(f"Priority assigned: {scored_intent}")
        return scored_intent
    except Exception as e:
        logger.error(f"Error assigning priority for {email_data['filename']}: {str(e)}")
        return None


# Step 6: Compute email hash for duplicate detection
def compute_email_hash(email_data):
    logger.info(f"Computing hash for email: {email_data['filename']}")
    try:
        text = email_data['subject'] + email_data['body']
        email_hash = hashlib.md5(text.encode()).hexdigest()
        logger.info(f"Hash computed: {email_hash}")
        return email_hash
    except Exception as e:
        logger.error(f"Error computing hash for {email_data['filename']}: {str(e)}")
        return None


# Step 7: Set up H2 database
def setup_h2_database():
    logger.info("Setting up H2 database...")
    try:
        conn = jaydebeapi.connect(
            "org.h2.Driver",
            "jdbc:h2:./email_db",
            ["sa", ""],
            "h2-latest.jar"
        )
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emails (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255),
                request_type VARCHAR(255),
                sub_request_type VARCHAR(255),
                confidence FLOAT,
                context TEXT,
                email_hash VARCHAR(255)
            )
        """)
        logger.info("H2 database setup complete.")
        return conn, cursor
    except Exception as e:
        logger.error(f"Error setting up H2 database: {str(e)}")
        return None, None


# Step 8: Detect duplicates
def detect_duplicates(cursor, email_data, email_hash):
    logger.info(f"Detecting duplicates for email: {email_data['filename']}")
    try:
        cursor.execute("SELECT COUNT(*) FROM emails WHERE email_hash = ?", (email_hash,))
        count = cursor.fetchone()[0]
        is_duplicate = count > 0
        logger.info(f"Duplicate check: {'Duplicate' if is_duplicate else 'Not a duplicate'}")
        return is_duplicate
    except Exception as e:
        logger.error(f"Error detecting duplicates for {email_data['filename']}: {str(e)}")
        return False


# Step 9: Store email data in h2 database
def store_email_data(cursor, email_data, intent, context, email_hash):
    logger.info(f"Storing email data for: {email_data['filename']}")
    try:
        cursor.execute("""
            INSERT INTO emails (filename, request_type, sub_request_type, confidence, context, email_hash)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            email_data['filename'],
            intent['request_type'],
            intent['sub_request_type'],
            intent['request_confidence'],
            str(context),
            email_hash
        ))
        logger.info("Email data stored successfully.")
    except Exception as e:
        logger.error(f"Error storing email data for {email_data['filename']}: {str(e)}")


# Step 10: Route request
def route_request(result):
    logger.info(f"Routing request for email: {result['filename']}")
    try:
        request_type = result['intent']['request_type']
        team = ROUTING_RULES.get(request_type, "Default Team")
        result['routing'] = {'team': team}
        logger.info(f"Request routed to: {team}")
        return result
    except Exception as e:
        logger.error(f"Error routing request for {result['filename']}: {str(e)}")
        return result


# Step 11: Log classification decision for bias mitigation
def log_classification_decision(result):
    logger.info(f"Logging classification decision for: {result['filename']}")
    try:
        with open("classification_log.txt", "a") as f:
            f.write(
                f"{datetime.now()} - Email: {result['filename']}, Intent: {result['intent']['request_type']}, Confidence: {result['intent']['confidence']}\n")
        logger.info("Classification decision logged.")
    except Exception as e:
        logger.error(f"Error logging classification decision for {result['filename']}: {str(e)}")


# Step 12: Process a single email (without model loading)
def process_single_email(args):
    filepath, email_data, intent, context, primary_intent, all_intents = args
    logger.info(f"Processing email: {filepath}")
    try:
        # Assign priority and confidence
        scored_intent = assign_priority_and_confidence(intent, email_data)
        if not scored_intent:
            return None

        # Compute email hash
        email_hash = compute_email_hash(email_data)
        if not email_hash:
            return None

        # Set up database
        conn, cursor = setup_h2_database()
        if not conn or not cursor:
            return None

        # Detect duplicates
        is_duplicate = detect_duplicates(cursor, email_data, email_hash)

        # Store in database
        store_email_data(cursor, email_data, intent, context, email_hash)

        # Prepare result
        result = {
            'filename': email_data['filename'],
            'intent': scored_intent,
            'context': context,
            'is_duplicate': is_duplicate,
            'all_intents': all_intents
        }

        # Route request
        result = route_request(result)

        # Log classification decision
        log_classification_decision(result)

        conn.commit()
        conn.close()
        logger.info(f"Email processed successfully: {filepath}")
        return result
    except Exception as e:
        logger.error(f"Error processing email {filepath}: {str(e)}")
        return None


# Step 13: Process email pipeline with parallel processing
def process_email_pipeline(directory):
    logger.info(f"Starting email processing pipeline for directory: {directory}")
    try:
        filepaths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".eml")]
        logger.info(f"Found {len(filepaths)} EML files to process.")

        # Preprocess emails in the main process (model inference)
        preprocessed_data = []
        for filepath in filepaths:
            # Extract email components
            email_data = extract_email_components(filepath)
            if not email_data:
                continue

            # Classify intent
            intent = classify_email_intent(email_data)
            if not intent:
                continue

            # Extract context
            context = extract_context(email_data)
            if not context:
                continue

            # Handle multi-request
            primary_intent, all_intents = handle_multi_request(email_data)

            preprocessed_data.append((filepath, email_data, intent, context, primary_intent, all_intents))

        # Parallelize the remaining steps
        with Pool() as pool:
            results = pool.map(process_single_email, preprocessed_data)

        results = [r for r in results if r is not None]
        logger.info(f"Processed {len(results)} emails successfully.")

        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)
        logger.info("Results saved to results.json")
        return results
    except Exception as e:
        logger.error(f"Error in email processing pipeline: {str(e)}")
        return []


# Flask routes
@app.route('/')
def upload_page():
    logger.info("Serving upload page.")
    return render_template('upload.html')


@app.route('/upload', methods=['GET'])
def upload_get():
    logger.info("Received GET request for /upload, redirecting to upload page.")
    flash("Please use the form to upload files.", "info")
    return redirect(url_for('upload_page'))


@app.route('/upload', methods=['POST'])
def upload_files():
    logger.info("Received file upload request.")
    try:
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            flash("No files selected. Please select at least one EML file to upload.", "error")
            return redirect(url_for('upload_page'))

        for file in files:
            if file.filename.endswith('.eml'):
                file.save(os.path.join(UPLOAD_FOLDER, file.filename))
                logger.info(f"Uploaded file: {file.filename}")
        results = process_email_pipeline(UPLOAD_FOLDER)
        logger.info("Upload and processing completed.")
        return json.dumps(results)
    except Exception as e:
        logger.error(f"Error handling upload: {str(e)}")
        flash(f"Error processing files: {str(e)}", "error")
        return redirect(url_for('upload_page'))


if __name__ == "__main__":
    logger.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
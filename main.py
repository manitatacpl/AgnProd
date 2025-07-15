import ssl
import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import time
import gc
import psutil
from joblib import dump, load
import os
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless environments
import matplotlib.pyplot as plt
import requests
from group_mapping import get_group_id_by_name, get_group_name_by_id
from nltk.stem import WordNetLemmatizer
import string
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
import datetime
from sklearn.metrics.pairwise import cosine_similarity
import urllib3
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
import uvicorn

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Load environment variables from .env if present
load_dotenv()

FRESHSERVICE_API_KEY = os.environ.get("FRESHSERVICE_API_KEY")
FRESHSERVICE_DOMAIN = os.environ.get("FRESHSERVICE_DOMAIN")

if not FRESHSERVICE_API_KEY or not FRESHSERVICE_DOMAIN:
    print(f"FRESHSERVICE_API_KEY: {FRESHSERVICE_API_KEY}")
    print(f"FRESHSERVICE_DOMAIN: {FRESHSERVICE_DOMAIN}")
    raise RuntimeError("FRESHSERVICE_API_KEY or FRESHSERVICE_DOMAIN is not set. Please check your .env file.")

def setup_nltk():
    """Set up NLTK with error handling"""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            nltk.download('stopwords', quiet=True)
            nltk.download('names', quiet=True)
            nltk.download('wordnet', quiet=True)
        return True
    except Exception as e:
        return False

def print_stats(start_time=None):
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    print(f"CPU usage: {psutil.cpu_percent()}%")
    if start_time:
        print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

setup_nltk()

# Change input and output Excel filenames
file_path = r"6mosallstatus.xlsx"

try:
    print("\n=== Excel File Analysis ===")
    print(f"Reading file: {file_path}")
    print("Loading data...")
    start_time = time.time()

    xl = pd.ExcelFile(file_path)
    print(f"\nExcel Sheets found: {xl.sheet_names}")
    print(f"Reading from sheet: Sheet1")
    
    # Load all required columns
    df = pd.read_excel(
        file_path,
        sheet_name="Sheet1",
        usecols=['Department', 'Description', 'Group', 'Requester Email', 'Requester Location', 'Subject'],
        dtype=str
    )

    # Remove only rows where 'Group' is missing, keep all others for training
    df = df.dropna(subset=['Group'])

    # Print dataset information
    print("\nDataset Information:")
    print(f"Total rows detected: {len(df):,}")
    print("\nColumn Statistics:")
    for col in df.columns:
        null_count = df[col].isnull().sum()
        print(f"{col}:")
        print(f"  - Non-null entries: {df[col].count():,}")
        print(f"  - Null entries: {null_count:,}")
        if col in ['Group']:
            print(f"  - Unique values: {df[col].nunique():,}")
    
    print("\nMemory Usage:")
    print_stats(start_time)

except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Prepare custom stopwords and names
custom_stopwords = {
    "caution", "external", "mail", "do", "not", "click", "on", "links", "enter", "passwords", "team", "hi", "dear", "hello", "regards", "thanks"
}
try:
    from nltk.corpus import names
    nltk_names = set([n.lower() for n in names.words()])
except Exception:
    nltk_names = set()

def remove_phone_numbers(text):
    # Remove phone numbers (various formats)
    text = re.sub(r'\b\d{10,}\b', ' ', text)  # 10+ digit numbers
    text = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}\b', ' ', text)
    return text

def remove_names(text):
    # Remove names from NLTK names corpus
    words = text.split()
    filtered = [w for w in words if w not in nltk_names]
    return " ".join(filtered)

lemmatizer = WordNetLemmatizer()

# Preprocessing Function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'^(hi|hello|dear|greetings)[\s,:\-]+.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^(team[\s,:\-]+.*)$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'caution:.*external mail.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'do not click on links.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'external mail.*', '', text, flags=re.IGNORECASE)
    text = remove_phone_numbers(text)
    text = ' '.join(text.lower().split())
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\b[a-zA-Z0-9]\b', '', text)
    text = re.sub(r'[\s\-]+', ' ', text).strip()
    # Remove names
    text = remove_names(text)
    # Remove stopwords
    stop_word_set = set(stopwords.words('english')).union(custom_stopwords)
    words = [word for word in text.split() if word not in stop_word_set]
    # Remove words with length < 3
    words = [word for word in words if len(word) > 2]
    # Strip leading/trailing punctuation from words
    words = [word.strip(string.punctuation) for word in words]
    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]

    seen = set()
    words = [x for x in words if not (x in seen or seen.add(x))]
    return " ".join(words)

print("\nPreprocessing text...")
start_time = time.time()

# Preprocessing: concatenate all available fields for training
def build_full_text(row):
    parts = [
        str(row.get('Department', '')),
        str(row.get('Requester Email', '')),
        str(row.get('Requester Location', '')),
        str(row.get('Subject', '')),
        str(row.get('Description', ''))
    ]
    return " ".join([p for p in parts if p and p.lower() != 'nan'])

df['text'] = df.apply(build_full_text, axis=1)
df['text'] = df['text'].astype(str)

batch_size = 1000
for i in tqdm(range(0, len(df), batch_size)):
    batch = df.iloc[i:i+batch_size]
    df.iloc[i:i+batch_size, df.columns.get_loc('text')] = batch['text'].apply(preprocess_text)

print_stats(start_time)

processed_excel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.xlsx")
df.to_excel(processed_excel_path, index=False)
print(f"\nProcessed data saved to: {processed_excel_path}")

df = df[df['text'].str.strip() != '']
df = df[df['Group'].notna()]

group_counts = df['Group'].value_counts()
valid_groups = group_counts[group_counts >= 5].index
df = df[df['Group'].isin(valid_groups)]

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models")
os.makedirs(MODEL_DIR, exist_ok=True)
INITIAL_MODEL_PATH = os.path.join(MODEL_DIR, "svm_group_initial.joblib")
INITIAL_VEC_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer_initial.joblib")
INITIAL_LABEL_PATH = os.path.join(MODEL_DIR, "label_encoder_initial.joblib")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "svm_group_final.joblib")
FINAL_VEC_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer_final.joblib")
FINAL_LABEL_PATH = os.path.join(MODEL_DIR, "label_encoder_final.joblib")

if os.path.exists(FINAL_MODEL_PATH) and os.path.exists(FINAL_VEC_PATH) and os.path.exists(FINAL_LABEL_PATH):
    print("\nLoading pretrained final model...")
    svm = load(FINAL_MODEL_PATH)
    vectorizer = load(FINAL_VEC_PATH)
    label_encoder = load(FINAL_LABEL_PATH)
else:

    all_words = " ".join(df['text']).split()
    word_counts = pd.Series(all_words).value_counts()
    rare_words = set(word_counts[word_counts == 1].index)
    df['text'] = df['text'].apply(lambda t: " ".join([w for w in t.split() if w not in rare_words]))

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Group'])
    X = df['text']

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2),   # Use unigrams and bigrams
        min_df=2,            # Ignore rare words
        max_df=0.95          # Ignore very frequent words
    )
    X_vec = vectorizer.fit_transform(X)
    feature_names = vectorizer.get_feature_names_out()
    with open("selected_words.txt", "w", encoding="utf-8") as f:
        for word in feature_names:
            f.write(word + "\n")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nTraining SVM classifier...")
    # Try SVM with GridSearchCV for hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'probability': [True]
    }
    svm = SVC(random_state=42)
    grid = GridSearchCV(svm, param_grid, cv=2, n_jobs=-1, scoring='accuracy')
    grid.fit(X_train, y_train)
    svm = grid.best_estimator_
    print(f"\nBest SVM params: {grid.best_params_}")

    # Optionally, try Logistic Regression (often works well for text)
    # Uncomment below to use Logistic Regression instead of SVM
    # logreg = LogisticRegression(max_iter=1000, random_state=42)
    # logreg.fit(X_train, y_train)
    # y_pred = logreg.predict(X_test)
    # acc = accuracy_score(y_test, y_pred)
    # print(f"\nLogistic Regression Accuracy: {acc:.2%}")

    # Cross-validation for more reliable accuracy
    cv_scores = cross_val_score(svm, X_vec, y, cv=5, scoring='accuracy')
    print(f"\nCross-validated accuracy: {np.mean(cv_scores):.2%} (+/- {np.std(cv_scores):.2%})")

    # Save initial model
    dump(svm, INITIAL_MODEL_PATH)
    dump(vectorizer, INITIAL_VEC_PATH)
    dump(label_encoder, INITIAL_LABEL_PATH)

    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy on test set: {acc:.2%}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Group')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved as confusion_matrix.png")

    print("\nClassification Report:")

    present_labels = np.unique(np.concatenate([y_test, y_pred]))
    present_class_names = label_encoder.inverse_transform(present_labels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = classification_report(
            y_test, y_pred, 
            labels=present_labels, 
            target_names=present_class_names,
            zero_division=0,
            output_dict=True
        )
        print(classification_report(
            y_test, y_pred, 
            labels=present_labels, 
            target_names=present_class_names,
            zero_division=0
        ))
    dump(svm, FINAL_MODEL_PATH)
    dump(vectorizer, FINAL_VEC_PATH)
    dump(label_encoder, FINAL_LABEL_PATH)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy on test set: {acc:.2%}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Group')
    plt.show(block=True)  
    print("\nClassification Report:")

    present_labels = np.unique(np.concatenate([y_test, y_pred]))
    present_class_names = label_encoder.inverse_transform(present_labels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = classification_report(
            y_test, y_pred, 
            labels=present_labels, 
            target_names=present_class_names,
            zero_division=0,
            output_dict=True
        )
        print(classification_report(
            y_test, y_pred, 
            labels=present_labels, 
            target_names=present_class_names,
            zero_division=0
        ))
    dump(svm, FINAL_MODEL_PATH)
    dump(vectorizer, FINAL_VEC_PATH)
    dump(label_encoder, FINAL_LABEL_PATH)


# Alert this is Production Urls
# FRESHSERVICE_API_KEY = "nkfxbaNF223ZpiJUMURL"
# FRESHSERVICE_DOMAIN = "https://tataconsumerproductshelpdesk.freshservice.com"

FRESHSERVICE_API_KEY = os.environ.get("FRESHSERVICE_API_KEY", "EXHRMlmZPfust7PElttj")
FRESHSERVICE_DOMAIN = os.environ.get("FRESHSERVICE_DOMAIN", "https://060helpdesk.freshservice.com/")

FRESHSERVICE_API_URL = f"{FRESHSERVICE_DOMAIN}/api/v2/tickets"
REQUESTER_API_URL = f"{FRESHSERVICE_DOMAIN}/api/v2/requesters"
PENDING_TICKETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pending_tickets.xlsx")
EXCEL_MONITOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output_result.xlsx")
AUTO_UPDATE_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto_update_log.xlsx")

def get_requester_info(requester_id):
    url = f"{REQUESTER_API_URL}/{requester_id}"
    headers = {
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(url, auth=(FRESHSERVICE_API_KEY, "X"), headers=headers, verify=False)
        if response.status_code == 200:
            data = response.json()
            requester = data.get("requester", data)
            # Extract email
            email = requester.get("primary_email", "") or requester.get("email", "")
            # Extract location (try location_name, then address)
            location = requester.get("location_name", "") or requester.get("address", "")
            # Extract department (prefer department_names[0], then department_name, then department)
            dept_names = requester.get("department_names", [])
            if isinstance(dept_names, list) and dept_names:
                department = dept_names[0]
            else:
                department = requester.get("department_name", "") or requester.get("department", "")
            return {
                "email": email,
                "location": location,
                "department_name": department
            }
        else:
            # Always return empty fields if not found
            return {"email": "", "location": "", "department_name": ""}
    except Exception as e:
        # Always return empty fields on error
        return {"email": "", "location": "", "department_name": ""}

def fetch_tickets_from_freshservice():
    headers = {
        "Content-Type": "application/json"
    }
    all_tickets = []
    page = 1
    per_page = 100  # Max allowed by Freshservice
    max_tickets = 1000  # Fetch only last 1000 tickets
    while len(all_tickets) < max_tickets:
        params = {
            "page": page,
            "per_page": per_page
        }
        try:
            response = requests.get(
                FRESHSERVICE_API_URL,
                auth=(FRESHSERVICE_API_KEY, "X"),
                headers=headers,
                params=params,
                verify=False
            )
            if response.status_code == 200:
                data = response.json()
                tickets = data.get("tickets", [])
                if not tickets:
                    break
                # --- REMOVE FILTER: fetch all tickets, do not filter by status or source ---
                all_tickets.extend(tickets)
                if len(tickets) < per_page:
                    break
                page += 1
            else:
                print(f"Failed to fetch tickets: {response.status_code} {response.text}")
                break
        except requests.exceptions.SSLError as ssl_err:
            print("WARNING: SSL verification failed. SSL verification is disabled for this request.")
            print(f"SSL Error: {ssl_err}")
            break
        except Exception as e:
            print(f"Error fetching tickets: {e}")
            break
    # Only keep the most recent 1000 tickets
    return all_tickets[:max_tickets]

def get_group_name_by_id_api(group_id):

    try:
        return get_group_name_by_id(group_id)
    except Exception:
        return str(group_id)

def safe_to_excel(df, path, **kwargs):
    while True:
        try:
            df.to_excel(path, **kwargs)
            break
        except PermissionError:
            print(f"Cannot write to {path} (file is open). Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"Error writing to {path}: {e}. Retrying in 5 seconds...")
            time.sleep(5)

def monitor_tickets():
    print("\nMonitoring tickets from Freshservice API...")

    monitor_columns = [
        "Ticket ID", "Requester ID", "Requester Email", "Requester Location", "Requester Department", "Date", "Time", "Subject", "Description",
        "Assigned Group by Agent", "Predicted Group by ML", "Confidence", "Is Matched?"
    ]
    pending_columns = [
        "Ticket ID", "Requester ID", "Requester Email", "Requester Location", "Requester Department", "Date", "Time", "Subject", "Description", "Predicted Group by ML", "Confidence"
    ]

    # Load or create the Excel files for monitoring and pending tickets
    if os.path.exists(EXCEL_MONITOR_PATH):
        monitor_df = pd.read_excel(EXCEL_MONITOR_PATH)
        if "Ticket ID" not in monitor_df.columns:
            monitor_df = pd.DataFrame(columns=monitor_columns)
    else:
        monitor_df = pd.DataFrame(columns=monitor_columns)
    if os.path.exists(PENDING_TICKETS_PATH):
        pending_df = pd.read_excel(PENDING_TICKETS_PATH)
        if "Ticket ID" not in pending_df.columns:
            pending_df = pd.DataFrame(columns=pending_columns)
    else:
        pending_df = pd.DataFrame(columns=pending_columns)

    def print_ticket_row(row):
        # Print ticket info in tabular format
        print(
            f"{row['Ticket ID']}\t{row['Requester ID']}\t{row['Requester Email']}\t{row['Requester Location']}\t{row['Requester Department']}\t"
            f"{row['Subject']}\t{row['Description']}\t{row.get('Assigned Group by Agent', '')}\t{row.get('Predicted Group by ML', '')}\t{row.get('Confidence', '')}\t{row.get('Is Matched?', '')}"
        )

    # Print header for tabular output
    header = [
        "Ticket ID", "Requester ID", "Requester Email", "Requester Location", "Requester Department",
        "Subject", "Description", "Assigned Group", "Predicted Group", "Confidence", "Matched"
    ]
    print("\t".join(header))

    while True:
        tickets = fetch_tickets_from_freshservice()
        now = datetime.datetime.now()
        pending_ids = set(pending_df["Ticket ID"]) if not pending_df.empty else set()
        monitor_ids = set(monitor_df["Ticket ID"]) if not monitor_df.empty else set()

        for ticket in tickets:
            # Only process tickets with status == 5 and source != 14
            if ticket.get("status") != 5 or ticket.get("source") == 14:
                continue
            created_at = ticket.get("created_at")
            if created_at:
                try:
                    created_dt = pd.to_datetime(created_at)
                    if created_dt.tzinfo is not None:
                        created_dt = created_dt.replace(tzinfo=None)
                except Exception:
                    created_dt = now
            else:
                created_dt = now

            ticket_id = ticket.get("id")
            subject = ticket.get("subject", "")
            description = ticket.get("description", "")
            group_id = ticket.get("group_id")
            requester_id = ticket.get("requester_id")
            updated_at = ticket.get("updated_at") or ticket.get("created_at")
            if updated_at:
                updated_dt = pd.to_datetime(updated_at)
                if updated_dt.tzinfo is not None:
                    updated_dt = updated_dt.replace(tzinfo=None)
            else:
                updated_dt = now
            # Only date part for Excel
            date_str = created_dt.date() if created_at else updated_dt.date()
            time_str = updated_dt.time().strftime("%H:%M:%S")

            # Always fetch requester info (will be empty if not found)
            requester_info = get_requester_info(requester_id) if requester_id else {"email": "", "location": "", "department_name": ""}
            requester_email = requester_info.get("email", "")
            requester_location = requester_info.get("location", "")
            requester_department = requester_info.get("department_name", "")

            # Predict group using only the ML model (no top matching logic)
            predicted_group, confidence_score = predict_ticket(
                description=description,
                subject=subject,
                requester_id=requester_id
            )
            predicted_group_name = predicted_group.get("Group", "")
            confidence_percent = f"{int(round(confidence_score.get('Group', 0) * 100))}%"

            if not group_id:
                if ticket_id not in pending_ids:
                    row = {
                        "Ticket ID": ticket_id,
                        "Requester ID": requester_id if requester_id else "",
                        "Requester Email": requester_email,
                        "Requester Location": requester_location,
                        "Requester Department": requester_department,
                        "Date": date_str,
                        "Time": time_str,
                        "Subject": subject,
                        "Description": description,
                        "Predicted Group by ML": predicted_group_name,
                        "Confidence": confidence_percent
                    }
                    pending_df = pd.concat([pending_df, pd.DataFrame([row])], ignore_index=True)
                    print_ticket_row(row)
            else:
                assigned_group_name = get_group_name_by_id_api(group_id)
                is_matched = assigned_group_name == predicted_group_name
                if ticket_id not in monitor_ids:
                    row = {
                        "Ticket ID": ticket_id,
                        "Requester ID": requester_id if requester_id else "",
                        "Requester Email": requester_email,
                        "Requester Location": requester_location,
                        "Requester Department": requester_department,
                        "Date": date_str,
                        "Time": time_str,
                        "Subject": subject,
                        "Description": description,
                        "Assigned Group by Agent": assigned_group_name,
                        "Predicted Group by ML": predicted_group_name,
                        "Confidence": confidence_percent,
                        "Is Matched?": is_matched
                    }
                    monitor_df = pd.concat([monitor_df, pd.DataFrame([row])], ignore_index=True)
                    print_ticket_row(row)
                if ticket_id in pending_ids:
                    pending_df = pending_df[pending_df["Ticket ID"] != ticket_id]

        safe_to_excel(pending_df, PENDING_TICKETS_PATH, index=False)
        safe_to_excel(monitor_df, EXCEL_MONITOR_PATH, index=False)
        print(f"Excel files updated at {now.strftime('%Y-%m-%d %H:%M:%S')}. Waiting for new tickets...")
        time.sleep(60)  # Check every 60 seconds

def predict_ticket(description, subject, requester_id=None):
    try:
        # Fetch requester info if requester_id is provided
        department = ""
        requester_email = ""
        requester_location = ""
        if requester_id:
            requester_info = get_requester_info(requester_id)
            department = requester_info.get("department_name", "") or requester_info.get("department", "")
            requester_email = requester_info.get("email", "")
            requester_location = requester_info.get("location", "")

        # Build full text for prediction
        parts = [
            str(department),
            str(requester_email),
            str(requester_location),
            str(subject) if subject else "",
            str(description) if description else ""
        ]
        full_text = " ".join([p for p in parts if p and p.lower() != 'nan'])
        full_text_processed = preprocess_text(full_text)
        X_input = vectorizer.transform([full_text_processed])

        # Always use ML model for prediction (no top 5 matching logic)
        pred = svm.predict(X_input)
        prob = svm.predict_proba(X_input)
        label = label_encoder.inverse_transform(pred)[0]
        confidence = np.max(prob)
        return {'Group': label}, {'Group': confidence}
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {'Group': 'Unknown'}, {'Group': 0.0}

def test_model(tickets):
    print("\nAnalyzing Ticket:")
    print("=" * 50)
    for ticket in tickets:
        print(f"\nInput Details:")
        print(f"Subject: {ticket['subject']}")
        print(f"Description: {ticket['description']}")
        predicted_group, confidence_score = predict_ticket(
            ticket['description'],
            ticket['subject']
        )
        print("\nPrediction Result:")
        print(f"Recommended Group: {predicted_group.get('Group', 'Unknown')}")
        print(f"Confidence: {confidence_score.get('Group', 0):.2%}")
        print("=" * 50)

def show_last_completed_tickets():
    if os.path.exists(EXCEL_MONITOR_PATH):
        df = pd.read_excel(EXCEL_MONITOR_PATH)
        if not df.empty:
            print(f"\nLast 1000 completed tickets:")
            print(df.tail(1000).to_string(index=False))
        else:
            print("\nNo completed tickets found.")
    else:
        print("\nNo completed tickets file found.")

def auto_update_new_tickets():
    print("\nAuto-updating new tickets with predicted group...")
    processed_ids = set()
    # Load existing log to avoid duplicate updates
    if os.path.exists(AUTO_UPDATE_LOG_PATH):
        try:
            log_df = pd.read_excel(AUTO_UPDATE_LOG_PATH)
            if "Ticket ID" in log_df.columns:
                processed_ids = set(log_df["Ticket ID"].astype(str))
        except Exception:
            log_df = pd.DataFrame(columns=[
                "Timestamp", "Ticket ID", "Predicted Group", "Group ID", "Confidence", "API Status", "API Response"
            ])
    else:
        log_df = pd.DataFrame(columns=[
            "Timestamp", "Ticket ID", "Predicted Group", "Group ID", "Confidence", "API Status", "API Response"
        ])

    # --- Only update tickets created after script start ---
    # Find the max ticket ID at startup (assumes ticket IDs are increasing)
    tickets_at_start = fetch_tickets_from_freshservice()
    if tickets_at_start:
        try:
            max_ticket_id_at_start = max(int(t.get("id", 0)) for t in tickets_at_start if t.get("id"))
        except Exception:
            max_ticket_id_at_start = 0
    else:
        max_ticket_id_at_start = 0

    AGNEYA_AI_GROUP_ID = 151758
    AGNEYA_AI_GROUP_NAME = "AgneyaAI"

    # Define custom_fields once
    custom_fields = {
        "business_service": "Digital Application Services",
        "product_category_tier_1": "AgneyaAI Execution",
        "ops_categorization_tier_1": "AgneyaAI Flow Control"
    }

    while True:
        tickets = fetch_tickets_from_freshservice()
        for ticket in tickets:
            # Remove all source/status conditions: process every ticket
            ticket_id_raw = ticket.get("id")
            if not ticket_id_raw:
                continue
            try:
                ticket_id_int = int(ticket_id_raw)
            except Exception:
                continue
            ticket_id = str(ticket_id_raw)
            group_id_api = ticket.get("group_id")
            # Only process tickets with id > max_ticket_id_at_start and not already processed
            if (ticket_id_int > max_ticket_id_at_start) and (ticket_id not in processed_ids):
                subject = ticket.get("subject", "")
                description = ticket.get("description", "")
                requester_id = ticket.get("requester_id")
                # Predict group and confidence
                predicted_group, confidence_score = predict_ticket(
                    description=description,
                    subject=subject,
                    requester_id=requester_id
                )
                predicted_group_name = predicted_group.get("Group", "")
                confidence_percent = f"{int(round(confidence_score.get('Group', 0) * 100))}%"
                # Get group_id from group_mapping
                try:
                    group_id = get_group_id_by_name(predicted_group_name)
                except Exception:
                    group_id = None
                headers = {"Content-Type": "application/json"}
                update_url = f"{FRESHSERVICE_API_URL}/{ticket_id}"

                # 1. If not already assigned to AgneyaAI, assign it first
                if group_id_api != AGNEYA_AI_GROUP_ID:
                    update_payload_agneya = {
                        "group_id": int(AGNEYA_AI_GROUP_ID),
                        "custom_fields": custom_fields
                    }
                    try:
                        response_agneya = requests.put(
                            update_url,
                            auth=(FRESHSERVICE_API_KEY, "X"),
                            headers=headers,
                            json=update_payload_agneya,
                            verify=False
                        )
                        api_status_agneya = response_agneya.status_code
                        api_response_agneya = response_agneya.text
                        if api_status_agneya == 400:
                            print(f"DEBUG: Payload sent for AgneyaAI assignment: {update_payload_agneya}")
                            print(f"DEBUG: Response: {api_response_agneya}")
                    except Exception as e:
                        api_status_agneya = "ERROR"
                        api_response_agneya = str(e)
                    log_row_agneya = {
                        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Ticket ID": ticket_id,
                        "Predicted Group": AGNEYA_AI_GROUP_NAME,
                        "Group ID": AGNEYA_AI_GROUP_ID,
                        "Confidence": "",
                        "API Status": api_status_agneya,
                        "API Response": api_response_agneya
                    }
                    log_df = pd.concat([log_df, pd.DataFrame([log_row_agneya])], ignore_index=True)
                    print(f"Assigned Ticket {ticket_id} to '{AGNEYA_AI_GROUP_NAME}' (ID={AGNEYA_AI_GROUP_ID}), API Status={api_status_agneya}")
                    try:
                        log_df.to_excel(AUTO_UPDATE_LOG_PATH, index=False)
                    except Exception:
                        pass
                    # Wait a moment before next update to allow system to register the change
                    time.sleep(1)
                    # Fetch the ticket again to get updated group_id
                    refreshed = False
                    for _ in range(5):
                        refreshed_tickets = fetch_tickets_from_freshservice()
                        refreshed_ticket = next((t for t in refreshed_tickets if str(t.get("id")) == ticket_id), None)
                        if refreshed_ticket and refreshed_ticket.get("group_id") == AGNEYA_AI_GROUP_ID:
                            group_id_api = AGNEYA_AI_GROUP_ID
                            refreshed = True
                            break
                        time.sleep(1)
                    if not refreshed:
                        print(f"Warning: Ticket {ticket_id} did not update to AgneyaAI group in time.")
                    # Continue to predicted group update below

                # 2. Now update to predicted group if not already assigned
                if group_id_api != group_id and group_id is not None:
                    update_payload_pred = {
                        "group_id": int(group_id)
                        # Do NOT include custom_fields here
                    }
                    try:
                        response_pred = requests.put(
                            update_url,
                            auth=(FRESHSERVICE_API_KEY, "X"),
                            headers=headers,
                            json=update_payload_pred,
                            verify=False
                        )
                        api_status_pred = response_pred.status_code
                        api_response_pred = response_pred.text
                        if api_status_pred == 400:
                            print(f"DEBUG: Payload sent for predicted group: {update_payload_pred}")
                            print(f"DEBUG: Response: {api_response_pred}")
                    except Exception as e:
                        api_status_pred = "ERROR"
                        api_response_pred = str(e)
                    log_row_pred = {
                        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Ticket ID": ticket_id,
                        "Predicted Group": predicted_group_name,
                        "Group ID": group_id,
                        "Confidence": confidence_percent,
                        "API Status": api_status_pred,
                        "API Response": api_response_pred
                    }
                    log_df = pd.concat([log_df, pd.DataFrame([log_row_pred])], ignore_index=True)
                    print(f"Updated Ticket {ticket_id}: Group='{predicted_group_name}' (ID={group_id}), Confidence={confidence_percent}, API Status={api_status_pred}")
                    try:
                        log_df.to_excel(AUTO_UPDATE_LOG_PATH, index=False)
                    except Exception:
                        pass
                processed_ids.add(ticket_id)
        time.sleep(1)

app = FastAPI()

@app.post("/webhook/ticket")
async def webhook_ticket(request: Request):
    """
    Receives ticket info from Freshservice Workflow Automator webhook.
    Expects JSON payload with ticket fields.
    Updates ticket group in Freshservice after prediction.
    """
    data = await request.json()
    ticket_id_raw = data.get("id")
    subject = data.get("subject", "")
    description = data.get("description", "")
    requester_id = data.get("requester_id")
    # You can add more fields as needed

    # Remove INC- prefix if present
    ticket_id = str(ticket_id_raw)
    if ticket_id.startswith("INC-"):
        ticket_id_api = ticket_id.replace("INC-", "")
    else:
        ticket_id_api = ticket_id

    # Predict group and confidence (do not preprocess text twice)
    predicted_group, confidence_score = predict_ticket(
        description=description,
        subject=subject,
        requester_id=requester_id
    )
    predicted_group_name = predicted_group.get("Group", "")
    confidence_percent = f"{int(round(confidence_score.get('Group', 0) * 100))}%"

    print(f"Webhook received for Ticket {ticket_id}: Predicted Group={predicted_group_name}, Confidence={confidence_percent}")

    # --- Assign to AgneyaAI group first ---
    AGNEYA_AI_GROUP_ID = 151758
    AGNEYA_AI_GROUP_NAME = "AgneyaAI"
    custom_fields = {
        "business_service": "Digital Application Services",
        "product_category_tier_1": "AgneyaAI Execution",
        "ops_categorization_tier_1": "AgneyaAI Flow Control"
    }
    update_url = f"{FRESHSERVICE_API_URL}/{ticket_id_api}"
    headers = {"Content-Type": "application/json"}
    payload_agneya = {
        "group_id": int(AGNEYA_AI_GROUP_ID),
        "custom_fields": custom_fields
    }
    print(f"DEBUG: Update URL (AgneyaAI): {update_url}")
    print(f"DEBUG: Payload (AgneyaAI): {payload_agneya}")
    try:
        response_agneya = requests.put(
            update_url,
            auth=(FRESHSERVICE_API_KEY, "X"),
            headers=headers,
            json=payload_agneya,
            verify=False
        )
        print(f"DEBUG: Response Status (AgneyaAI): {response_agneya.status_code}")
        # print(f"DEBUG: Response Body (AgneyaAI): {response_agneya.text}")
        print(f"Assigned Ticket {ticket_id} to '{AGNEYA_AI_GROUP_NAME}' (ID={AGNEYA_AI_GROUP_ID}), API Status={response_agneya.status_code}")
    except Exception as e:
        print(f"Error assigning AgneyaAI group for ticket {ticket_id}: {e}")

    # --- Assign to predicted group ---
    try:
        group_id = get_group_id_by_name(predicted_group_name)
        payload_pred = {"group_id": int(group_id)} if group_id else {}
        print(f"DEBUG: Update URL (Predicted): {update_url}")
        print(f"DEBUG: Payload (Predicted): {payload_pred}")
        print(f"DEBUG: Ticket ID: {ticket_id_api}, Group ID: {group_id}")
        response_pred = requests.put(
            update_url,
            auth=(FRESHSERVICE_API_KEY, "X"),
            headers=headers,
            json=payload_pred,
            verify=False
        )
        print(f"DEBUG: Response Status (Predicted): {response_pred.status_code}")
        # print(f"DEBUG: Response Body (Predicted): {response_pred.text}")
        if group_id:
            print(f"Ticket {ticket_id} updated to group '{predicted_group_name}' (ID={group_id}), API Status={response_pred.status_code}")
        else:
            print(f"Could not find group_id for '{predicted_group_name}'. Ticket not updated.")
    except Exception as e:
        print(f"Error updating ticket {ticket_id}: {e}")

    # Return prediction result
    return {
        "ticket_id": ticket_id,
        "predicted_group": predicted_group_name,
        "confidence": confidence_percent
    }

def run_fastapi():
    """
    Run FastAPI app.
    """
    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=False)

if __name__ == "__main__":
    while True:
        print("\nChoose an option:")
        print("1. Test a ticket manually")
        print("2. Monitor tickets from Freshservice API")
        print("3. Exit")
        print("4. Auto-update new tickets with predicted group")
        print("5. Run FastAPI webhook server")  # <-- Added option 5
        choice = input("Enter your choice (1/2/3/4/5): ").strip()
        if choice == "1":
            try:
                test_tickets = []
                print("\nEnter ticket details (Ctrl+C to exit):")
                subject = input("Enter ticket subject: ").strip()
                description = input("\nEnter ticket description: ").strip()
                if description and subject:
                    test_tickets.append({
                        "description": description,
                        "subject": subject
                    })
                    test_model(test_tickets)
            except KeyboardInterrupt:
                print("\nExiting manual test...")
        elif choice == "2":
            try:
                show_last_completed_tickets()
                monitor_tickets()
            except KeyboardInterrupt:
                print("\nStopped monitoring.")
                break
        elif choice == "3":
            print("Exiting...")
            break
        elif choice == "4":
            try:
                auto_update_new_tickets()
            except KeyboardInterrupt:
                print("\nStopped auto-updating.")
                break
        elif choice == "5":
            print("Starting FastAPI webhook server...")
            run_fastapi()
            break
        else:
            print("Invalid choice. Please try again.")
            print("Invalid choice. Please try again.")
            print("Invalid choice. Please try again.")
            print("Exiting...")
            break

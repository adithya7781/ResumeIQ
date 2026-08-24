import hashlib
import json
import re
import secrets
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, UTC

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_cookies_manager import EncryptedCookieManager


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ResumeIQ",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATABASE_NAME = str(__import__("pathlib").Path(__file__).resolve().parent / "resumeiq.db")
MAX_FILE_SIZE = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".docx"}

SESSION_COOKIE_NAME = "resumeiq_session"
SESSION_DAYS = 7

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# COOKIE MANAGER
# ============================================================

try:
    COOKIE_PASSWORD = st.secrets["COOKIE_PASSWORD"]
except Exception:
    COOKIE_PASSWORD = "development-secret-change-me"

cookies = EncryptedCookieManager(
    prefix="resumeiq/",
    password=COOKIE_PASSWORD,
)

if not cookies.ready():
    st.stop()


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    return sqlite3.connect(DATABASE_NAME)


def initialize_database():
    """Create tables and migrate older ResumeIQ databases safely."""
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            mode TEXT NOT NULL,
            score REAL NOT NULL DEFAULT 0,
            result_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT ''
        )
        """
    )

    # Migrate an analyses table created by an older ResumeIQ version.
    cursor.execute("PRAGMA table_info(analyses)")
    columns = {row[1] for row in cursor.fetchall()}

    if "result_json" not in columns:
        cursor.execute(
            "ALTER TABLE analyses ADD COLUMN result_json TEXT NOT NULL DEFAULT '{}'"
        )

    if "score" not in columns:
        cursor.execute(
            "ALTER TABLE analyses ADD COLUMN score REAL NOT NULL DEFAULT 0"
        )

    if "created_at" not in columns:
        cursor.execute(
            "ALTER TABLE analyses ADD COLUMN created_at TEXT NOT NULL DEFAULT ''"
        )

    if "user_id" not in columns:
        cursor.execute(
            "ALTER TABLE analyses ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0"
        )

    if "filename" not in columns:
        cursor.execute(
            "ALTER TABLE analyses ADD COLUMN filename TEXT NOT NULL DEFAULT ''"
        )

    if "mode" not in columns:
        cursor.execute(
            "ALTER TABLE analyses ADD COLUMN mode TEXT NOT NULL DEFAULT ''"
        )

    connection.commit()
    connection.close()


initialize_database()


# ============================================================
# AUTHENTICATION
# ============================================================

def hash_password(password):
    salt = secrets.token_bytes(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        100_000,
    )

    return salt.hex() + ":" + password_hash.hex()


def verify_password(password, stored_password):
    try:
        salt_hex, hash_hex = stored_password.split(":")

        salt = bytes.fromhex(salt_hex)

        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            100_000,
        )

        return secrets.compare_digest(
            password_hash.hex(),
            hash_hex,
        )
    except Exception:
        return False


def hash_session_token(token):
    return hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()


def create_login_session(user_id):
    token = secrets.token_urlsafe(48)
    token_hash = hash_session_token(token)

    expires_at = (
        datetime.now(UTC) + timedelta(days=SESSION_DAYS)
    ).isoformat()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        "DELETE FROM sessions WHERE user_id = ?",
        (user_id,),
    )

    cursor.execute(
        """
        INSERT INTO sessions
        (user_id, token_hash, expires_at, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            user_id,
            token_hash,
            expires_at,
            datetime.now(UTC).isoformat(),
        ),
    )

    connection.commit()
    connection.close()

    cookies[SESSION_COOKIE_NAME] = token
    cookies.save()


def restore_login():
    token = cookies.get(SESSION_COOKIE_NAME)

    if not token:
        return None

    token_hash = hash_session_token(token)

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT u.id, u.username, u.email, s.expires_at
        FROM sessions s
        JOIN users u ON u.id = s.user_id
        WHERE s.token_hash = ?
        """,
        (token_hash,),
    )

    user = cursor.fetchone()
    connection.close()

    if not user:
        return None

    try:
        expires_at = datetime.fromisoformat(user[3])
        # Backward compatibility with sessions created before UTC-aware timestamps.
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
    except (ValueError, TypeError):
        delete_login_session()
        return None

    if datetime.now(UTC) >= expires_at:
        delete_login_session()
        return None

    return {
        "id": user[0],
        "username": user[1],
        "email": user[2],
    }


def delete_login_session():
    token = cookies.get(SESSION_COOKIE_NAME)

    if token:
        token_hash = hash_session_token(token)

        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            "DELETE FROM sessions WHERE token_hash = ?",
            (token_hash,),
        )

        connection.commit()
        connection.close()

    try:
        del cookies[SESSION_COOKIE_NAME]
        cookies.save()
    except Exception:
        pass


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "logged_in": False,
    "user_id": None,
    "username": None,
    "email": None,
    "resume_text": None,
    "resume_filename": None,
    "nlp_result": None,
    "ats_result": None,
    "jd_result": None,
    "job_description": "",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


if not st.session_state.logged_in:
    saved_user = restore_login()

    if saved_user:
        st.session_state.logged_in = True
        st.session_state.user_id = saved_user["id"]
        st.session_state.username = saved_user["username"]
        st.session_state.email = saved_user["email"]


def logout():
    delete_login_session()

    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = value

    st.rerun()


# ============================================================
# USER FUNCTIONS
# ============================================================

def validate_email(email):
    return bool(
        re.match(
            r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
            email,
        )
    )


def create_user(username, email, password):
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO users
            (username, email, password_hash, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                username,
                email,
                hash_password(password),
                datetime.now(UTC).isoformat(),
            ),
        )

        connection.commit()
        return True, "Account created successfully."

    except sqlite3.IntegrityError:
        return False, "Username or email already exists."

    finally:
        connection.close()


def authenticate_user(email, password):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, username, email, password_hash
        FROM users
        WHERE email = ?
        """,
        (email,),
    )

    user = cursor.fetchone()
    connection.close()

    if not user:
        return None

    if not verify_password(password, user[3]):
        return None

    return {
        "id": user[0],
        "username": user[1],
        "email": user[2],
    }


# ============================================================
# AUTHENTICATION UI
# ============================================================

def show_authentication():
    st.title("📄 ResumeIQ")
    st.caption("AI-powered resume intelligence")
    st.divider()

    login_tab, signup_tab = st.tabs(
        ["🔐 Sign In", "✨ Create Account"]
    )

    with login_tab:
        st.subheader("Welcome back")

        email = st.text_input(
            "Email",
            key="login_email",
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password",
        )

        if st.button(
            "Sign In",
            width="stretch",
        ):
            if not email or not password:
                st.warning("Enter your email and password.")
            else:
                user = authenticate_user(
                    email.strip().lower(),
                    password,
                )

                if user:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user["id"]
                    st.session_state.username = user["username"]
                    st.session_state.email = user["email"]

                    create_login_session(user["id"])
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

    with signup_tab:
        st.subheader("Create your account")

        username = st.text_input("Username")
        email = st.text_input("Email", key="signup_email")

        password = st.text_input(
            "Password",
            type="password",
            key="signup_password",
        )

        confirm = st.text_input(
            "Confirm Password",
            type="password",
            key="confirm_password",
        )

        if st.button(
            "Create Account",
            width="stretch",
        ):
            if len(username.strip()) < 3:
                st.warning(
                    "Username must contain at least 3 characters."
                )
            elif not validate_email(email.strip()):
                st.warning("Enter a valid email.")
            elif len(password) < 8:
                st.warning(
                    "Password must contain at least 8 characters."
                )
            elif password != confirm:
                st.warning("Passwords do not match.")
            else:
                success, message = create_user(
                    username.strip(),
                    email.strip().lower(),
                    password,
                )

                if success:
                    st.success(message)
                    st.info("Go to Sign In.")
                else:
                    st.error(message)


if not st.session_state.logged_in:
    show_authentication()
    st.stop()


# ============================================================
# FILE EXTRACTION
# ============================================================

def validate_file(uploaded_file):
    if uploaded_file is None:
        return False, "Upload a resume."

    extension = "." + uploaded_file.name.lower().split(".")[-1]

    if extension not in ALLOWED_EXTENSIONS:
        return False, "Only PDF and DOCX files are supported."

    if uploaded_file.size > MAX_FILE_SIZE:
        return False, "Maximum file size is 5 MB."

    return True, "Valid"


def extract_pdf_text(uploaded_file):
    from pypdf import PdfReader

    reader = PdfReader(uploaded_file)
    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def extract_docx_text(uploaded_file):
    from docx import Document

    document = Document(uploaded_file)
    paragraphs = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            paragraphs.append(paragraph.text.strip())

    return "\n".join(paragraphs)


def extract_resume_text(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
        return extract_pdf_text(uploaded_file)

    return extract_docx_text(uploaded_file)


def clean_text(text):
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = []

    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


def normalize_text(text):
    return re.sub(
        r"\s+",
        " ",
        text.lower(),
    ).strip()


# ============================================================
# RESUME SECTION DETECTION
# ============================================================

SECTION_PATTERNS = {
    "summary": [
        "summary",
        "professional summary",
        "career summary",
        "profile",
        "professional profile",
        "objective",
        "career objective",
        "about me",
    ],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history",
        "career history",
    ],
    "education": [
        "education",
        "educational background",
        "academic background",
        "academic qualifications",
        "qualifications",
    ],
    "skills": [
        "skills",
        "skill set",
        "technical skills",
        "core skills",
        "key skills",
        "professional skills",
        "technical expertise",
        "technologies",
        "competencies",
        "core competencies",
    ],
    "projects": [
        "projects",
        "project",
        "academic projects",
        "personal projects",
        "project experience",
    ],
    "certifications": [
        "certifications",
        "certification",
        "certificates",
        "professional certifications",
    ],
    "achievements": [
        "achievements",
        "achievement",
        "awards",
        "honors",
        "accomplishments",
    ],
    "publications": [
        "publications",
        "research publications",
        "papers",
        "research",
    ],
    "languages": [
        "languages",
        "language skills",
    ],
    "interests": [
        "interests",
        "hobbies",
        "personal interests",
    ],
}


def normalize_heading(line):
    line = line.lower().strip()

    line = re.sub(
        r"^[•●▪■◆◦*\-–—]+\s*",
        "",
        line,
    )

    line = re.sub(
        r"^\d+[\s.)\-:]+",
        "",
        line,
    )

    line = re.sub(
        r"[^a-zA-Z ]",
        "",
        line,
    )

    return re.sub(
        r"\s+",
        " ",
        line,
    ).strip()


def detect_heading(line):
    heading = normalize_heading(line)

    for section, patterns in SECTION_PATTERNS.items():
        if heading in patterns:
            return section

    return None


def detect_sections(text):
    sections = {}
    current = None

    for line in text.splitlines():
        section = detect_heading(line)

        if section:
            current = section
            sections.setdefault(section, [])
            continue

        if current:
            sections[current].append(line)

    return {
        key: "\n".join(value).strip()
        for key, value in sections.items()
        if "\n".join(value).strip()
    }


# ============================================================
# SKILL DETECTION
# ============================================================

SKILL_DATABASE = {
    "Technical Skills": [
        "python",
        "java",
        "javascript",
        "typescript",
        "c++",
        "c#",
        "sql",
        "html",
        "css",
        "react",
        "angular",
        "node.js",
        "api",
        "rest api",
        "graphql",
    ],
    "Data & Analytics": [
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "power bi",
        "tableau",
        "excel",
        "statistics",
        "data analysis",
        "data visualization",
        "business intelligence",
        "data analytics",
        "eda",
        "dax",
        "power query",
        "etl",
        "elt",
    ],
    "AI & Machine Learning": [
        "machine learning",
        "deep learning",
        "natural language processing",
        "nlp",
        "artificial intelligence",
        "generative ai",
        "large language model",
        "llm",
        "computer vision",
        "tensorflow",
        "pytorch",
        "scikit-learn",
        "langchain",
        "langgraph",
        "hugging face",
        "transformers",
    ],
    "Databases": [
        "mysql",
        "postgresql",
        "oracle",
        "mongodb",
        "sqlite",
        "redis",
        "sql server",
    ],
    "Cloud & DevOps": [
        "aws",
        "azure",
        "google cloud",
        "gcp",
        "docker",
        "kubernetes",
        "terraform",
        "ci/cd",
    ],
    "Tools & Technologies": [
        "git",
        "github",
        "gitlab",
        "jira",
        "jupyter",
        "postman",
        "flask",
        "fastapi",
        "django",
        "streamlit",
    ],
    "Soft Skills": [
        "communication",
        "leadership",
        "teamwork",
        "collaboration",
        "problem solving",
        "critical thinking",
        "time management",
        "adaptability",
        "creativity",
        "decision making",
        "presentation",
        "negotiation",
        "attention to detail",
        "analytical thinking",
        "stakeholder management",
        "business communication",
    ],
    "Business & Domain": [
        "project management",
        "product management",
        "business analysis",
        "requirements gathering",
        "market research",
        "financial analysis",
        "sales",
        "marketing",
        "risk management",
        "process improvement",
        "agile",
        "scrum",
    ],
}


def extract_skills(text):
    text = normalize_text(text)
    result = {}

    for category, skills in SKILL_DATABASE.items():
        found = []

        for skill in skills:
            pattern = (
                r"(?<!\w)"
                + re.escape(skill)
                + r"(?!\w)"
            )

            if re.search(pattern, text):
                found.append(skill)

        if found:
            result[category] = sorted(found)

    return result


STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "have",
    "has",
    "was",
    "were",
    "are",
    "you",
    "your",
    "our",
    "their",
    "into",
    "using",
    "used",
    "been",
    "will",
    "can",
    "also",
    "about",
    "which",
    "while",
    "where",
    "what",
    "when",
    "how",
    "all",
    "any",
    "not",
    "but",
    "its",
    "they",
    "them",
    "his",
    "her",
    "she",
    "he",
}


def keywords(text, top_n=25):
    tokens = re.findall(
        r"[a-zA-Z0-9+#.\-]+",
        text.lower(),
    )

    useful = [
        token
        for token in tokens
        if token not in STOP_WORDS
        and len(token) > 2
    ]

    return Counter(useful).most_common(top_n)


def analyze_resume(text):
    return {
        "sections": detect_sections(text),
        "skills": extract_skills(text),
        "keywords": keywords(text),
        "tokens": re.findall(
            r"[a-zA-Z0-9+#.\-]+",
            text.lower(),
        ),
    }


# ============================================================
# CONTACT ANALYSIS
# ============================================================

def contact_analysis(text):
    return {
        "email": bool(
            re.search(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                text,
            )
        ),
        "phone": bool(
            re.search(
                r"(?<!\d)\+?\d[\d\s().-]{8,}\d(?!\d)",
                text,
            )
        ),
        "linkedin": "linkedin.com" in text.lower(),
        "github": "github.com" in text.lower(),
    }


# ============================================================
# ATS ANALYSIS
# ============================================================

def ats_analysis(text, nlp):
    sections = nlp["sections"]
    skills = nlp["skills"]
    key = nlp["keywords"]
    contact = contact_analysis(text)

    core = {
        "summary",
        "skills",
        "experience",
        "education",
    }

    optional = {
        "projects",
        "certifications",
        "achievements",
    }

    section_score = (
        len(core.intersection(sections)) / 4 * 15
        + len(optional.intersection(sections)) / 3 * 5
    )

    contact_score = sum(
        [
            3 if contact["email"] else 0,
            3 if contact["phone"] else 0,
            2 if contact["linkedin"] else 0,
            2 if contact["github"] else 0,
        ]
    )

    skill_count = sum(
        len(items)
        for items in skills.values()
    )

    if skill_count >= 15:
        skill_score = 15
    elif skill_count >= 10:
        skill_score = 13
    elif skill_count >= 7:
        skill_score = 11
    elif skill_count >= 5:
        skill_score = 8
    elif skill_count >= 3:
        skill_score = 5
    else:
        skill_score = 2

    word_count = len(text.split())

    if 250 <= word_count <= 900:
        length_score = 10
    elif 150 <= word_count <= 1200:
        length_score = 7
    else:
        length_score = 4

    keyword_score = min(len(key), 15)

    evidence_score = 0

    if "experience" in sections:
        evidence_score += 8

    if "projects" in sections:
        evidence_score += 5

    if "certifications" in sections:
        evidence_score += 2

    readability_score = 15

    if max(
        (len(line) for line in text.splitlines()),
        default=0,
    ) > 500:
        readability_score -= 3

    total_score = round(
        section_score
        + contact_score
        + skill_score
        + length_score
        + keyword_score
        + evidence_score
        + readability_score
    )

    total_score = min(
        max(total_score, 0),
        100,
    )

    recommendations = []

    if not contact["email"]:
        recommendations.append(
            "Add a professional email address."
        )

    if not contact["phone"]:
        recommendations.append(
            "Add a phone number."
        )

    if not contact["linkedin"]:
        recommendations.append(
            "Add LinkedIn if relevant."
        )

    for section, name in {
        "summary": "summary",
        "skills": "skills",
        "experience": "experience",
        "education": "education",
    }.items():
        if section not in sections:
            recommendations.append(
                f"Consider adding a {name} section."
            )

    if skill_count < 5:
        recommendations.append(
            "Add more relevant skills."
        )

    if word_count < 250:
        recommendations.append(
            "Add stronger evidence of your impact."
        )

    if not recommendations:
        recommendations.append(
            "No major ATS issues detected."
        )

    return {
        "total": total_score,
        "section_score": round(section_score, 1),
        "contact_score": contact_score,
        "skill_score": skill_score,
        "length_score": length_score,
        "keyword_score": keyword_score,
        "evidence_score": evidence_score,
        "readability_score": readability_score,
        "recommendations": recommendations,
    }


# ============================================================
# JOB DESCRIPTION MATCHING
# ============================================================

def tfidf_similarity(resume, jd):
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        matrix = vectorizer.fit_transform(
            [
                normalize_text(resume),
                normalize_text(jd),
            ]
        )

        return float(
            cosine_similarity(
                matrix[0:1],
                matrix[1:2],
            )[0][0]
        )

    except ValueError:
        return 0.0


def flatten_skills(skill_dict):
    return {
        skill.lower()
        for skills in skill_dict.values()
        for skill in skills
    }


@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def semantic_similarity(resume, jd):
    model = load_embedding_model()

    embeddings = model.encode(
        [resume, jd],
        normalize_embeddings=True,
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]],
    )[0][0]

    return float(similarity)


def semantic_requirement_match(resume, jd):
    model = load_embedding_model()

    resume_sentences = [
        sentence.strip()
        for sentence in re.split(r"[.\n]", resume)
        if len(sentence.strip()) > 20
    ]

    jd_sentences = [
        sentence.strip()
        for sentence in re.split(r"[.\n]", jd)
        if len(sentence.strip()) > 20
    ]

    if not resume_sentences or not jd_sentences:
        return []

    resume_vectors = model.encode(
        resume_sentences,
        normalize_embeddings=True,
    )

    jd_vectors = model.encode(
        jd_sentences,
        normalize_embeddings=True,
    )

    results = []

    for counter, jd_vector in enumerate(jd_vectors):
        scores = cosine_similarity(
            [jd_vector],
            resume_vectors,
        )[0]

        best_index = scores.argmax()
        best_score = scores[best_index]

        results.append(
            {
                "requirement": jd_sentences[counter],
                "resume_evidence": resume_sentences[best_index],
                "score": round(
                    float(best_score) * 100,
                    1,
                ),
            }
        )

    return sorted(
        results,
        key=lambda item: item["score"],
        reverse=True,
    )


def semantic_jd_analysis(resume, jd, resume_nlp):
    tfidf_score = (
        tfidf_similarity(resume, jd) * 100
    )

    semantic_score = (
        semantic_similarity(resume, jd) * 100
    )

    resume_skills = flatten_skills(
        resume_nlp["skills"]
    )

    jd_skills = flatten_skills(
        extract_skills(jd)
    )

    matched_skills = resume_skills.intersection(
        jd_skills
    )

    missing_skills = jd_skills - resume_skills

    if jd_skills:
        skill_score = (
            len(matched_skills)
            / len(jd_skills)
            * 100
        )
    else:
        skill_score = 0

    requirements = semantic_requirement_match(
        resume,
        jd,
    )

    if requirements:
        requirement_score = (
            sum(item["score"] for item in requirements)
            / len(requirements)
        )
    else:
        requirement_score = 0

    final_score = round(
        tfidf_score * 0.20
        + semantic_score * 0.35
        + skill_score * 0.25
        + requirement_score * 0.20
    )

    final_score = min(
        max(final_score, 0),
        100,
    )

    return {
        "match_score": final_score,
        "tfidf_score": round(tfidf_score, 1),
        "semantic_score": round(semantic_score, 1),
        "skill_score": round(skill_score, 1),
        "requirement_score": round(
            requirement_score,
            1,
        ),
        "matched_skills": sorted(matched_skills),
        "missing_skills": sorted(missing_skills),
        "requirements": requirements,
        "jd_skills": sorted(jd_skills),
    }


# ============================================================
# AI-STYLE FEEDBACK
# ============================================================

def generate_ai_feedback(resume, jd, nlp, jd_result):
    strengths = []
    weaknesses = []
    actions = []

    score = jd_result["match_score"]

    if jd_result["semantic_score"] >= 70:
        strengths.append(
            "The resume has strong semantic alignment with the job description."
        )

    if jd_result["skill_score"] >= 70:
        strengths.append(
            "A large proportion of the detected JD skills are present in the resume."
        )

    if "experience" in nlp["sections"]:
        strengths.append(
            "Relevant work-experience evidence is present."
        )

    if "projects" in nlp["sections"]:
        strengths.append(
            "Projects provide additional evidence of practical capability."
        )

    if jd_result["skill_score"] < 50:
        weaknesses.append(
            "The resume is missing several skills detected in the JD."
        )

    if jd_result["requirement_score"] < 60:
        weaknesses.append(
            "Some JD requirements do not have strong matching evidence."
        )

    if jd_result["semantic_score"] < 50:
        weaknesses.append(
            "The overall language and context are not strongly aligned with the role."
        )

    missing = jd_result["missing_skills"]

    if missing:
        actions.append(
            "If you genuinely possess them, explicitly mention: "
            + ", ".join(
                skill.title()
                for skill in missing[:8]
            )
        )

    actions.append(
        "Rewrite the professional summary around the target role."
    )

    actions.append(
        "Prioritize experience bullets that demonstrate the JD's most important requirements."
    )

    actions.append(
        "Use measurable outcomes instead of only describing responsibilities."
    )

    if score >= 80:
        overall = (
            "Excellent alignment. The resume is already "
            "strongly targeted toward this role."
        )
    elif score >= 65:
        overall = (
            "Strong alignment, but targeted improvements "
            "could make the resume more competitive."
        )
    elif score >= 50:
        overall = (
            "Moderate alignment. The resume should be "
            "tailored more closely to the job description."
        )
    else:
        overall = (
            "Low alignment. Significant tailoring is recommended "
            "before applying."
        )

    return {
        "overall": overall,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "actions": actions,
    }


# ============================================================
# HISTORY
# ============================================================

def save_analysis(filename, mode, score, result):
    """Save analysis results while supporting old database schemas."""
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO analyses
            (user_id, filename, mode, score, result_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                st.session_state.user_id,
                filename,
                mode,
                score,
                json.dumps(result),
                datetime.now(UTC).isoformat(),
            ),
        )
    except sqlite3.OperationalError as error:
        # Old databases may not have result_json.
        if "result_json" not in str(error).lower():
            connection.close()
            raise

        cursor.execute(
            """
            INSERT INTO analyses
            (user_id, filename, mode, score, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                st.session_state.user_id,
                filename,
                mode,
                score,
                datetime.now(UTC).isoformat(),
            ),
        )

    connection.commit()
    connection.close()


def get_history():
    """
    Load history using only columns required by the UI.

    result_json is deliberately excluded because older databases
    may not contain that column.
    """
    connection = get_connection()

    query = """
        SELECT
            id,
            filename,
            mode,
            score,
            created_at
        FROM analyses
        WHERE user_id = ?
        ORDER BY created_at DESC
    """

    dataframe = pd.read_sql_query(
        query,
        connection,
        params=(st.session_state.user_id,),
    )

    connection.close()
    return dataframe


def delete_history():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        "DELETE FROM analyses WHERE user_id = ?",
        (st.session_state.user_id,),
    )

    connection.commit()
    connection.close()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("📄 ResumeIQ")
    st.caption("AI Resume Intelligence")
    st.divider()

    st.write(
        f"👤 {st.session_state.username}"
    )

    st.divider()

    page = st.radio(
        "Workspace",
        [
            "🏠 Dashboard",
            "🔍 Analyze Resume",
            "🕘 History",
            "👤 Profile",
            "⚙️ Settings",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    if st.button(
        "🚪 Logout",
        width="stretch",
    ):
        logout()


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":
    st.caption("WORKSPACE")

    st.title(
        f"Welcome back, {st.session_state.username}."
    )

    st.write(
        "AI-powered ATS analysis, semantic job matching "
        "and actionable resume feedback."
    )

    st.divider()

    history = get_history()

    ats_history = history[
        history["mode"] == "ATS Analysis"
    ]

    jd_history = history[
        history["mode"] == "Job Description Analysis"
    ]

    latest_ats = (
        ats_history.iloc[0]["score"]
        if not ats_history.empty
        else None
    )

    latest_jd = (
        jd_history.iloc[0]["score"]
        if not jd_history.empty
        else None
    )

    current_skills = 0

    if st.session_state.nlp_result:
        current_skills = sum(
            len(items)
            for items in st.session_state.nlp_result[
                "skills"
            ].values()
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "🛡️ Latest ATS",
            f"{latest_ats:.0f}/100"
            if latest_ats is not None
            else "—",
        )

    with c2:
        st.metric(
            "🎯 Latest JD Match",
            f"{latest_jd:.0f}/100"
            if latest_jd is not None
            else "—",
        )

    with c3:
        st.metric(
            "🧠 Skills",
            current_skills,
        )

    with c4:
        st.metric(
            "📊 Analyses",
            len(history),
        )

    st.divider()
    st.subheader("ResumeIQ Intelligence")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 🧠 NLP Analysis")
        st.caption(
            "Analyze resume structure, keywords, "
            "skills, contact information and readability."
        )

    with c2:
        st.markdown("### 🎯 Semantic Matching")
        st.caption(
            "Compare meaning rather than relying only "
            "on exact keyword matches."
        )

    with c3:
        st.markdown("### 🔎 JD Analysis")
        st.caption(
            "Identify matched skills, missing skills "
            "and requirement-level evidence."
        )

    if not history.empty:
        st.divider()
        st.subheader("📈 Analysis Trend")
        
        chart_data = history[["created_at", "score"]].copy()

         # Convert every timestamp to UTC.
         # This handles both:
         # 2026-08-24 11:35:39
         # 2026-08-24T11:35:39.123456+00:00
        chart_data["created_at"] = pd.to_datetime(
            chart_data["created_at"],
            format="mixed",
            errors="coerce",
            utc=True,
        )

        # Remove invalid timestamps
        chart_data = chart_data.dropna(
        subset=["created_at"]
    )

    # Now every timestamp is timezone-aware UTC,
    # so sorting cannot produce a naive/aware error.
        chart_data = chart_data.sort_values(
        "created_at"
    )

        chart_data = chart_data.set_index(
        "created_at"
    )

        st.line_chart(
        chart_data["score"]
    )

# ============================================================
# ANALYZE RESUME
# ============================================================

elif page == "🔍 Analyze Resume":
    st.title("Analyze Resume")
    st.caption("Choose your analysis mode.")
    st.divider()

    mode = st.radio(
        "Analysis Mode",
        [
            "ATS Analysis",
            "Job Description Analysis",
        ],
        horizontal=True,
    )

    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf", "docx"],
    )

    job_description = ""

    if mode == "Job Description Analysis":
        st.subheader("🎯 Job Description")

        job_description = st.text_area(
            "Paste the complete JD",
            height=250,
            placeholder="Paste the complete job description...",
        )

    if uploaded_file:
        valid, message = validate_file(uploaded_file)

        if not valid:
            st.error(message)
        else:
            if (
                st.session_state.resume_filename
                != uploaded_file.name
            ):
                st.session_state.resume_filename = (
                    uploaded_file.name
                )
                st.session_state.resume_text = None
                st.session_state.nlp_result = None
                st.session_state.ats_result = None
                st.session_state.jd_result = None

            st.success(
                f"Ready: {uploaded_file.name}"
            )

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "📄 Type",
                    uploaded_file.name.split(".")[-1].upper(),
                )

            with c2:
                st.metric(
                    "📦 Size",
                    f"{uploaded_file.size / 1024:.1f} KB",
                )

            if st.button(
                "📄 Extract Resume",
                width="stretch",
            ):
                try:
                    raw_text = extract_resume_text(
                        uploaded_file
                    )

                    text = clean_text(raw_text)

                    if len(text.split()) < 30:
                        st.error(
                            "Very little text was extracted. "
                            "This may be a scanned PDF. OCR will "
                            "be added as a future extension."
                        )
                    else:
                        st.session_state.resume_text = text
                        st.session_state.nlp_result = None
                        st.session_state.ats_result = None
                        st.session_state.jd_result = None

                        st.success(
                            "Resume extracted successfully."
                        )

                except Exception as error:
                    st.error(
                        "Unable to process the resume."
                    )
                    st.caption(str(error))

    if st.session_state.resume_text:
        st.divider()
        st.subheader("📋 Extracted Resume")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Words",
                len(
                    st.session_state.resume_text.split()
                ),
            )

        with c2:
            st.metric(
                "Characters",
                len(st.session_state.resume_text),
            )

        with c3:
            st.metric(
                "File",
                st.session_state.resume_filename,
            )

        with st.expander("View extracted text"):
            st.text(
                st.session_state.resume_text
            )

        if st.button(
            "🧠 Run NLP Analysis",
            width="stretch",
        ):
            with st.spinner("Analyzing resume..."):
                st.session_state.nlp_result = (
                    analyze_resume(
                        st.session_state.resume_text
                    )
                )

            st.success(
                "NLP analysis completed."
            )

        if st.session_state.nlp_result:
            nlp = st.session_state.nlp_result
            skills = nlp["skills"]
            sections = nlp["sections"]

            total_skills = sum(
                len(items)
                for items in skills.values()
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "📑 Sections",
                    len(sections),
                )

            with c2:
                st.metric(
                    "🛠️ Skills",
                    total_skills,
                )

            with c3:
                st.metric(
                    "🔑 Keywords",
                    len(nlp["keywords"]),
                )

            st.subheader("🛠️ Detected Skills")

            for category, skill_list in skills.items():
                with st.expander(
                    f"{category} · {len(skill_list)}"
                ):
                    st.write(
                        ", ".join(
                            skill.title()
                            for skill in skill_list
                        )
                    )

            st.subheader("📑 Resume Sections")

            for section, section_content in sections.items():
                with st.expander(section.title()):
                    st.write(section_content)

            if mode == "ATS Analysis":
                st.divider()

                if st.button(
                    "🛡️ Calculate ATS Score",
                    width="stretch",
                ):
                    with st.spinner(
                        "Calculating ATS score..."
                    ):
                        result = ats_analysis(
                            st.session_state.resume_text,
                            nlp,
                        )

                    st.session_state.ats_result = result

                    save_analysis(
                        st.session_state.resume_filename,
                        "ATS Analysis",
                        result["total"],
                        result,
                    )

                    st.success(
                        "ATS analysis completed."
                    )

                if st.session_state.ats_result:
                    result = st.session_state.ats_result

                    st.subheader("🛡️ ATS Score")

                    score = result["total"]

                    c1, c2 = st.columns([1, 2])

                    with c1:
                        st.metric(
                            "ATS Score",
                            f"{score}/100",
                        )

                    with c2:
                        st.progress(score / 100)

                    st.subheader("Score Breakdown")

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.metric(
                            "Sections",
                            f"{result['section_score']}/20",
                        )
                        st.metric(
                            "Contact",
                            f"{result['contact_score']}/10",
                        )
                        st.metric(
                            "Skills",
                            f"{result['skill_score']}/15",
                        )

                    with c2:
                        st.metric(
                            "Length",
                            f"{result['length_score']}/10",
                        )
                        st.metric(
                            "Keywords",
                            f"{result['keyword_score']}/15",
                        )

                    with c3:
                        st.metric(
                            "Evidence",
                            f"{result['evidence_score']}/15",
                        )
                        st.metric(
                            "Readability",
                            f"{result['readability_score']}/15",
                        )

                    st.subheader("💡 Recommendations")

                    for item in result["recommendations"]:
                        st.write(f"• {item}")

            else:
                st.divider()

                if not job_description.strip():
                    st.info(
                        "Paste a job description above."
                    )
                elif len(job_description.split()) < 20:
                    st.warning(
                        "Paste a more complete job description."
                    )
                else:
                    if st.button(
                        "🎯 Run AI Job Matching",
                        width="stretch",
                    ):
                        with st.spinner(
                            "Running semantic AI matching..."
                        ):
                            result = semantic_jd_analysis(
                                st.session_state.resume_text,
                                job_description,
                                nlp,
                            )

                            feedback = generate_ai_feedback(
                                st.session_state.resume_text,
                                job_description,
                                nlp,
                                result,
                            )

                            result["feedback"] = feedback

                            st.session_state.jd_result = result

                            save_analysis(
                                st.session_state.resume_filename,
                                "Job Description Analysis",
                                result["match_score"],
                                result,
                            )

                        st.success(
                            "AI job matching completed."
                        )

                    if st.session_state.jd_result:
                        result = st.session_state.jd_result

                        st.subheader("🎯 AI Job Match")

                        score = result["match_score"]

                        c1, c2 = st.columns([1, 2])

                        with c1:
                            st.metric(
                                "JD Match",
                                f"{score}/100",
                            )

                        with c2:
                            st.progress(score / 100)

                        st.subheader("📊 Matching Signals")

                        c1, c2, c3, c4 = st.columns(4)

                        with c1:
                            st.metric(
                                "TF-IDF",
                                f"{result['tfidf_score']}%",
                            )

                        with c2:
                            st.metric(
                                "Semantic AI",
                                f"{result['semantic_score']}%",
                            )

                        with c3:
                            st.metric(
                                "Skills",
                                f"{result['skill_score']}%",
                            )

                        with c4:
                            st.metric(
                                "Requirements",
                                f"{result['requirement_score']}%",
                            )

                        st.subheader("✅ Matched Skills")

                        if result["matched_skills"]:
                            st.write(
                                ", ".join(
                                    skill.title()
                                    for skill in result["matched_skills"]
                                )
                            )
                        else:
                            st.warning(
                                "No matching skills detected."
                            )

                        st.subheader("⚠️ Missing Skills")

                        if result["missing_skills"]:
                            st.write(
                                ", ".join(
                                    skill.title()
                                    for skill in result["missing_skills"]
                                )
                            )
                        else:
                            st.success(
                                "No detected JD skills are missing."
                            )

                        st.subheader(
                            "🔎 Requirement Evidence"
                        )

                        for item in result["requirements"][:10]:
                            with st.expander(
                                f"{item['score']}% — "
                                f"{item['requirement'][:100]}"
                            ):
                                st.write(
                                    "**Resume evidence:**"
                                )
                                st.write(
                                    item["resume_evidence"]
                                )

                        feedback = result["feedback"]

                        st.divider()
                        st.subheader(
                            "🤖 AI Resume Feedback"
                        )

                        st.info(feedback["overall"])

                        c1, c2 = st.columns(2)

                        with c1:
                            st.markdown("### Strengths")

                            for item in feedback["strengths"]:
                                st.write(f"• {item}")

                            st.markdown("### Weaknesses")

                            for item in feedback["weaknesses"]:
                                st.write(f"• {item}")

                        with c2:
                            st.markdown("### Recommended Actions")

                            for item in feedback["actions"]:
                                st.write(f"• {item}")


# ============================================================
# HISTORY
# ============================================================

elif page == "🕘 History":
    st.title("Analysis History")
    st.caption(
        "Review your previous ResumeIQ analyses."
    )

    history = get_history()

    if history.empty:
        st.info("No analyses yet.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Total Analyses",
                len(history),
            )

        with c2:
            st.metric(
                "Average Score",
                f"{history['score'].mean():.1f}/100",
            )

        st.divider()

        st.dataframe(
            history[
                [
                    "filename",
                    "mode",
                    "score",
                    "created_at",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

        st.divider()
        st.subheader("📈 Score History")

        chart = history.copy()

        chart["created_at"] = pd.to_datetime(
            chart["created_at"],
            format="mixed",
            errors="coerce",
            utc=True,
        )

        # Remove invalid timestamps
        chart = chart.dropna(
        subset=["created_at"])

        chart = chart.sort_values("created_at")

        chart = chart.set_index("created_at")                                                                                                        

        st.line_chart(chart["score"])

        st.divider()

        if st.button(
            "🗑️ Delete My Analysis History",
            type="secondary",
        ):
            delete_history()
            st.success(
                "Analysis history deleted."
            )
            st.rerun()


# ============================================================
# PROFILE
# ============================================================

elif page == "👤 Profile":
    st.title("Profile")
    st.caption("Your ResumeIQ account.")
    st.divider()

    st.write(
        f"**Username:** {st.session_state.username}"
    )

    st.write(
        f"**Email:** {st.session_state.email}"
    )

    st.divider()

    st.info(
        "Resume text is processed in memory. "
        "The database stores analysis metadata and results, "
        "not the uploaded resume document itself."
    )


# ============================================================
# SETTINGS
# ============================================================

elif page == "⚙️ Settings":
    st.title("Settings")
    st.caption("ResumeIQ configuration.")
    st.divider()

    st.subheader("🤖 AI Engine")

    st.write("Semantic model:")

    st.code(EMBEDDING_MODEL)

    st.caption(
        "The model converts resume and job-description text "
        "into numerical embeddings so their semantic similarity "
        "can be calculated."
    )

    st.divider()

    st.subheader("🔐 Security")

    st.success(
        "Passwords are salted and hashed."
    )

    st.success(
        "Login sessions use random server-side tokens."
    )

    st.success(
        "Resume files are not stored in SQLite."
    )

    st.success(
        "Uploaded file type and size are validated."
    )

    st.info(
        "☁️ Streamlit Cloud deployment: SQLite is suitable for this portfolio/demo version. "
        "For a future multi-user production release, migrate persistent data to PostgreSQL or Supabase "
        "and keep secrets in Streamlit Secrets."
    )

    st.divider()

    if st.button(
        "🚪 Sign Out",
        width="stretch",
    ):
        logout()

    st.divider()

    st.caption(
        "ResumeIQ · AI Resume Intelligence"
    )
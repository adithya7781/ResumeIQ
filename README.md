# ResumeIQ – AI Resume Intelligence

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![NLP](https://img.shields.io/badge/NLP-Enabled-6C63FF)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![AI](https://img.shields.io/badge/AI-Semantic%20Matching-8B5CF6)](https://www.sbert.net/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen)]()

> AI-powered resume intelligence platform that analyzes resumes using NLP, ATS rules, keyword analysis, semantic similarity, skill matching, and job-description intelligence.

---

## Project Overview

ResumeIQ is an AI-powered resume analysis application built with Python and Streamlit.

The application allows users to upload a resume and analyze it in two different modes:

### Mode 1 – ATS Analysis

Analyze the resume independently to identify:

- ATS compatibility
- Resume sections
- Technical skills
- Soft skills
- Keywords
- Contact information
- Resume length
- Experience evidence
- Readability
- Improvement recommendations

### Mode 2 – Job Description Analysis

Compare a resume against a specific job description to identify:

- Overall job-match score
- Semantic similarity
- TF-IDF similarity
- Matching skills
- Missing skills
- Requirement-level evidence
- Skill coverage
- Resume strengths
- Resume weaknesses
- Recommended improvements

The goal is to provide a practical resume intelligence system rather than relying on a single black-box score.

---

# Features

## 🔐 Authentication

ResumeIQ provides user account functionality with:

- Sign up
- Sign in
- Secure password hashing
- Persistent login sessions
- Session expiration
- Logout
- SQLite-backed user management

Passwords are never stored as plain text.

---

## 📄 Resume Upload

Supported formats:

- PDF
- DOCX

The application validates:

- File type
- File size
- Extracted text availability

Maximum upload size:

```text
5 MB

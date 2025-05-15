# 🧠 CodeAuditor: BERT-Powered Code Auditor

[![License](https://img.shields.io/github/license/karthyick/CodeAuditor)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange.svg)](https://streamlit.io/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> **Analyze emotional signals in your codebase and commits to boost engineering health, morale, and productivity!**

---

## 🚀 Overview

**CodeAuditor** is an open-source AI tool that brings developer wellbeing and engineering culture to the forefront.  
It analyzes code comments and commit messages using a BERT-based NLP model to detect emotions like joy, frustration, pride, urgency, and more — right from your own repository.

Explore emotion trends, burnout signals, hotspots, and developer profiles via an interactive Streamlit dashboard — all locally, securely, and across multiple languages.

---

## ✨ Features

- **Multi-language support:** Python, JavaScript, TypeScript, Java, C#, C++, Ruby, HTML, CSS, PHP, Swift, Kotlin, and more.
- **Code comment & commit analysis:** Extracts and classifies emotions from both code comments and git commit messages.
- **Powered by BERT:** Uses HuggingFace Transformers for robust emotion classification (fine-tuned or vanilla BERT).
- **Rich analytics dashboard:** Streamlit app with:
  - Emotion distribution & trends
  - Burnout and technical debt signals
  - Hotspots in files
  - Developer emotional profiles
  - Export to CSV/JSON
- **Private by design:** All analysis is local; no data leaves your system.
- **Extensible:** Modular architecture for adding new analyzers, models, or languages.

---

## 🏗️ Project Structure

```plaintext
CodeAuditor/
└── bert-code-emotion-auditor/
    ├── app/
    │   └── dashboard.py            # Streamlit dashboard UI
    ├── data/                       # Example data or outputs
    ├── extract/
    │   ├── comment_extractor.py    # Code comment extraction logic
    │   └── commit_extractor.py     # Git commit metadata extractor
    ├── model/
    │   └── emotion_classifier.py   # BERT emotion classifier
    ├── utils/
    │   └── preprocess.py           # Comment cleaning & utility functions
    ├── requirements.txt            # Dependencies
```

## ⚡️ Quickstart

1. **Clone the Repo**

    ```bash
    git clone https://github.com/karthyick/CodeAuditor.git
    cd CodeAuditor/bert-code-emotion-auditor
    ```

2. **Install Requirements**

    Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    # On Linux/Mac:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Run the Streamlit Dashboard**

    ```bash
    streamlit run app/dashboard.py
    ```

4. **Analyze Your Repository**

    - Set your repo path in the sidebar (absolute or relative path).
    - Select "Code Comments", "Commit Messages", "Combined Analysis", or "Developer Profiles".
    - Click **Scan/Analyze** — explore the charts, trends, and profiles!

---

## 🔬 Emotion Categories

The default classifier detects the following emotions in code and commit messages:

- `neutral`
- `joy`
- `frustration`
- `anger`
- `pride`
- `exhaustion`
- `confusion`
- `urgency`
- `concern`

> _Custom emotion classes possible if you fine-tune your own model!_

---

## 🏆 Key Modules

- **`extract/comment_extractor.py`**: Extracts comments using regex (and optionally tree-sitter for advanced parsing).
- **`extract/commit_extractor.py`**: Parses git commit logs, metadata, and supports conventional commits.
- **`model/emotion_classifier.py`**: Loads BERT (pre-trained or fine-tuned) and classifies emotions, with batch and fallback rule-based options.
- **`utils/preprocess.py`**: Cleans, normalizes, and analyzes comments for variable naming sentiment.

---

## 🛠️ Customization

- **Use your own BERT model:** Place your fine-tuned model path in the UI or edit `model/emotion_classifier.py`.
- **Add more languages:** Extend the supported file extensions in `comment_extractor.py`.
- **Modify emotion rules:** Edit or expand in `emotion_classifier.py` for rule-based fallback.

---

## 🖇️ Export & Reporting

- Download results as CSV or JSON from the dashboard.
- _Coming soon: PDF/HTML reporting!_

---

## 🤝 Contributing

Pull requests and issues are welcome!

- **Fork this repo**
- **Create a branch** (`feature/my-feature`)
- **Submit a PR**

_For major changes, please open an issue first to discuss your ideas._

---

## 📚 References

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [GitPython](https://gitpython.readthedocs.io/)
- [Python logging](https://docs.python.org/3/library/logging.html)
- [tree-sitter](https://tree-sitter.github.io/tree-sitter/)

---

## 🔗 Repo

**GitHub:** [https://github.com/karthyick/CodeAuditor.git](https://github.com/karthyick/CodeAuditor.git)


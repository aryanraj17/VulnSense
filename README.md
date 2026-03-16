# VulnSense — AI-Powered Code Vulnerability Detector

An intelligent vulnerability detection system combining CodeBERT, 
Graph Neural Networks, YARA rules, and LLM-powered auto-fix suggestions.

## Features
- CodeBERT-based vulnerability classification (multi-class CWE)
- AST + Code Property Graph analysis using GNN
- SHAP explainability with line-level highlighting
- YARA rule-based pattern matching
- CVSS-style severity scoring
- LLM auto-fix patch suggestions
- Active learning feedback loop

## Tech Stack
Python | PyTorch | HuggingFace Transformers | tree-sitter | 
PyTorch Geometric | SHAP | YARA | Streamlit | OpenAI/Anthropic API

## Setup
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage
```bash
streamlit run app.py
```
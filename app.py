"""
VulnSense Streamlit Frontend
----------------------------
Single-file Streamlit app with 3 pages:
1) Scanner
2) Dashboard
3) About
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

# project-root imports
sys.path.insert(0, '.')

from core.ensemble import EnsembleScanner
from core.severity_scorer import SeverityScorer
from core.autofix import AutoFixer
from core.active_learning import FeedbackStore
from core.utils import (
    detect_language,
    get_language_display,
)

# ── Paths ────────────────────────────────────────────────────────────────────
SCAN_HISTORY_PATH = 'data/scan_history.json'
METRICS_BINARY = 'models/codebert/binary_metrics.json'
METRICS_GNN = 'models/gnn/gnn_metrics.json'

# ── Streamlit Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title='VulnSense',
    page_icon='🔍',
    layout='wide'
)


# ── Cached services ──────────────────────────────────────────────────────────
@st.cache_resource
def load_scanner():
    return EnsembleScanner()


@st.cache_resource
def load_scorer():
    return SeverityScorer()


@st.cache_resource
def load_fixer():
    return AutoFixer()


@st.cache_resource
def load_feedback_store():
    return FeedbackStore()


# ── Utilities ────────────────────────────────────────────────────────────────
def ensure_session_state():
    defaults = {
        'last_result': None,
        'last_severity': None,
        'last_code': '',
        'last_lang': 'c',
        'last_filename': None,
        'scan_history': [],
        'fix_result': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def append_scan_history(result: dict, severity: dict, lang: str):
    history = load_json(SCAN_HISTORY_PATH, [])
    history.append({
        'timestamp': datetime.utcnow().isoformat(),
        'verdict': 'VULNERABLE' if result.get('is_vulnerable') else 'SAFE',
        'severity': severity.get('severity', 'SAFE'),
        'cvss_score': severity.get('cvss_score', 0.0),
        'cwe': severity.get('cwe', 'Unknown'),
        'language': lang,
        'yara_matches': len(result.get('yara_matches', [])),
        'dangerous_lines': result.get('dangerous_lines', []),
        'final_score': result.get('final_score', 0.0),
    })
    save_json(SCAN_HISTORY_PATH, history)


def language_from_selector(selection: str):
    if selection == 'Auto-detect':
        return None
    if selection in ('C', 'C++'):
        return 'c'
    if selection == 'Python':
        return 'python'
    return None


def render_verdict_banner(severity: dict):
    sev = severity.get('severity', 'SAFE')
    cvss = severity.get('cvss_score', 0)
    cwe = severity.get('cwe', 'Unknown')
    label = f"{severity.get('risk_label', '⚪ UNKNOWN')} — CVSS {cvss}/10 | {cwe}"

    if sev == 'CRITICAL':
        st.error(label)
    elif sev in ('HIGH', 'MEDIUM'):
        st.warning(label)
    elif sev == 'LOW':
        st.info(label)
    else:
        st.success(label)


def render_score_breakdown(result: dict, severity: dict):
    st.subheader('Scanner Breakdown')
    scores = result.get('individual_scores', {})
    for key in ('codebert', 'gnn', 'yara', 'ast'):
        v = float(scores.get(key, 0.0))
        st.write(f"**{key.upper()}**: `{v:.3f}`")
        st.progress(min(max(v, 0.0), 1.0))

    st.write(f"**Final Score**: `{result.get('final_score', 0):.3f}`")
    st.write(f"**Verdict**: `{'VULNERABLE' if result.get('is_vulnerable') else 'SAFE'}`")
    st.caption(severity.get('summary', ''))


def render_dangerous_lines(result: dict, code: str):
    st.subheader('Code Analysis (Dangerous Lines)')
    lines = result.get('dangerous_lines', [])
    if not lines:
        st.info('No dangerous lines identified by AST parser.')
        return

    code_lines = code.splitlines()
    for ln in lines:
        text = code_lines[ln - 1] if 1 <= ln <= len(code_lines) else ''
        st.write(f"- **Line {ln}**: `{text.strip()[:120]}`")


def render_yara_matches(result: dict):
    st.subheader('YARA Matches')
    matches = result.get('yara_matches', [])
    if not matches:
        st.info('No YARA rules matched.')
        return

    for i, m in enumerate(matches, 1):
        st.write(f"**[{i}] {m.get('rule', 'Unknown Rule')}** | {m.get('severity', 'LOW')} | {m.get('cwe', 'Unknown')}")
        if m.get('description'):
            st.caption(m['description'])


def render_remediation(severity: dict):
    st.subheader('Remediation Steps')
    rem = severity.get('remediation', [])
    if not rem:
        st.info('No remediation suggestions available.')
        return
    for i, step in enumerate(rem, 1):
        st.write(f"{i}. {step}")


def build_report(code: str, result: dict, severity: dict, fix_result: dict | None):
    timestamp = datetime.utcnow().isoformat()
    report = []
    report.append(f"# VulnSense Scan Report\n")
    report.append(f"Timestamp: {timestamp}\n")
    report.append(f"Verdict: {'VULNERABLE' if result.get('is_vulnerable') else 'SAFE'}\n")
    report.append(f"Severity: {severity.get('severity', 'SAFE')}\n")
    report.append(f"CVSS: {severity.get('cvss_score', 0)}\n")
    report.append(f"CWE: {severity.get('cwe', 'Unknown')}\n")
    report.append(f"Final Score: {result.get('final_score', 0)}\n")
    report.append(f"Dangerous Lines: {result.get('dangerous_lines', [])}\n")

    report.append('\n## YARA Matches\n')
    for m in result.get('yara_matches', []):
        report.append(f"- {m.get('rule')} | {m.get('severity')} | {m.get('cwe')}\n")

    report.append('\n## Remediation\n')
    for r in severity.get('remediation', []):
        report.append(f"- {r}\n")

    report.append('\n## Submitted Code (Truncated to 200 lines)\n')
    code_lines = code.splitlines()[:200]
    report.append('```\n' + '\n'.join(code_lines) + '\n```\n')

    if fix_result:
        report.append('\n## AI Suggested Fix\n')
        report.append(f"Provider: {fix_result.get('provider', 'unknown')}\n")
        report.append('```\n' + fix_result.get('fixed_code', '')[:6000] + '\n```\n')

    return ''.join(report)


def scanner_page():
    st.title('🔍 VulnSense Scanner')
    st.caption('AI-powered vulnerability scanning for code snippets and files')

    col_l, col_r = st.columns([2, 1])
    with col_r:
        language_choice = st.selectbox('Language', ['Auto-detect', 'C', 'C++', 'Python'])
        run_scan = st.button('Run Scan', type='primary', use_container_width=True)

    with col_l:
        tab_paste, tab_upload = st.tabs(['Paste Code', 'Upload File'])
        input_code = ''
        filename = None

        with tab_paste:
            input_code = st.text_area('Paste your code here', height=320, key='code_text')

        with tab_upload:
            up = st.file_uploader('Upload code file', type=['c', 'cpp', 'h', 'py', 'js'])
            if up is not None:
                filename = up.name
                input_code = up.read().decode('utf-8', errors='ignore')
                st.code(input_code[:6000], language='c')

    if run_scan:
        if not input_code.strip():
            st.warning('Please paste code or upload a file before scanning.')
            return

        selected_lang = language_from_selector(language_choice)
        lang = selected_lang or detect_language(input_code, filename)

        scanner = load_scanner()
        scorer = load_scorer()

        with st.spinner('Scanning code...'):
            result = scanner.scan(input_code, lang)
            severity = scorer.score(result)

        st.session_state['last_result'] = result
        st.session_state['last_severity'] = severity
        st.session_state['last_code'] = input_code
        st.session_state['last_lang'] = lang
        st.session_state['last_filename'] = filename
        st.session_state['fix_result'] = None

        append_scan_history(result, severity, lang)

    if st.session_state['last_result']:
        result = st.session_state['last_result']
        severity = st.session_state['last_severity']
        input_code = st.session_state['last_code']

        st.divider()
        render_verdict_banner(severity)
        render_score_breakdown(result, severity)
        render_dangerous_lines(result, input_code)
        render_yara_matches(result)
        render_remediation(severity)

        st.subheader('AI Suggested Fix')
        if st.button('Generate AI Fix'):
            fixer = load_fixer()
            with st.spinner('Generating fix suggestion...'):
                fix = fixer.get_fix(
                    code=input_code,
                    cwe=severity.get('cwe', 'CWE-Other'),
                    dangerous_lines=result.get('dangerous_lines', []),
                    yara_matches=result.get('yara_matches', []),
                )
            st.session_state['fix_result'] = fix

        if st.session_state.get('fix_result'):
            fix = st.session_state['fix_result']
            st.caption(f"Provider: {fix.get('provider', 'unknown')} | Success: {fix.get('success')}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('**Original**')
                st.code(fix.get('original_code', ''), language='c')
            with c2:
                st.markdown('**Suggested Fix**')
                st.code(fix.get('fixed_code', ''), language='c')

        st.subheader('Feedback')
        fb_col1, fb_col2 = st.columns(2)

        with fb_col1:
            if st.button('👍 Correct'):
                store = load_feedback_store()
                store.add_feedback(
                    code=input_code,
                    predicted_label=1 if result.get('is_vulnerable') else 0,
                    correct_label=1 if result.get('is_vulnerable') else 0,
                    confidence=float(result.get('final_score', 0.0)),
                    cwe=severity.get('cwe', 'Unknown')
                )
                st.success('Feedback saved as confirmed.')

        with fb_col2:
            with st.expander('👎 Wrong — mark feedback'):
                issue = st.radio(
                    'Select issue type',
                    ['Safe code flagged as vulnerable', 'Vulnerable code missed'],
                    key='fb_issue'
                )
                if st.button('Submit Feedback'):
                    store = load_feedback_store()
                    if issue == 'Safe code flagged as vulnerable':
                        predicted, correct = 1, 0
                    else:
                        predicted, correct = 0, 1

                    store.add_feedback(
                        code=input_code,
                        predicted_label=predicted,
                        correct_label=correct,
                        confidence=float(result.get('final_score', 0.0)),
                        cwe=severity.get('cwe', 'Unknown')
                    )
                    st.success('Successfully submitted Feedback.')
                    st.rerun()

        report_text = build_report(
            code=input_code,
            result=result,
            severity=severity,
            fix_result=st.session_state.get('fix_result')
        )
        st.download_button(
            label='📥 Download Report',
            data=report_text,
            file_name=f"vulnsense_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md",
            mime='text/markdown'
        )


def dashboard_page():
    st.title('📊 Dashboard')

    history = load_json(SCAN_HISTORY_PATH, [])
    feedback_store = load_feedback_store()
    feedback_stats = feedback_store.get_stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Scans', len(history))
    c2.metric('Feedback Items', feedback_stats.get('total', 0))

    vuln_count = sum(1 for x in history if x.get('verdict') == 'VULNERABLE')
    c3.metric('Vulnerable Findings', vuln_count)

    avg_cvss = round(sum(x.get('cvss_score', 0) for x in history) / len(history), 2) if history else 0.0
    c4.metric('Average CVSS', avg_cvss)

    if not history:
        st.info('No scan history yet. Run scans on the Scanner page.')
        return

    df = pd.DataFrame(history)

    st.subheader('Most Common CWEs')
    cwe_counts = df['cwe'].value_counts().reset_index()
    cwe_counts.columns = ['cwe', 'count']
    st.bar_chart(cwe_counts.set_index('cwe'))

    st.subheader('Severity Distribution')
    sev_counts = df['severity'].value_counts().reset_index()
    sev_counts.columns = ['severity', 'count']
    fig = px.pie(sev_counts, values='count', names='severity', title='Severity Distribution')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Recent Scans')
    recent = df.sort_values('timestamp', ascending=False).head(20)
    st.dataframe(recent, use_container_width=True)


def about_page():
    st.title('ℹ️ About VulnSense')
    st.markdown(
        """
VulnSense is an AI-powered vulnerability detection platform that combines:
- GraphCodeBERT-style transformer classification
- AST + Graph Neural Network structural analysis
- YARA signature matching
- CVSS-style severity scoring
- LLM-assisted remediation suggestions
- Active learning feedback loop
"""
    )

    st.subheader('Tech Stack')
    st.write('Python, PyTorch, Transformers, PyTorch Geometric, tree-sitter, YARA, SHAP, Streamlit')

    st.subheader('Model Metrics (Artifacts)')
    bin_metrics = load_json(METRICS_BINARY, {})
    gnn_metrics = load_json(METRICS_GNN, {})

    m1, m2 = st.columns(2)
    with m1:
        st.markdown('**Binary Classifier**')
        if bin_metrics:
            st.json(bin_metrics)
        else:
            st.info('binary_metrics.json not found')
    with m2:
        st.markdown('**GNN Model**')
        if gnn_metrics:
            st.json(gnn_metrics)
        else:
            st.info('gnn_metrics.json not found')

    st.subheader('Project')
    st.write('Use this frontend for demo and rapid security triage workflows.')


def main():
    ensure_session_state()

    with st.sidebar:
        st.title('VulnSense')
        st.caption('Security code scanner')
        page = st.radio('Navigate', ['Scanner', 'Dashboard', 'About'])

    if page == 'Scanner':
        scanner_page()
    elif page == 'Dashboard':
        dashboard_page()
    else:
        about_page()


if __name__ == '__main__':
    main()

"""
utils.py
--------
Utility functions for VulnSense.
Includes language detection and file handling.
"""

import os
import re

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit

def validate_file_size(path: str):
    if os.path.getsize(path) > MAX_FILE_SIZE:
        raise ValueError("File too large. Maximum size is 10MB.")

# ── Supported languages ───────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    'c'     : 'C/C++',
    'python': 'Python',
}

PARTIAL_SUPPORT = {
    'javascript': 'JavaScript (YARA only)',
    'java'      : 'Java (YARA only)',
}

# ── File extension mapping ────────────────────────────────────────────────────
EXTENSION_MAP = {
    '.c'   : 'c',
    '.cpp' : 'c',
    '.cc'  : 'c',
    '.h'   : 'c',
    '.hpp' : 'c',
    '.py'  : 'python',
    '.js'  : 'javascript',
    '.java': 'java',
}

# ── Code pattern signatures ───────────────────────────────────────────────────
LANGUAGE_PATTERNS = {
    'c': [
        r'#include\s*[<"]',
        r'int\s+main\s*\(',
        r'void\s+\w+\s*\(',
        r'printf\s*\(',
        r'malloc\s*\(',
        r'char\s+\w+\[',
        r'struct\s+\w+',
        r'->\w+',
    ],
    'python': [
        r'^def\s+\w+\s*\(',
        r'^import\s+\w+',
        r'^from\s+\w+\s+import',
        r'^\s*elif\s+',
        r'print\s*\(',
        r':\s*$',
        r'__init__',
        r'self\.',
    ],
    'javascript': [
        r'function\s+\w+\s*\(',
        r'const\s+\w+\s*=',
        r'let\s+\w+\s*=',
        r'var\s+\w+\s*=',
        r'=>\s*{',
        r'console\.log',
        r'document\.',
    ],
    'java': [
        r'public\s+class\s+\w+',
        r'public\s+static\s+void\s+main',
        r'System\.out\.println',
        r'import\s+java\.',
        r'@Override',
        r'extends\s+\w+',
    ],
}


# ────────────────────────────────────────────────────────────────────────────
def detect_language_from_extension(filename: str) -> str:
    """
    Detects language from file extension.

    Args:
        filename: file name or path

    Returns: language string or 'unknown'
    """
    if not filename:
        return 'unknown'

    ext = os.path.splitext(filename.lower())[1]
    return EXTENSION_MAP.get(ext, 'unknown')


def detect_language_from_code(code: str) -> str:
    """
    Detects language by matching code patterns.
    Used when no filename is available.

    Args:
        code: source code string

    Returns: language string or 'c' as default
    """
    if not code or len(code.strip()) == 0:
        return 'c'

    scores = {lang: 0 for lang in LANGUAGE_PATTERNS}

    for lang, patterns in LANGUAGE_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, code, re.MULTILINE)
            scores[lang] += len(matches)

    # return language with highest score
    best_lang = max(scores, key=scores.get)

    # if no patterns matched default to C
    if scores[best_lang] == 0:
        return 'c'

    return best_lang


def detect_language(code: str, filename: str = None) -> str:
    """
    Main language detection function.
    Tries extension first then falls back to pattern matching.

    Args:
        code    : source code string
        filename: optional filename for extension-based detection

    Returns: language string ('c', 'python', 'javascript', 'java')
    """
    # try extension first — most reliable
    if filename:
        lang = detect_language_from_extension(filename)
        if lang != 'unknown':
            return lang

    # fall back to pattern matching
    return detect_language_from_code(code)


def get_language_display(lang: str) -> str:
    """Returns human readable language name."""
    all_langs = {**SUPPORTED_LANGUAGES, **PARTIAL_SUPPORT}
    return all_langs.get(lang, lang.upper())


def is_supported(lang: str) -> bool:
    """Returns True if language is fully supported."""
    return lang in SUPPORTED_LANGUAGES


def read_uploaded_file(file_path: str) -> tuple:
    """
    Reads an uploaded code file.

    Returns: (code_string, filename, language)
    """
    try:
        filename = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        lang = detect_language(code, filename)
        return code, filename, lang
    except Exception as e:
        return None, None, None


def truncate_code(code: str, max_lines: int = 100) -> str:
    """
    Truncates code to max_lines for display purposes.
    Adds a note if truncated.
    """
    lines = code.split('\n')
    if len(lines) <= max_lines:
        return code

    truncated = '\n'.join(lines[:max_lines])
    truncated += f'\n\n// ... ({len(lines) - max_lines} more lines truncated)'
    return truncated


def format_file_size(size_bytes: int) -> str:
    """Returns human readable file size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024*1024):.1f} MB"


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Testing Language Detector...")
    print("=" * 55)

    test_cases = [
        {
            'code': '#include <stdio.h>\nvoid func(char *s) { char buf[64]; strcpy(buf, s); }',
            'filename': 'test.c',
            'expected': 'c'
        },
        {
            'code': 'def hello():\n    print("hello")\n    return True',
            'filename': 'test.py',
            'expected': 'python'
        },
        {
            'code': '#include <string.h>\nint main() { return 0; }',
            'filename': None,
            'expected': 'c'
        },
        {
            'code': 'import os\nfrom pathlib import Path\ndef main():\n    pass',
            'filename': None,
            'expected': 'python'
        },
    ]

    all_passed = True
    for i, case in enumerate(test_cases):
        detected = detect_language(case['code'], case['filename'])
        passed   = detected == case['expected']
        status   = '✅' if passed else '❌'

        if not passed:
            all_passed = False

        print(f"  {status} Test {i+1}: detected={detected} "
              f"expected={case['expected']} "
              f"file={case['filename']}")

    print(f"\n  {'All tests passed!' if all_passed else 'Some tests failed'}")

    print("\nLanguage display names:")
    for lang in ['c', 'python', 'javascript', 'java', 'unknown']:
        print(f"  {lang:<12} → {get_language_display(lang):<30} "
              f"supported={is_supported(lang)}")
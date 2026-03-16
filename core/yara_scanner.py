"""
yara_scanner.py
---------------
Scans code using custom YARA rules.
Returns matched rules with CWE, severity, and description.
"""

import os
import yara
import tempfile
from pathlib import Path


SEVERITY_MAP = {
    'CRITICAL': 4,
    'HIGH'    : 3,
    'MEDIUM'  : 2,
    'LOW'     : 1,
    'INFO'    : 0,
}


def compile_rules(rules_dir: str = 'rules/custom') -> yara.Rules:
    rule_files = {}
    for path in Path(rules_dir).rglob('*.yar'):
        rule_files[path.stem] = str(path)

    if not rule_files:
        print(f"  Warning: No .yar files found in {rules_dir}")
        return None

    print(f"  Compiled {len(rule_files)} YARA rule files")
    return yara.compile(filepaths=rule_files)


def scan_code(code: str, rules: yara.Rules) -> list:
    if not rules:
        return []

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.c',
        delete=False, encoding='utf-8'
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        matches = rules.match(tmp_path)
    finally:
        os.unlink(tmp_path)

    results = []
    for match in matches:
        meta        = match.meta
        cwe         = meta.get('cwe', 'Unknown')
        severity    = meta.get('severity', 'LOW')
        description = meta.get('description', '')

        matched_strings = list(set([
            s.instances[0].matched_data.decode('utf-8', errors='ignore')
            for s in match.strings
        ]))

        results.append({
            'rule'        : match.rule,
            'cwe'         : cwe,
            'severity'    : severity,
            'severity_int': SEVERITY_MAP.get(severity, 0),
            'description' : description,
            'strings'     : matched_strings
        })

    results.sort(key=lambda x: x['severity_int'], reverse=True)
    return results


def get_highest_severity(matches: list) -> str:
    if not matches:
        return 'NONE'
    return max(matches, key=lambda x: x['severity_int'])['severity']


def format_yara_results(matches: list) -> str:
    if not matches:
        return "No YARA rules matched."

    lines = [f"YARA Scanner — {len(matches)} match(es) found:\n"]
    for i, match in enumerate(matches, 1):
        lines.append(f"  [{i}] {match['rule']}")
        lines.append(f"       CWE      : {match['cwe']}")
        lines.append(f"       Severity : {match['severity']}")
        lines.append(f"       Detail   : {match['description']}")
        if match['strings']:
            lines.append(f"       Matched  : {', '.join(match['strings'][:3])}")
        lines.append("")
    return '\n'.join(lines)


if __name__ == '__main__':
    print("Testing YARA Scanner...")
    print("=" * 50)

    rules = compile_rules('rules/custom')

    test_code = """
    void vulnerable_function(char *input) {
        char buffer[64];
        strcpy(buffer, input);

        char *ptr = malloc(100);
        memcpy(ptr, input, strlen(input));

        char query[256];
        sprintf(query, "SELECT * FROM users WHERE name = " + input);

        system(argv[1]);
    }
    """

    print("\nScanning test code...")
    matches = scan_code(test_code, rules)
    print(format_yara_results(matches))
    print(f"Highest severity: {get_highest_severity(matches)}")
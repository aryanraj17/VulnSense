"""
severity_scorer.py
------------------
Converts ensemble scores into a CVSS-style severity rating.
Provides a human-readable risk assessment for each scan.
"""

import json
import os


# ── CVSS-style severity thresholds ───────────────────────────────────────────
SEVERITY_LEVELS = {
    'CRITICAL': (0.90, 1.00),
    'HIGH'    : (0.70, 0.90),
    'MEDIUM'  : (0.50, 0.70),
    'LOW'     : (0.30, 0.50),
    'SAFE'    : (0.00, 0.30),
}

# ── CWE base severity scores ─────────────────────────────────────────────────
# Based on real CVSS base scores for each CWE category
CWE_BASE_SEVERITY = {
    'CWE-119': 8.5,   # Buffer overflow — critical in most cases
    'CWE-120': 8.0,   # Buffer copy — high
    'CWE-125': 7.5,   # Out of bounds read — high
    'CWE-787': 8.5,   # Out of bounds write — critical
    'CWE-476': 6.5,   # NULL pointer — medium/high
    'CWE-416': 8.0,   # Use after free — high
    'CWE-190': 7.0,   # Integer overflow — high
    'CWE-20' : 6.5,   # Improper input — medium/high
    'CWE-89' : 9.0,   # SQL injection — critical
    'CWE-94' : 9.5,   # Code injection — critical
    'CWE-Other': 6.0, # Unknown — medium
    'Safe'   : 0.0,   # Safe — no risk
    'Unknown': 5.0,   # Unknown CWE — assume medium
}

# ── Remediation advice per CWE ────────────────────────────────────────────────
CWE_REMEDIATION = {
    'CWE-119': [
        "Replace strcpy() with strncpy() and specify buffer size",
        "Use strlcpy() for safer string copying",
        "Always validate input length before copying",
        "Consider using safe string libraries like SafeStr",
    ],
    'CWE-120': [
        "Always specify maximum length in string copy operations",
        "Use snprintf() instead of sprintf()",
        "Validate source string length before copying",
    ],
    'CWE-125': [
        "Add bounds checking before array access",
        "Validate index values against array size",
        "Use safe array access wrappers",
    ],
    'CWE-787': [
        "Validate all write operations stay within buffer bounds",
        "Use memset() with verified size parameters",
        "Consider address sanitizers during development",
    ],
    'CWE-476': [
        "Always check return value of malloc() for NULL",
        "Initialize pointers to NULL after declaration",
        "Set pointer to NULL immediately after free()",
    ],
    'CWE-416': [
        "Set pointer to NULL immediately after free()",
        "Use smart pointers or RAII patterns",
        "Avoid storing pointers to freed memory",
    ],
    'CWE-190': [
        "Check for integer overflow before arithmetic operations",
        "Use safe integer libraries",
        "Validate input ranges before multiplication",
        "Use SIZE_MAX checks before malloc with multiplication",
    ],
    'CWE-20' : [
        "Validate and sanitize all user inputs",
        "Use allowlist validation instead of blocklist",
        "Never pass unsanitized input to system functions",
    ],
    'CWE-89' : [
        "Use parameterized queries or prepared statements",
        "Never concatenate user input into SQL queries",
        "Use an ORM layer to abstract database access",
        "Apply input validation and escaping",
    ],
    'CWE-94' : [
        "Never pass user input to eval() or exec()",
        "Use subprocess with argument lists, not shell=True",
        "Validate and sanitize all dynamic code inputs",
        "Consider safer alternatives to dynamic execution",
    ],
    'CWE-Other': [
        "Review code for security best practices",
        "Consider a full security audit",
        "Use static analysis tools regularly",
    ],
    'Safe'   : ["No action required"],
    'Unknown': [
        "Review flagged code sections manually",
        "Run additional security analysis tools",
    ],
}


# ────────────────────────────────────────────────────────────────────────────
class SeverityScorer:
    """
    Converts ensemble scan results into a complete severity assessment.
    """

    def score(self, scan_result: dict) -> dict:
        """
        Takes ensemble scan result and returns full severity assessment.

        Args:
            scan_result: output from EnsembleScanner.scan()

        Returns:
        {
            'severity'      : 'HIGH',
            'cvss_score'    : 8.2,
            'risk_label'    : '🔴 HIGH RISK',
            'cwe'           : 'CWE-119',
            'cwe_score'     : 8.5,
            'final_score'   : 0.94,
            'remediation'   : [...],
            'summary'       : '...',
            'breakdown'     : {...}
        }
        """
        final_score    = scan_result.get('final_score', 0.0)
        cwe_prediction = scan_result.get('cwe_prediction', 'Unknown')
        is_vulnerable  = scan_result.get('is_vulnerable', False)
        yara_matches   = scan_result.get('yara_matches', [])
        dangerous_lines= scan_result.get('dangerous_lines', [])
        ind_scores     = scan_result.get('individual_scores', {})

        # ── Get severity level ────────────────────────────────────────────────
        severity = self._get_severity_level(final_score, is_vulnerable)

        # ── Calculate CVSS-style score (0-10) ────────────────────────────────
        cwe_base  = CWE_BASE_SEVERITY.get(cwe_prediction, 5.0)
        cvss_score = self._calculate_cvss(
            final_score, cwe_base, yara_matches, dangerous_lines
        )

        # ── Get remediation advice ────────────────────────────────────────────
        remediation = CWE_REMEDIATION.get(
            cwe_prediction,
            CWE_REMEDIATION['Unknown']
        )

        # ── Build summary ─────────────────────────────────────────────────────
        summary = self._build_summary(
            severity, cwe_prediction, cvss_score,
            yara_matches, dangerous_lines
        )

        # ── Risk label with emoji ─────────────────────────────────────────────
        risk_label = self._get_risk_label(severity)

        return {
            'severity'      : severity,
            'cvss_score'    : round(cvss_score, 1),
            'risk_label'    : risk_label,
            'cwe'           : cwe_prediction,
            'cwe_base_score': cwe_base,
            'final_score'   : final_score,
            'is_vulnerable' : is_vulnerable,
            'remediation'   : remediation,
            'summary'       : summary,
            'breakdown'     : {
                'codebert_score': round(ind_scores.get('codebert', 0), 3),
                'yara_score'    : round(ind_scores.get('yara', 0), 3),
                'ast_score'     : round(ind_scores.get('ast', 0), 3),
                'gnn_score'     : round(ind_scores.get('gnn', 0), 3),
                'yara_matches'  : len(yara_matches),
                'dangerous_lines': len(dangerous_lines),
            }
        }

    def _get_severity_level(self, score: float, is_vulnerable: bool) -> str:
        """Maps final score to severity level."""
        if not is_vulnerable:
            return 'SAFE'

        for level, (low, high) in SEVERITY_LEVELS.items():
            if low <= score <= high:
                return level

        return 'MEDIUM'

    def _calculate_cvss(
        self,
        final_score  : float,
        cwe_base     : float,
        yara_matches : list,
        dangerous_lines: list
    ) -> float:
        """
        Calculates a CVSS-style score from 0-10.

        Formula:
        - Base: ensemble score × 10 (60% weight)
        - CWE:  CWE base severity (30% weight)
        - Bonus: extra factors (10% weight)
        """
        base_component = final_score * 10 * 0.60
        cwe_component  = cwe_base * 0.30

        # bonus for multiple YARA matches and dangerous lines
        bonus = 0.0
        if len(yara_matches) >= 3:
            bonus += 0.5
        if len(dangerous_lines) >= 3:
            bonus += 0.5

        cvss = base_component + cwe_component + bonus
        return min(cvss, 10.0)

    def _get_risk_label(self, severity: str) -> str:
        """Returns emoji-labeled risk string."""
        labels = {
            'CRITICAL': '🔴 CRITICAL RISK',
            'HIGH'    : '🟠 HIGH RISK',
            'MEDIUM'  : '🟡 MEDIUM RISK',
            'LOW'     : '🔵 LOW RISK',
            'SAFE'    : '🟢 SAFE',
        }
        return labels.get(severity, '⚪ UNKNOWN')

    def _build_summary(
        self,
        severity      : str,
        cwe           : str,
        cvss_score    : float,
        yara_matches  : list,
        dangerous_lines: list
    ) -> str:
        """Builds a human readable summary of the scan result."""
        if severity == 'SAFE':
            return "No significant vulnerabilities detected. Code appears safe."

        cwe_names = {
            'CWE-119': 'Buffer Overflow',
            'CWE-120': 'Buffer Copy Without Size Check',
            'CWE-125': 'Out-of-Bounds Read',
            'CWE-787': 'Out-of-Bounds Write',
            'CWE-476': 'NULL Pointer Dereference',
            'CWE-416': 'Use After Free',
            'CWE-190': 'Integer Overflow',
            'CWE-20' : 'Improper Input Validation',
            'CWE-89' : 'SQL Injection',
            'CWE-94' : 'Code Injection',
        }

        cwe_name = cwe_names.get(cwe, 'Unknown Vulnerability Type')
        yara_str = f"{len(yara_matches)} YARA rule(s) matched" if yara_matches else "no YARA matches"
        line_str = f"dangerous patterns on {len(dangerous_lines)} line(s)" if dangerous_lines else ""

        summary = (
            f"{severity} severity vulnerability detected. "
            f"Classified as {cwe} ({cwe_name}) "
            f"with a CVSS score of {cvss_score}/10. "
            f"Analysis found {yara_str}"
        )

        if line_str:
            summary += f" and {line_str}."
        else:
            summary += "."

        return summary


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Testing Severity Scorer...")
    print("=" * 55)

    scorer = SeverityScorer()

    # simulate ensemble output for vulnerable code
    vulnerable_result = {
        'final_score'   : 0.9425,
        'is_vulnerable' : True,
        'cwe_prediction': 'CWE-119',
        'yara_matches'  : [
            {'rule': 'CWE119_BufferOverflow', 'severity': 'HIGH'},
            {'rule': 'CWE94_CodeInjection',   'severity': 'CRITICAL'},
        ],
        'dangerous_lines': [4, 5, 7, 9, 10],
        'individual_scores': {
            'codebert': 0.950,
            'yara'    : 1.000,
            'ast'     : 0.800,
            'gnn'     : 0.950,
        }
    }

    # simulate ensemble output for safe code
    safe_result = {
        'final_score'   : 0.7125,
        'is_vulnerable' : False,
        'cwe_prediction': 'Safe',
        'yara_matches'  : [],
        'dangerous_lines': [],
        'individual_scores': {
            'codebert': 0.950,
            'yara'    : 0.000,
            'ast'     : 0.000,
            'gnn'     : 0.950,
        }
    }

    print("\n--- Vulnerable Code Assessment ---")
    v_score = scorer.score(vulnerable_result)
    print(f"  Risk Label  : {v_score['risk_label']}")
    print(f"  Severity    : {v_score['severity']}")
    print(f"  CVSS Score  : {v_score['cvss_score']}/10")
    print(f"  CWE         : {v_score['cwe']}")
    print(f"  Summary     : {v_score['summary']}")
    print(f"\n  Remediation steps:")
    for i, step in enumerate(v_score['remediation'], 1):
        print(f"    {i}. {step}")
    print(f"\n  Score breakdown: {v_score['breakdown']}")

    print("\n--- Safe Code Assessment ---")
    s_score = scorer.score(safe_result)
    print(f"  Risk Label  : {s_score['risk_label']}")
    print(f"  CVSS Score  : {s_score['cvss_score']}/10")
    print(f"  Summary     : {s_score['summary']}")
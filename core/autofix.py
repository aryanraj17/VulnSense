"""
autofix.py
----------
Uses Groq (Llama 3 70B) to suggest fixes for detected vulnerabilities.
Free tier: 1000 requests/day, no credit card needed.
Get your key at: console.groq.com
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


# ── Config ────────────────────────────────────────────────────────────────────
GROQ_MODEL = 'llama-3.3-70b-versatile'


# ── CWE descriptions for better prompts ──────────────────────────────────────
CWE_DESCRIPTIONS = {
    'CWE-119': 'Buffer Overflow — memory written beyond allocated buffer bounds',
    'CWE-120': 'Buffer Copy Without Size Check — unbounded string copy operation',
    'CWE-125': 'Out-of-Bounds Read — reading memory outside allocated buffer',
    'CWE-787': 'Out-of-Bounds Write — writing memory outside allocated buffer',
    'CWE-476': 'NULL Pointer Dereference — pointer used without NULL check',
    'CWE-416': 'Use After Free — memory accessed after being freed',
    'CWE-190': 'Integer Overflow — arithmetic result exceeds type bounds',
    'CWE-20' : 'Improper Input Validation — user input not validated',
    'CWE-89' : 'SQL Injection — user input in SQL query without sanitization',
    'CWE-94' : 'Code Injection — user input passed to code execution function',
    'CWE-Other': 'Security Vulnerability — general security issue detected',
}


# ────────────────────────────────────────────────────────────────────────────
class AutoFixer:
    """
    Generates fix suggestions for vulnerable code using Groq + Llama 3.
    Falls back to rule-based fixes if API key not available.
    """

    def __init__(self):
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initializes Groq client."""
        try:
            from groq import Groq
            api_key = os.getenv('GROQ_API_KEY')
            if api_key:
                masked = api_key[:8] + '...' if api_key else 'None'
                print(f"  AutoFixer: Groq initialized (key: {masked})")
            else:
                print("  AutoFixer: GROQ_API_KEY not found in .env")
                print("  Get free key at console.groq.com")
                print("  Using rule-based fallback fixes for now")
        except ImportError:
            print("  AutoFixer: groq not installed. Run: pip install groq")

    def _build_prompt(
        self,
        code           : str,
        cwe            : str,
        dangerous_lines: list,
        yara_matches   : list
    ) -> str:
        """
        Builds a detailed prompt for Llama 3.
        More context = better fix suggestions.
        """
        cwe_desc = CWE_DESCRIPTIONS.get(cwe, 'Security vulnerability')
        line_str = ', '.join(map(str, dangerous_lines)) if dangerous_lines else 'unknown'
        yara_str = ', '.join([m['rule'] for m in yara_matches[:3]]) if yara_matches else 'none'

        prompt = f"""You are an expert security engineer specializing in C/C++ vulnerabilities.

VULNERABILITY DETAILS:
- Type     : {cwe} — {cwe_desc}
- Dangerous lines: {line_str}
- YARA rules matched: {yara_str}

VULNERABLE CODE:
```c
{code.strip()}
```

INSTRUCTIONS:
1. Fix the specific vulnerability while preserving original functionality
2. Add brief inline comments on changed lines explaining what was fixed
3. Use safe alternatives (strncpy instead of strcpy, fgets instead of gets etc.)
4. Add NULL checks after malloc calls
5. Set pointers to NULL after free
6. Return ONLY the fixed code inside a code block — nothing else

FIXED CODE:
```c"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Calls Groq API with the prompt."""
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.1    # low temp = more deterministic fixes
        )
        return response.choices[0].message.content

    def _clean_response(self, response: str) -> str:
        """
        Cleans LLM response.
        Removes markdown code blocks and extracts just the code.
        """
        if '```c' in response:
            parts = response.split('```c')
            if len(parts) > 1:
                return parts[1].split('```')[0].strip()

        if '```' in response:
            parts = response.split('```')
            if len(parts) >= 2:
                return parts[1].strip()

        return response.strip()

    def _generate_diff(self, original: str, fixed: str) -> list:
        """
        Generates line by line diff between original and fixed code.
        Returns list of (status, line) tuples.
        status: '+' = added, '-' = removed, '=' = unchanged
        """
        import difflib

        original_lines = original.strip().splitlines()
        fixed_lines    = fixed.strip().splitlines()
        diff           = []
        matcher        = difflib.SequenceMatcher(None, original_lines, fixed_lines)

        for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
            if opcode == 'equal':
                for line in original_lines[i1:i2]:
                    diff.append(('=', line))
            elif opcode == 'replace':
                for line in original_lines[i1:i2]:
                    diff.append(('-', line))
                for line in fixed_lines[j1:j2]:
                    diff.append(('+', line))
            elif opcode == 'delete':
                for line in original_lines[i1:i2]:
                    diff.append(('-', line))
            elif opcode == 'insert':
                for line in fixed_lines[j1:j2]:
                    diff.append(('+', line))

        return diff

    def format_diff(self, diff: list) -> str:
        """Returns readable diff string for terminal/display."""
        lines = []
        for status, line in diff:
            if status == '+':
                lines.append(f"+ {line}")
            elif status == '-':
                lines.append(f"- {line}")
            else:
                lines.append(f"  {line}")
        return '\n'.join(lines)

    def get_fix(
        self,
        code           : str,
        cwe            : str  = 'CWE-Other',
        dangerous_lines: list = None,
        yara_matches   : list = None
    ) -> dict:
        """
        Main method — gets fix suggestion for vulnerable code.

        Args:
            code            : vulnerable source code
            cwe             : CWE category detected
            dangerous_lines : line numbers flagged as dangerous
            yara_matches    : YARA rule matches from scanner

        Returns:
        {
            'original_code' : '...',
            'fixed_code'    : '...',
            'diff'          : [...],
            'provider'      : 'groq' or 'fallback_rules',
            'cwe'           : 'CWE-119',
            'success'       : True/False
        }
        """
        if dangerous_lines is None:
            dangerous_lines = []
        if yara_matches is None:
            yara_matches = []

        # use LLM if available
        if self.client:
            try:
                prompt     = self._build_prompt(code, cwe, dangerous_lines, yara_matches)
                raw        = self._call_llm(prompt)
                fixed_code = self._clean_response(raw)
                diff       = self._generate_diff(code, fixed_code)

                return {
                    'original_code': code,
                    'fixed_code'   : fixed_code,
                    'diff'         : diff,
                    'provider'     : 'groq',
                    'cwe'          : cwe,
                    'success'      : True
                }

            except Exception as e:
                print(f"  Groq API error: {e}")
                print("  Falling back to rule-based fixes")

        # fallback to rule-based fixes
        return self._fallback_fix(code, cwe)

    def _fallback_fix(self, code: str, cwe: str) -> dict:
        """
        Rule-based fallback fixes when Groq API not available.
        Handles the most common vulnerability patterns automatically.
        """
        import re
        fixed = code

        if cwe in ('CWE-119', 'CWE-120'):
            # strcpy → strncpy
            fixed = re.sub(
                r'strcpy\s*\((\w+)\s*,\s*(\w+)\s*\)',
                r'strncpy(\1, \2, sizeof(\1)-1)',
                fixed
            )
            # gets → fgets
            fixed = re.sub(
                r'gets\s*\((\w+)\s*\)',
                r'fgets(\1, sizeof(\1), stdin)',
                fixed
            )
            # sprintf → snprintf
            fixed = re.sub(
                r'sprintf\s*\((\w+)\s*,',
                r'snprintf(\1, sizeof(\1),',
                fixed
            )

        elif cwe == 'CWE-476':
            # add NULL check after malloc
            fixed = re.sub(
                r'((\w+)\s*=\s*malloc\([^;]+;)',
                r'\1\n    if (\2 == NULL) { return; }  /* NULL check added */',
                fixed
            )

        elif cwe == 'CWE-416':
            # set pointer to NULL after free
            fixed = re.sub(
                r'free\s*\((\w+)\s*\)\s*;',
                r'free(\1);\n    \1 = NULL;  /* prevent use-after-free */',
                fixed
            )

        elif cwe == 'CWE-190':
            # add overflow check before malloc with multiplication
            fixed = re.sub(
                r'malloc\s*\((\w+)\s*\*\s*(\w+)\s*\)',
                r'malloc(\1 * \2)  /* TODO: add overflow check for \1 * \2 */',
                fixed
            )

        diff = self._generate_diff(code, fixed)

        return {
            'original_code': code,
            'fixed_code'   : fixed,
            'diff'         : diff,
            'provider'     : 'fallback_rules',
            'cwe'          : cwe,
            'success'      : fixed != code
        }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Testing AutoFixer...")
    print("=" * 55)

    fixer = AutoFixer()

    test_code = """
void vulnerable_function(char *input, int size) {
    char buffer[64];
    char *ptr = malloc(size);
    strcpy(buffer, input);
    if (ptr != NULL) {
        memcpy(ptr, input, strlen(input));
    }
    free(ptr);
    system(input);
}"""

    print("\nGenerating fix for CWE-119 Buffer Overflow...")
    result = fixer.get_fix(
        code            = test_code,
        cwe             = 'CWE-119',
        dangerous_lines = [4, 5, 7, 9],
        yara_matches    = [{'rule': 'CWE119_BufferOverflow_strcpy'}]
    )

    print(f"\n  Provider : {result['provider']}")
    print(f"  Success  : {result['success']}")
    print(f"\n  Diff (- removed, + added):")
    print(fixer.format_diff(result['diff']))

    print(f"\n  Fixed code:")
    print(result['fixed_code'])

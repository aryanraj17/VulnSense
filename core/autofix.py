"""
autofix.py
----------
Uses Groq (Llama 3 70B) to suggest fixes for detected vulnerabilities.
Free tier: 1000 requests/day, no credit card needed.
Get your key at: console.groq.com
"""

import os
import re
import sys
import shutil
import tempfile
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("  Warning: python-dotenv not installed; using OS env vars only.")


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
    'CWE-20': 'Improper Input Validation — user input not validated',
    'CWE-89': 'SQL Injection — user input in SQL query without sanitization',
    'CWE-94': 'Code Injection — user input passed to code execution function',
    'CWE-Other': 'Security Vulnerability — general security issue detected',
}


class AutoFixer:
    """
    Generates fix suggestions for vulnerable code using Groq + Llama 3.
    Falls back to rule-based fixes if API key not available or LLM output is invalid.
    """

    def __init__(self):
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initializes Groq client safely."""
        try:
            from groq import Groq
            api_key = os.getenv('GROQ_API_KEY')

            if api_key and api_key.strip():
                self.client = Groq(api_key=api_key.strip())
                masked = api_key[:8] + '...'
                print(f"  AutoFixer: Groq initialized (key: {masked})")
            else:
                self.client = None
                print("  AutoFixer: GROQ_API_KEY not found in environment/.env")
                print("  Get free key at console.groq.com")
                print("  Using rule-based fallback fixes for now")
        except ImportError:
            self.client = None
            print("  AutoFixer: groq not installed. Run: pip install groq")
        except Exception as e:
            self.client = None
            print(f"  AutoFixer: client init error: {e}")

    def _build_prompt(
        self,
        code: str,
        cwe: str,
        dangerous_lines: list,
        yara_matches: list
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
            temperature=0.1
        )
        return response.choices[0].message.content or ""

    def _extract_code_block(self, response: str) -> str:
        """
        Extract code from fenced blocks first; fallback to whole text.
        Handles ```c, ```cpp, and generic ``` blocks.
        """
        if not response:
            return ""

        patterns = [
            r"```c\s*(.*?)```",
            r"```cpp\s*(.*?)```",
            r"```C\s*(.*?)```",
            r"```.*?\n(.*?)```",
        ]
        for pat in patterns:
            m = re.search(pat, response, flags=re.DOTALL)
            if m:
                return m.group(1).strip()

        return response.strip()

    def _clean_response(self, response: str) -> str:
        """Extracts and normalizes code from LLM response."""
        return self._extract_code_block(response).strip()

    def _balanced_braces(self, code: str) -> bool:
        """Quick structural sanity check for C-like code."""
        stack = []
        pairs = {')': '(', ']': '[', '}': '{'}
        opens = set(pairs.values())
        closes = set(pairs.keys())

        for ch in code:
            if ch in opens:
                stack.append(ch)
            elif ch in closes:
                if not stack or stack[-1] != pairs[ch]:
                    return False
                stack.pop()

        return len(stack) == 0

    def _looks_like_c_code(self, code: str) -> bool:
        """Heuristic check that output resembles C/C++ code."""
        if not code or len(code.strip()) < 10:
            return False

        signals = [
            ';' in code,
            '{' in code and '}' in code,
            any(x in code for x in ['if (', 'for (', 'while (', 'return']),
        ]
        return sum(bool(x) for x in signals) >= 2

    def _compile_check_c(self, code: str) -> tuple[bool, str]:
        """
        Optional compile check (syntax-level) if compiler exists.
        Returns (ok, message).
        """
        compiler = shutil.which("gcc") or shutil.which("clang")
        if not compiler:
            return True, "No gcc/clang available; compile check skipped"

        tmp_c = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False, encoding="utf-8") as f:
                tmp_c = f.name
                f.write(code)

            proc = subprocess.run(
                [compiler, "-fsyntax-only", tmp_c],
                capture_output=True,
                text=True
            )
            if proc.returncode == 0:
                return True, "Compile syntax check passed"

            err = (proc.stderr or proc.stdout or "").strip()
            return False, f"Compile check failed: {err[:500]}"
        except Exception as e:
            return False, f"Compile check error: {e}"
        finally:
            if tmp_c and os.path.exists(tmp_c):
                try:
                    os.unlink(tmp_c)
                except Exception:
                    pass

    def _validate_fixed_code(self, original: str, fixed: str) -> tuple[bool, str]:
        """Consolidated validator before accepting LLM output."""
        if not fixed or not fixed.strip():
            return False, "Empty fixed code"

        if fixed.strip() == original.strip():
            return False, "LLM returned unchanged code"

        if not self._looks_like_c_code(fixed):
            return False, "Output does not look like valid C/C++ code"

        if not self._balanced_braces(fixed):
            return False, "Unbalanced brackets/braces/parentheses"

        ok, msg = self._compile_check_c(fixed)
        if not ok:
            return False, msg

        return True, "Validation passed"

    def _generate_diff(self, original: str, fixed: str) -> list:
        """
        Generates line-by-line diff between original and fixed code.
        Returns list of (status, line) tuples.
        status: '+' = added, '-' = removed, '=' = unchanged
        """
        import difflib

        original_lines = original.strip().splitlines()
        fixed_lines = fixed.strip().splitlines()
        diff = []
        matcher = difflib.SequenceMatcher(None, original_lines, fixed_lines)

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
        code: str,
        cwe: str = 'CWE-Other',
        dangerous_lines: list = None,
        yara_matches: list = None
    ) -> dict:
        """
        Main method — gets fix suggestion for vulnerable code.
        Hardened with validation + safe fallback.
        """
        if dangerous_lines is None:
            dangerous_lines = []
        if yara_matches is None:
            yara_matches = []

        if self.client:
            try:
                prompt = self._build_prompt(code, cwe, dangerous_lines, yara_matches)
                raw = self._call_llm(prompt)
                fixed_code = self._clean_response(raw)

                valid, reason = self._validate_fixed_code(code, fixed_code)
                if valid:
                    diff = self._generate_diff(code, fixed_code)
                    return {
                        'original_code': code,
                        'fixed_code': fixed_code,
                        'diff': diff,
                        'provider': 'groq',
                        'cwe': cwe,
                        'success': True,
                        'validation': reason
                    }

                print(f"  AutoFixer: LLM output rejected -> {reason}")
                print("  Falling back to rule-based fixes")

            except Exception as e:
                print(f"  Groq API error: {e}")
                print("  Falling back to rule-based fixes")

        fallback = self._fallback_fix(code, cwe)
        fallback['validation'] = 'Fallback mode'
        return fallback

    def _fallback_fix(self, code: str, cwe: str) -> dict:
        """
        Rule-based fallback fixes when Groq API not available.
        Handles common vulnerability patterns automatically.
        """
        fixed = code

        if cwe in ('CWE-119', 'CWE-120'):
            fixed = re.sub(
                r'strcpy\s*\((\w+)\s*,\s*(\w+)\s*\)',
                r'strncpy(\1, \2, sizeof(\1)-1)',
                fixed
            )
            fixed = re.sub(
                r'gets\s*\((\w+)\s*\)',
                r'fgets(\1, sizeof(\1), stdin)',
                fixed
            )
            fixed = re.sub(
                r'sprintf\s*\((\w+)\s*,',
                r'snprintf(\1, sizeof(\1),',
                fixed
            )

        elif cwe == 'CWE-476':
            fixed = re.sub(
                r'((\w+)\s*=\s*malloc\([^;]+;)',
                r'\1\n    if (\2 == NULL) { return; }  /* NULL check added */',
                fixed
            )

        elif cwe == 'CWE-416':
            fixed = re.sub(
                r'free\s*\((\w+)\s*\)\s*;',
                r'free(\1);\n    \1 = NULL;  /* prevent use-after-free */',
                fixed
            )

        elif cwe == 'CWE-190':
            fixed = re.sub(
                r'malloc\s*\((\w+)\s*\*\s*(\w+)\s*\)',
                r'malloc(\1 * \2)  /* TODO: add overflow check for \1 * \2 */',
                fixed
            )

        diff = self._generate_diff(code, fixed)
        changed = fixed.strip() != code.strip()

        return {
            'original_code': code,
            'fixed_code': fixed,
            'diff': diff,
            'provider': 'fallback_rules',
            'cwe': cwe,
            'success': changed
        }


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
        code=test_code,
        cwe='CWE-119',
        dangerous_lines=[4, 5, 7, 9],
        yara_matches=[{'rule': 'CWE119_BufferOverflow_strcpy'}]
    )

    print(f"\n  Provider   : {result['provider']}")
    print(f"  Success    : {result['success']}")
    print(f"  Validation : {result.get('validation', 'N/A')}")
    print("\n  Diff (- removed, + added):")
    print(fixer.format_diff(result['diff']))

    print("\n  Fixed code:")
    print(result['fixed_code'])
"""
build_grammars.py
-----------------
Verifies tree-sitter-languages is working correctly.
No building needed — grammars come pre-compiled.
"""

from tree_sitter_languages import get_language, get_parser

def verify_grammars():
    print("Verifying tree-sitter grammars...")

    # test C
    try:
        c_parser = get_parser('c')
        tree = c_parser.parse(b"int main() { return 0; }")
        print("  C grammar     : OK")
    except Exception as e:
        print(f"  C grammar     : FAILED — {e}")

    # test Python
    try:
        py_parser = get_parser('python')
        tree = py_parser.parse(b"def main(): pass")
        print("  Python grammar: OK")
    except Exception as e:
        print(f"  Python grammar: FAILED — {e}")

    print("\nDone! Grammars ready.")

if __name__ == '__main__':
    verify_grammars()
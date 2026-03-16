"""
ast_parser.py
-------------
Parses C/C++ and Python code into Abstract Syntax Trees
using tree-sitter. Extracts features for GNN training
and provides structural analysis of code.
"""

import os
import json
import numpy as np
from pathlib import Path

# ── Load compiled grammars ───────────────────────────────────────────────────
try:
    from tree_sitter_languages import get_language, get_parser as get_ts_parser
    C_LANGUAGE      = get_language('c')
    PYTHON_LANGUAGE = get_language('python')
    print("  AST Parser: grammars loaded successfully")
except Exception as e:
    print(f"  AST Parser WARNING: could not load grammars — {e}")
    C_LANGUAGE      = None
    PYTHON_LANGUAGE = None

# ── Node types we care about for vulnerability detection ─────────────────────
VULNERABLE_NODE_TYPES = {
    'call_expression',
    'pointer_declarator',
    'subscript_expression',
    'assignment_expression',
    'binary_expression',
    'if_statement',
    'return_statement',
    'declaration',
}

# dangerous function names that often indicate vulnerabilities
DANGEROUS_FUNCTIONS = {
    'strcpy', 'strcat', 'gets', 'sprintf', 'scanf',
    'memcpy', 'memset', 'malloc', 'free', 'realloc',
    'system', 'exec', 'popen', 'eval',
    'printf', 'fprintf', 'vsprintf',
}


# ── Core Parser Class ─────────────────────────────────────────────────────────
class ASTParser:

    def __init__(self):
        self.c_parser      = None
        self.python_parser = None
        try:
            from tree_sitter_languages import get_parser as get_ts_parser
            self.c_parser      = get_ts_parser('c')
            self.python_parser = get_ts_parser('python')
        except Exception as e:
            print(f"  Parser init error: {e}")

    def get_parser(self, lang: str):
        lang = lang.lower().strip()
        if lang in ('python', 'py'):
            return self.python_parser
        return self.c_parser

    def parse(self, code: str, lang: str = 'c'):
        parser = self.get_parser(lang)
        if not parser:
            return None
        try:
            tree = parser.parse(bytes(code, 'utf-8'))
            return tree
        except Exception as e:
            print(f"  Parse error: {e}")
            return None

    def get_all_nodes(self, tree) -> list:
        if not tree:
            return []

        nodes            = []
        cursor           = tree.walk()
        visited_children = False

        while True:
            node = cursor.node

            if not visited_children:
                nodes.append({
                    'type'       : node.type,
                    'text'       : node.text.decode('utf-8', errors='ignore'),
                    'start_line' : node.start_point[0],
                    'end_line'   : node.end_point[0],
                    'start_col'  : node.start_point[1],
                    'end_col'    : node.end_point[1],
                    'child_count': node.child_count,
                })

                if cursor.goto_first_child():
                    continue

            visited_children = False

            if cursor.goto_next_sibling():
                continue

            if not cursor.goto_parent():
                break

            visited_children = True

        return nodes

    def extract_features(self, code: str, lang: str = 'c') -> dict:
        tree = self.parse(code, lang)
        if not tree:
            return self._empty_features()

        nodes = self.get_all_nodes(tree)
        if not nodes:
            return self._empty_features()

        # count node types
        node_counts = {}
        for node in nodes:
            ntype = node['type']
            node_counts[ntype] = node_counts.get(ntype, 0) + 1

        # find dangerous function calls
        dangerous_calls = []
        for node in nodes:
            if node['type'] == 'identifier':
                text = node['text'].strip()
                if text in DANGEROUS_FUNCTIONS:
                    dangerous_calls.append({
                        'function': text,
                        'line'    : node['start_line'] + 1
                    })

        # calculate AST depth
        depth = self._calculate_depth(tree.root_node)

        # cyclomatic complexity
        cyclomatic     = 1
        decision_nodes = {
            'if_statement', 'while_statement', 'for_statement',
            'case_statement', '&&', '||'
        }
        for node in nodes:
            if node['type'] in decision_nodes:
                cyclomatic += 1

        # fixed size feature vector for GNN
        feature_vector = [
            node_counts.get('call_expression', 0),
            node_counts.get('pointer_declarator', 0),
            node_counts.get('subscript_expression', 0),
            node_counts.get('assignment_expression', 0),
            node_counts.get('binary_expression', 0),
            node_counts.get('if_statement', 0),
            node_counts.get('return_statement', 0),
            node_counts.get('declaration', 0),
            node_counts.get('function_definition', 0),
            len(dangerous_calls),
            depth,
            len(nodes),
            cyclomatic,
            int('malloc(' in code),
            int('free(' in code),
            int('*' in code),
            int('strcpy(' in code or 'strcat(' in code),
            int('system(' in code or 'exec(' in code),
        ]

        return {
            'node_counts'    : node_counts,
            'dangerous_calls': dangerous_calls,
            'depth'          : depth,
            'num_nodes'      : len(nodes),
            'num_functions'  : node_counts.get('function_definition', 0),
            'has_pointer'    : node_counts.get('pointer_declarator', 0) > 0,
            'has_malloc'     : 'malloc(' in code,
            'has_free'       : 'free(' in code,
            'cyclomatic'     : cyclomatic,
            'feature_vector' : feature_vector,
        }

    def get_dangerous_lines(self, code: str, lang: str = 'c') -> list:
        tree = self.parse(code, lang)
        if not tree:
            return []

        nodes           = self.get_all_nodes(tree)
        dangerous_lines = set()

        for node in nodes:
            if node['type'] == 'identifier':
                if node['text'].strip() in DANGEROUS_FUNCTIONS:
                    dangerous_lines.add(node['start_line'] + 1)

        return sorted(list(dangerous_lines))

    def _calculate_depth(self, node, current_depth: int = 0) -> int:
        if not node.children:
            return current_depth
        return max(
            self._calculate_depth(child, current_depth + 1)
            for child in node.children
        )

    def _empty_features(self) -> dict:
        return {
            'node_counts'    : {},
            'dangerous_calls': [],
            'depth'          : 0,
            'num_nodes'      : 0,
            'num_functions'  : 0,
            'has_pointer'    : False,
            'has_malloc'     : False,
            'has_free'       : False,
            'cyclomatic'     : 1,
            'feature_vector' : [0] * 18,
        }

    def visualize_ast(self, code: str, lang: str = 'c', max_depth: int = 3) -> str:
        tree = self.parse(code, lang)
        if not tree:
            return "Could not parse code."

        lines = []
        self._print_tree(tree.root_node, lines, 0, max_depth)
        return '\n'.join(lines)

    def _print_tree(self, node, lines: list, depth: int, max_depth: int):
        if depth > max_depth:
            return

        indent  = '  ' * depth
        text    = node.text.decode('utf-8', errors='ignore').strip()
        preview = f' → "{text[:30]}"' if text and node.child_count == 0 else ''
        lines.append(f"{indent}[{node.type}]{preview}")

        for child in node.children:
            self._print_tree(child, lines, depth + 1, max_depth)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Testing AST Parser...")
    print("=" * 55)

    parser = ASTParser()

    test_code = """
    void vulnerable_function(char *input, int size) {
        char buffer[64];
        char *ptr = malloc(size);

        strcpy(buffer, input);

        if (ptr != NULL) {
            memcpy(ptr, input, strlen(input));
        }

        free(ptr);
        ptr->data = 1;
    }
    """

    print("\nExtracting AST features...")
    features = parser.extract_features(test_code, lang='c')

    print(f"  Total AST nodes  : {features['num_nodes']}")
    print(f"  AST depth        : {features['depth']}")
    print(f"  Cyclomatic       : {features['cyclomatic']}")
    print(f"  Has pointer      : {features['has_pointer']}")
    print(f"  Has malloc       : {features['has_malloc']}")
    print(f"  Has free         : {features['has_free']}")
    print(f"  Feature vector   : {features['feature_vector']}")

    print(f"\nDangerous function calls found:")
    for call in features['dangerous_calls']:
        print(f"  Line {call['line']}: {call['function']}()")

    print(f"\nDangerous lines: {parser.get_dangerous_lines(test_code)}")

    print(f"\nAST Structure (depth 2):")
    print(parser.visualize_ast(test_code, max_depth=2))
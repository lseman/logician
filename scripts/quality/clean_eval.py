#!/usr/bin/env python3
"""
Clean eval output helper. Strips ANSI escape codes and formats output.
Usage: python3 clean_eval.py <code>
"""
import sys
import re

def strip_ansi(text):
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)

def format_code_output(code, output):
    """Format code and output with clean separation."""
    lines = output.strip().split('\n')
    result = []
    
    # Show the code block
    result.append('```')
    result.append(code.strip())
    result.append('```')
    result.append('')
    
    # Show the output
    if lines:
        cleaned = [strip_ansi(line) for line in lines]
        result.extend(cleaned)
    
    return '\n'.join(result)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 clean_eval.py <code>')
        sys.exit(1)
    
    code = sys.argv[1]
    output = sys.stdin.read()
    print(format_code_output(code, output))

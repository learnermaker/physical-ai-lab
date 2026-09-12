import ast
content = open('api/server.py', encoding='utf-8').read()
try:
    ast.parse(content)
    print('Syntax OK')
except SyntaxError as e:
    print(f'SyntaxError: {e}')

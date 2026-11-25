import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / 'backend'))
from fastapi.testclient import TestClient
from app.main import app
c = TestClient(app)
PY = 'def add(a, b):\n    """Add."""\n    return a + b\n'
C = '// sum\nint f(int n) {\n int s = 0;\n for (int i=0;i<n;i++) { if (i>2) s+=i; }\n return s;\n}\n'
J = 'class A {\n // go\n static int g(int n) {\n  int unused = 1;\n  while (n>0) { n--; }\n  return n;\n }\n}\n'
def post(l, code, fn='pasted code'): return c.post('/api/analyze', json=dict(language=l, code=code, filename=fn))
def test_python():
    r = post('python', PY).json(); assert r['metrics']['functions'] == 1 and 0 <= r['score'] <= 100
def test_c():
    r = post('c', C).json(); assert r['metrics']['functions'] == 1 and r['metrics']['loops'] == 1
def test_java():
    r = post('java', J).json(); assert r['metrics']['classes'] == 1 and any(i['title'] == 'Unused variable' for i in r['issues'])
def test_errors():
    assert post('python', '  ').status_code == 400
    assert post('python', 'def (:').status_code == 422
    assert post('c', C, 'x.py').status_code == 400
def test_bad_scores_lower():
    bad = 'def f(a):\n' + ''.join('    ' * i + 'for x in a:\n' for i in range(1, 7)) + '    ' * 7 + 'q = 1\n'
    assert post('python', bad).json()['score'] < post('python', PY).json()['score']
def test_report():
    r = post('python', PY).json(); assert b'ISSUES' in c.post('/api/report', json=r).content

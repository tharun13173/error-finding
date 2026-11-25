"""Language-aware feature extraction + rule-based issue detection (static; code is never executed)."""
import ast, io, re, tokenize
import numpy as np, pandas as pd

KW = {'if','for','while','switch','return','else','sizeof','catch','do','synchronized'}
DECL = re.compile(r'\b(?:int|long|float|double|char|boolean|String|short|size_t)\s+\**(\w+)\s*(?:=|;|,|\[)')
CC = re.compile(r'\b(?:if|for|while|case|catch)\b|&&|\|\||\?')
FN = re.compile(r'(\w+)\s*\([^;{}()]*\)\s*(?:throws\s+[\w., ]+)?\{')
FEATS = ['comment_ratio','avg_cc','max_cc','max_nesting','avg_func_len','naming_quality','doc_ratio']

def strip_c_like(src):
    """Blank out comments/strings (keeps line numbers). Returns (clean_text, comment_line_set)."""
    out, i, n, line, com = [], 0, len(src), 1, set()
    while i < n:
        c, two = src[i], src[i:i+2]
        if two == '//':
            com.add(line)
            while i < n and src[i] != '\n': i += 1
        elif two == '/*':
            j = src.find('*/', i+2); j = n if j < 0 else j+2
            for ch in src[i:j]:
                if ch == '\n': out.append('\n'); line += 1
                else: com.add(line)
            i = j
        elif c in '"\'':
            out.append(c); i += 1
            while i < n and src[i] != c and src[i] != '\n': i += 2 if src[i] == '\\' else 1
            out.append(c); i += 1
        else:
            out.append(c); line += c == '\n'; i += 1
    return ''.join(out), com

def _py_depth(node, d=0):
    kids = [_py_depth(c, d+1) if isinstance(c,(ast.If,ast.For,ast.While,ast.Try,ast.With)) else _py_depth(c, d)
            for c in ast.iter_child_nodes(node)]
    return max(kids + [d])

def _py_cc(fn):
    t = 1
    for x in ast.walk(fn):
        if isinstance(x,(ast.If,ast.For,ast.While,ast.ExceptHandler,ast.IfExp,ast.comprehension)): t += 1
        elif isinstance(x, ast.BoolOp): t += len(x.values)-1
    return t

def analyze_python(code):
    tree = ast.parse(code)
    com = {t.start[0] for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type == tokenize.COMMENT}
    funcs, unused, names = [], [], []
    for fn in [n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]:
        stores = {n.id: n.lineno for n in ast.walk(fn) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
        loads = {n.id for n in ast.walk(fn) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
        unused += [(k, v) for k, v in stores.items() if k not in loads and not k.startswith('_')]
        names += list(stores) + [fn.name]
        funcs.append(dict(name=fn.name, line=fn.lineno, length=fn.end_lineno-fn.lineno+1, cc=_py_cc(fn),
                          nesting=_py_depth(fn), documented=ast.get_docstring(fn) is not None))
    allp = list(ast.walk(tree))
    return dict(funcs=funcs, classes=sum(isinstance(n,ast.ClassDef) for n in allp), names=names, unused=unused,
                loops=sum(isinstance(n,(ast.For,ast.While)) for n in allp), conds=sum(isinstance(n,ast.If) for n in allp),
                comments=com, exact=True)

def analyze_c_like(code, lang):
    clean, com = strip_c_like(code)
    funcs = []
    for m in FN.finditer(clean):
        if m.group(1) in KW: continue
        depth, j = 1, m.end()
        while j < len(clean) and depth:
            depth += (clean[j] == '{') - (clean[j] == '}'); j += 1
        body, d, mx = clean[m.end():j-1], 0, 0
        for ch in body:
            d += (ch == '{') - (ch == '}'); mx = max(mx, d)
        line = clean[:m.start()].count('\n') + 1
        funcs.append(dict(name=m.group(1), line=line, length=body.count('\n')+1, cc=1+len(CC.findall(body)),
                          nesting=mx+1, documented=(line-1 in com or line in com)))
    names = DECL.findall(clean)
    unused = [(v, clean[:clean.find(v)].count('\n')+1) for v in set(names) if len(re.findall(r'\b%s\b' % v, clean)) == 1]
    return dict(funcs=funcs, classes=len(re.findall(r'\bclass\s+\w+', clean)), names=names + [f['name'] for f in funcs],
                unused=unused, loops=len(re.findall(r'\b(?:for|while)\b', clean)), conds=len(re.findall(r'\bif\b', clean)),
                comments=com, exact=False)

SEV_W = {'Critical':15,'High':8,'Medium':4,'Low':1.5,'Informational':0.5}

def analyze(code, lang):
    lines = code.splitlines()
    r = analyze_python(code) if lang == 'python' else analyze_c_like(code, lang)
    blank = sum(1 for l in lines if not l.strip())
    loc = len(lines)
    F = pd.DataFrame(r['funcs'], columns=['name','line','length','cc','nesting','documented'])
    nm = pd.Series(r['names'], dtype=str)
    f = dict(
        comment_ratio=len(r['comments'])/max(loc-blank,1),
        avg_cc=float(F.cc.mean()) if len(F) else 1.0, max_cc=int(F.cc.max()) if len(F) else 1,
        max_nesting=int(F.nesting.max()) if len(F) else 0,
        avg_func_len=float(F.length.mean()) if len(F) else float(loc),
        naming_quality=float((nm.str.len().ge(3) | nm.isin(list('ijkn'))).mean()) if len(nm) else 1.0,
        doc_ratio=float(F.documented.mean()) if len(F) else 1.0)
    iss = []
    add = lambda t,s,l,d,rec,k='rule': iss.append(dict(title=t,severity=s,line=l,description=d,recommendation=rec,kind=k))
    for _, x in F.iterrows():
        if x.cc > 10: add('High cyclomatic complexity','High' if x.cc > 20 else 'Medium',x.line,f"'{x['name']}' has complexity {x.cc}.",'Split into smaller functions.')
        if x.length > 50: add('Long function','High' if x.length > 100 else 'Medium',x.line,f"'{x['name']}' is {x.length} lines.",'Extract helper functions.')
        if x.nesting >= 4: add('Excessive nesting','High' if x.nesting >= 6 else 'Medium',x.line,f"'{x['name']}' nests {x.nesting} levels deep.",'Use early returns / extract inner blocks.')
        if not x.documented: add('Missing documentation','Low',x.line,f"'{x['name']}' has no docstring/comment.",'Add a short description of purpose.')
    for v, l in r['unused']: add('Unused variable','Low',l,f"'{v}' is assigned but never used.",'Remove it.','ast' if r['exact'] else 'heuristic')
    short = sorted({v for v in r['names'] if len(v) < 3 and v not in 'ijkn'})
    if short: add('Non-descriptive names','Low',None,'Short names: '+', '.join(short[:8]),'Use meaningful names.','heuristic')
    if loc > 30 and f['comment_ratio'] < 0.05: add('Very few comments','Informational',None,f"Comment ratio {f['comment_ratio']:.1%}.",'Comment non-obvious logic.')
    I = pd.DataFrame(iss, columns=['title','severity','line','description','recommendation','kind'])
    pen = float(np.clip(I.severity.map(SEV_W).sum(), 0, 100)) if len(I) else 0.0
    f['rule_score'] = 100 - pen
    M = dict(total_lines=loc, blank_lines=blank, comment_lines=len(r['comments']), code_lines=loc-blank,
             functions=len(F), classes=r['classes'], loops=r['loops'], conditionals=r['conds'], variables=len(nm))
    return dict(metrics=M, features=f, functions=F, issues=I, exact=r['exact'])

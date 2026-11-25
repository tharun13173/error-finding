"""PPT pipeline: data collection (synthetic labelled corpus) -> features -> EDA -> RandomForest -> accuracy/precision/recall."""
import json, pathlib, joblib
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from .analyzer import analyze, FEATS

D = pathlib.Path(__file__).parent / 'model'
GOOD = ['total','count','price','average','result','items','limit','length','value']
BAD = ['a','b','x','t','q','z','d','tmp']

def _func(lang, k, good, rng):
    d = int(rng.integers(1, 4) if good else rng.integers(2, 7)); n = int(rng.integers(2, 9) if good else rng.integers(5, 28))
    desc = rng.random() < (.85 if good else .3); pool = GOOD if desc else BAD
    com = rng.random() < (.7 if good else .25); unused = (not good) and rng.random() < .5
    fn = f'compute{k}' if desc else f'f{k}'; v = pool[int(rng.integers(len(pool)))]
    py = lang == 'python'; L = []
    if com and not py: L.append('// computes a value')
    L.append({'python': f'def {fn}(n):', 'c': f'int {fn}(int n) {{', 'java': f'static int {fn}(int n) {{'}[lang])
    if com and py: L.append('    """Compute a value."""')
    ind = lambda i: '    ' * i
    L.append(ind(1) + (f'{v} = 0' if py else f'int {v} = 0;'))
    if unused: L.append(ind(1) + ('tmp_u = 1' if py else 'int tmp_u = 1;'))
    for lv in range(d):
        c = (f'for i{lv} in range(n):' if lv % 2 == 0 else f'if n > {lv}:') if py else \
            (f'for (int i{lv} = 0; i{lv} < n; i{lv}++) {{' if lv % 2 == 0 else f'if (n > {lv}) {{')
        L.append(ind(lv+1) + c)
    for s in range(n): L.append(ind(d+1) + (f'{v} = {v} + {s}' if py else f'{v} = {v} + {s};'))
    if not py:
        for lv in range(d, 0, -1): L.append(ind(lv) + '}')
    L.append(ind(1) + f'return {v}' + ('' if py else ';'))
    if not py: L.append('}')
    return '\n'.join(L)

def make_corpus(n=600, seed=7):
    rng = np.random.default_rng(seed); rows = []
    for i in range(n):
        lang = ['python','c','java'][i % 3]; good = i % 2 == 0
        body = '\n\n'.join(_func(lang, k, good, rng) for k in range(int(rng.integers(1, 4))))
        if lang == 'java': body = 'class Demo {\n' + body + '\n}'
        f = analyze(body, lang)['features']; rows.append({**{k: f[k] for k in FEATS}, 'lang': lang, 'label': int(good)})
    return pd.DataFrame(rows)

def train():
    df = make_corpus()
    X, y = df[FEATS], df.label
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=1)
    m = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=1).fit(Xtr, ytr)
    p = m.predict(Xte)
    eda = df.groupby('label')[FEATS].mean().round(3)
    info = dict(accuracy=accuracy_score(yte, p), precision=precision_score(yte, p), recall=recall_score(yte, p),
                n_train=len(Xtr), n_test=len(Xte), eda={('good' if k else 'needs_improvement'): v for k, v in eda.to_dict('index').items()},
                importance=dict(zip(FEATS, map(float, m.feature_importances_))),
                note='Trained on a synthetic, template-generated labelled corpus (see README). Scores show feature separability, not real-world accuracy.')
    D.mkdir(exist_ok=True); joblib.dump(m, D / 'model.pkl'); (D / 'info.json').write_text(json.dumps(info, indent=1))
    return m, info

def load():
    if not (D / 'model.pkl').exists(): return train()
    return joblib.load(D / 'model.pkl'), json.loads((D / 'info.json').read_text())

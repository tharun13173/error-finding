import io, datetime, pathlib
import numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from .analyzer import analyze, FEATS
from . import ml

app = FastAPI(title='Syntaxis')
MODEL, INFO = ml.load()
MAX = 200_000
EXT = {'python': ('.py',), 'c': ('.c', '.h'), 'java': ('.java',)}

class Req(BaseModel):
    language: str
    code: str
    filename: str = 'pasted code'

def band(s): return 'Excellent' if s >= 85 else 'Good' if s >= 70 else 'Needs Improvement' if s >= 50 else 'Poor'

@app.post('/api/analyze')
def run(r: Req):
    if r.language not in EXT: raise HTTPException(400, 'Language must be c, python or java.')
    if not r.code.strip(): raise HTTPException(400, 'Please enter or upload some code.')
    if len(r.code) > MAX: raise HTTPException(413, 'Code exceeds 200 KB limit.')
    if r.filename != 'pasted code' and not r.filename.lower().endswith(EXT[r.language]):
        raise HTTPException(400, f'File type does not match {r.language} ({", ".join(EXT[r.language])}).')
    try: a = analyze(r.code, r.language)
    except SyntaxError as e: raise HTTPException(422, f'Syntax error on line {e.lineno}: {e.msg}')
    except Exception: raise HTTPException(500, 'Analysis failed for this input.')
    x = pd.DataFrame([a['features']])[FEATS]
    p = float(MODEL.predict_proba(x)[0][1])
    score = int(round(float(np.average([a['features']['rule_score'], 100 * p], weights=[.6, .4]))))
    I = a['issues']
    sev = I.groupby('severity').size() if len(I) else pd.Series(dtype=int)
    sev = sev.reindex(['Critical','High','Medium','Low','Informational'], fill_value=0)
    cc = a['functions'].cc.to_numpy()
    return dict(filename=r.filename, language=r.language, timestamp=datetime.datetime.now().isoformat(timespec='seconds'),
        score=score, rating=band(score), ml_label='Good quality' if p >= .5 else 'Needs improvement', ml_prob_good=round(p, 3),
        rule_score=round(a['features']['rule_score'], 1), metrics=a['metrics'],
        complexity=dict(mean=round(float(cc.mean()), 2), max=int(cc.max()), p90=round(float(np.percentile(cc, 90)), 2)) if len(cc) else None,
        features={k: round(float(v), 3) for k, v in a['features'].items()},
        severity={k: int(v) for k, v in sev.items()}, issues=I.astype(object).where(I.notna(), None).to_dict('records'),
        functions=a['functions'].to_dict('records'), model=INFO,
        limitations=['Python uses the AST (exact structure).' if a['exact'] else 'C/Java use a comment/string-aware brace parser, not a full compiler; function, variable and unused-variable detection are heuristic.',
                     'Quality score = 60% rule-based penalties + 40% ML probability. Not an industry standard.'])

@app.post('/api/report')
def report(res: dict):
    b = io.StringIO()
    b.write(f"Syntaxis report\nfile,{res['filename']}\nlanguage,{res['language']}\ntimestamp,{res['timestamp']}\nscore,{res['score']} ({res['rating']})\n\nMETRICS\n")
    pd.Series(res['metrics']).rename('value').to_csv(b, header=True)
    b.write('\nISSUES\n'); pd.DataFrame(res['issues']).to_csv(b, index=False)
    return Response(b.getvalue(), media_type='text/csv', headers={'Content-Disposition': 'attachment; filename=syntaxis_report.csv'})

@app.get('/')
def index(): return FileResponse(pathlib.Path(__file__).parents[2] / 'frontend' / 'index.html')

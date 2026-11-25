# Syntaxis — Code Quality Analyser using Data Science
CGB1204 Data Science · Active Learning-I. Supports **C, Python, Java**.

## Run
```
pip install -r requirements.txt
cd backend && uvicorn app.main:app --reload      # open http://127.0.0.1:8000
python -m pytest tests                            # from the project root
```
(The ML model trains automatically on first start and is cached in `backend/app/model/`.)

## Pipeline (maps to the PPT)
1. **Input** – paste/upload code (≤200 KB, extension validated). Never executed.
2. **Feature extraction** – complexity (cyclomatic), comment ratio, naming quality (NLP-style name-length analysis), nesting, function length, doc ratio.
3. **EDA** – good vs needs-improvement feature means (`info.json`, shown via `eda`).
4. **ML** – RandomForest classifies *good* / *needs improvement*; reports accuracy, precision, recall.
5. **Rule-based checker** – long functions, complexity, nesting, unused variables, missing docs, naming; each with severity + plain-English fix.
6. **Score (0–100)** = 0.6 × rule score (100 − severity penalties) + 0.4 × 100·P(good). Bands: ≥85 Excellent, ≥70 Good, ≥50 Needs Improvement, else Poor. Transparent and *not* an industry standard.

**NumPy**: penalty clipping, weighted average for score, percentiles of complexity, corpus generation. **Pandas**: function/issue/feature DataFrames, groupby severity, EDA means, CSV report.

## Exact vs estimated
- Python: AST-based (exact structure; unused variable = AST).
- C/Java: comment/string-aware brace parser — **heuristic** (no full compiler). Variables, functions and unused detection are estimates.

## Known limitations (read before presenting)
- **Training data is synthetic**: the PPT calls for labelled code, but no real labelled dataset was available offline. A template generator produces good/bad snippets with overlapping parameter ranges. The reported ~98% accuracy shows feature separability, **not real-world accuracy**. Replace `make_corpus()` in `ml.py` with real labelled files (e.g. from GitHub/PROMISE) to make it genuine.
- No syntax-highlighted editor, no history/SQLite, no security-vulnerability detection (out of PPT scope).

## Assumptions
Plain HTML frontend served by FastAPI (no React build step) to keep it simple to demo. Header/branding renamed CodeDex → Syntaxis.

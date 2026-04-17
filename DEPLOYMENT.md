# CardioRisk AI — Deployment Guide & Validation Checklist

---

## 1 · System Architecture

```
USER INPUT (Streamlit form)
        │
        ▼
compute_engineered()          ← mirrors training pipeline exactly
        │  Derives: lifestyle_score, stress_index, cardio_fitness,
        │           sleep_hr_recovery, pulse_pressure, chol_hr_ratio
        ▼
build_input_vector()          ← reads feature_names.json for column order
        │  Fills missing fields with FEATURE_DEFAULTS (dataset medians)
        │  Returns 1-row DataFrame aligned to training schema
        ▼
StandardScaler.transform()    ← loaded from scaler.pkl
        │  Only scales continuous columns (skips binary 0/1 flags)
        ▼
model.predict_proba()         ← loaded from model.pkl
        │  Returns [P(no_cvd), P(cvd)]
        ▼
Risk classification
        │  < 15% → Low | 15–35% → Moderate | >35% → High
        ▼
Render results (gauge + badge + factors)
```

**Key design decision:** `feature_names.json` is the single source of truth
for column order. Any new feature must be added to both training and this file.

---

## 2 · Pre-Deployment Checklist

### Local run
- [ ] `python src/create_demo_model.py` runs without errors
- [ ] `models/` contains: `model.pkl`, `scaler.pkl`, `feature_names.json`, `feature_importance.csv`
- [ ] `streamlit run app.py` opens at `localhost:8501`
- [ ] Sidebar shows all green (✅ Model loaded, ✅ Scaler loaded, ✅ Features loaded)
- [ ] Clicking "Analyse Cardiovascular Risk" shows results panel
- [ ] Risk gauge renders with colour
- [ ] KPI cards display correct values
- [ ] No Python tracebacks in terminal

### Model integrity
- [ ] `feature_names.json` feature count matches `scaler.pkl` dimensions
- [ ] Changing all inputs from default → prediction changes
- [ ] High-risk inputs (age=70, stress=5, low steps) → High Risk output
- [ ] Low-risk inputs (age=35, active, good sleep) → Low Risk output
- [ ] No `KeyError` or `ValueError` in terminal during inference

### Deployment
- [ ] `requirements.txt` committed to repo root
- [ ] `models/` folder committed (check `.gitignore` is not excluding `.pkl`)
- [ ] `.streamlit/config.toml` committed
- [ ] `app.py` at repo root
- [ ] GitHub repo is public (or Streamlit Cloud has access)

---

## 3 · Step-by-Step GitHub Deployment

### 3a · Create repository

```bash
# Option A: GitHub CLI
gh repo create cardiorisk-ai --public --source=. --remote=origin --push

# Option B: Manual
# 1. Go to github.com → New repository
# 2. Name: cardiorisk-ai
# 3. Public, no template
# 4. Create
```

### 3b · Push all files

```bash
cd /path/to/cvd_app           # your project root

git init
git add .
git status                    # verify models/ is included

git commit -m "feat: CardioRisk AI — production Streamlit app"

git remote add origin https://github.com/YOUR_USERNAME/cardiorisk-ai.git
git branch -M main
git push -u origin main
```

**Verify on GitHub:**
- `app.py` at root ✓
- `requirements.txt` at root ✓
- `models/model.pkl` committed ✓
- `.streamlit/config.toml` committed ✓

### 3c · Large model files (> 50 MB)

If `model.pkl` exceeds 50 MB, use Git LFS:

```bash
git lfs install
git lfs track "models/*.pkl"
git add .gitattributes
git add models/
git commit -m "chore: track pkl files with LFS"
git push
```

---

## 4 · Streamlit Cloud Deployment

### Step 1 — Sign in
Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

### Step 2 — New App
Click **"New app"** → **"From existing repo"**

### Step 3 — Configure
| Field | Value |
|---|---|
| Repository | `YOUR_USERNAME/cardiorisk-ai` |
| Branch | `main` |
| Main file path | `app.py` |
| App URL | `cardiorisk-ai` (custom suffix) |

### Step 4 — Advanced settings (optional)
- Python version: `3.11`
- No secrets needed for this app

### Step 5 — Deploy
Click **"Deploy!"**

Build log appears. First deploy takes ~3 minutes (installing packages).

### Step 6 — Share
Your live URL:
```
https://YOUR_USERNAME-cardiorisk-ai-app-XXXXX.streamlit.app
```

Add this to your README badge and portfolio.

---

## 5 · Common Errors & Exact Fixes

### Error: `ModuleNotFoundError: No module named 'xgboost'`
```
Fix: requirements.txt must contain:  xgboost>=2.0.0
     Commit the updated file and redeploy.
```

### Error: `FileNotFoundError: models/model.pkl`
```
Fix: The models/ folder was likely in .gitignore.
     Check .gitignore — remove the line "models/" if present.
     Run:  git add models/ && git commit -m "add model" && git push
```

### Error: `ValueError: X has N features but scaler expects M`
```
Fix: scaler.pkl and feature_names.json are out of sync.
     Run export_model.py again and recommit both files.
     Never edit feature_names.json manually.
```

### Error: `AttributeError: 'RandomForestClassifier' has no attribute 'predict_proba'`
```
Fix: Rare — model was saved as a Decision Tree variant without proba.
     The app handles this with a try/except fallback to predict().
     No action needed — check prediction output is still valid.
```

### Error: `pickle.UnpicklingError: invalid load key`
```
Fix: Git corrupted the binary file (common with LFS misconfiguration).
     Run:  git lfs ls-files       (verify pkl listed)
     Or:   git lfs pull           (download LFS files)
     Re-export and push if corruption persists.
```

### Error: App loads but prediction is always "Low Risk"
```
Fix: Scaler was fit on a different feature set.
     Ensure scaler.pkl and feature_names.json are from the same
     export_model.py run (check file timestamps).
```

### Error: Streamlit Cloud build fails at `pip install shap`
```
Fix: Add to requirements.txt:  shap>=0.44.0
     SHAP requires numpy and scipy to be installed first.
     Order in requirements.txt:
       numpy>=1.26.0
       scipy>=1.12.0
       shap>=0.44.0
```

---

## 6 · UI/UX Enhancement Roadmap

| Enhancement | Implementation |
|---|---|
| **Batch CSV upload** | Add `st.file_uploader` → loop `predict()` → download results |
| **SHAP waterfall plot** | `shap.plots.waterfall(explainer(X_input)[0])` in expander |
| **Trend chart** | Store session predictions in `st.session_state` → line chart |
| **PDF report** | `reportlab` or `fpdf2` → download button |
| **A/B model comparison** | Load two models, show side-by-side predictions |
| **Multilingual UI** | `streamlit-multilang` or i18n dict lookup |
| **Patient ID tracking** | `st.text_input("Patient ID")` → prefix CSV rows |
| **Dark/Light toggle** | `st.theme` API (Streamlit ≥ 1.35) |

---

## 7 · Feature Consistency — The Golden Rule

The **only** way to guarantee zero feature-mismatch errors in production:

```
Training                          Inference (app.py)
─────────────────────────────     ─────────────────────────────
1. Load raw data                  1. Collect user inputs
2. add_engineered_features()      2. compute_engineered()   ← same logic
3. StandardScaler.fit()           3. scaler.transform()     ← same scaler
4. Save feature_names.json        4. Load feature_names.json
5. model.fit(X[feature_names])    5. model.predict(X[feature_names])
```

**Never** add features to `app.py` without also adding them to `export_model.py`.
**Never** change feature engineering formulas without retraining and reexporting.

---

*Generated for CardioRisk AI v1.0*

# Deploying the project website

The website (`app/app.py`) needs only what is in the repository: the code in `app/` and `src/`, the small data bundle
in `app/assets/` (aggregates and model outputs, no raw data, no `SK_ID_CURR`) and the LightGBM model in
`app/models/lgbm_model.txt`. The Kaggle data in `data/` is **not** needed and must not be uploaded.

## 1. Check what will be published

From the project root:

```bash
python app/build_assets.py          # only if notebooks were re-run; rebuilds app/assets and app/models
python -m streamlit run app/app.py  # open http://localhost:8501 and click through the pages
```

`.gitignore` already excludes `data/raw/`, `data/processed/`, `kaggle.json` (at any depth), the 57 MB random-forest
model (`models/rf_pipeline.joblib`, recreated only if notebook 03 is re-run), the `_to_delete/` cleanup folder and
local logs. Before the first push, confirm that no
data file is staged:

```bash
git status --ignored     # data/ and kaggle.json must appear under "Ignored files"
```

## 2. Push to GitHub

```bash
git init
git add .
git commit -m "Home Credit default-risk project and website"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Create the empty repository on GitHub first (public, so Streamlit Community Cloud can read it for free). Never commit
`kaggle.json`: it holds your personal Kaggle API key.

## 3. Deploy on Streamlit Community Cloud (free)

1. Sign in at https://share.streamlit.io with your GitHub account.
2. Click **Create app → Deploy a public app from GitHub**.
3. Repository: `<your-username>/<your-repo>`, branch: `main`, **main file path: `app/app.py`**.
4. Open **Advanced settings** and pick the newest Python version offered (the site was built with Python 3.14; it
   needs at least the version required by pandas 3 and numpy 2.5). Community Cloud installs the packages from
   `app/requirements.txt`, which sits next to the entry point.
5. Click **Deploy**. The first build takes a few minutes; later pushes to `main` redeploy automatically.

The theme comes from `.streamlit/config.toml` at the repository root (light and dark variants follow the viewer's
system setting).

## 4. After deployment

- Put the public URL in `README.md` and in the *About* box of the Conclusion page (`app/site_pages/12_conclusion.py`,
  "GitHub repository link") together with the GitHub link.
- If the build fails because a pinned version is not available for the selected Python version, choose another
  Python version in the app settings, or relax that pin in `app/requirements.txt` (for example `pandas>=2.2`) and push
  again; the site uses only standard pandas, numpy, plotly and LightGBM features.
- Resource limits: the bundle is about 4 MB plus a 4 MB model file, well within the free tier.

# TRUTHLENS AI — Future Interface (Build 2.1)

A Streamlit-based student research prototype for news-text classification using TF-IDF and Logistic Regression.

## Files
- `app.py` — application and animated interface
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — dark theme defaults

## Deploy / update
1. Upload `app.py`, `requirements.txt`, and `README.md` to the root of your GitHub repository.
2. Create `.streamlit/config.toml` using the included file.
3. Deploy `app.py` through Streamlit Community Cloud, or commit changes to an existing connected repository.

## Dataset
The app offers a small built-in demo, an optional public ISOT sample fetched from a configured public CSV mirror, or a user-uploaded CSV. The public sample is historical and may be unavailable if its source URL changes. If it cannot load, choose demo mode or upload a properly labelled dataset.

## Important limitation
The model learns text patterns from its labelled training examples. Its output is not proof that a story is true or false, and the class probability is not a verified probability of factual truth. Historical datasets can contain topic/source bias. Use metrics as a teaching demonstration, not a production fact-checking claim.

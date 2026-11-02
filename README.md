# TRUTHLENS AI — Reference Red Glass UI

Streamlit news-intelligence prototype with a black/red interface inspired by the provided four-panel reference, an animated digital globe illustration on Home, Scanner, Live News, Evidence dashboard, Model Lab, and About.

## Deploy
1. Download and extract this ZIP.
2. Replace the files in your GitHub repository with these files, including `.streamlit/config.toml`.
3. Commit changes and let Streamlit Community Cloud rebuild the app.

## Live news and evidence
The app uses Google News RSS to discover recent headlines and links to original reports. The Evidence page searches reports related to a claim. Headlines are not automatically proven true/false by RSS. The existing ML classifier identifies text patterns from its dataset and is not a fact-checking engine.

## Files
- `app.py` — app and UI
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — dark theme

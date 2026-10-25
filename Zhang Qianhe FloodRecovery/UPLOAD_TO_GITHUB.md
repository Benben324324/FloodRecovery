# Upload this project to GitHub

The upload-ready copy excludes the large downloaded raw inputs and local macOS metadata.

## Command-line upload

```bash
cd henan-flood-recovery-github
git init -b main
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/henan-flood-recovery.git
git push -u origin main
```

Replace `YOUR_USERNAME` and the repository name with the GitHub repository you create.

After cloning on another computer:

```bash
python -m pip install -r requirements.txt
python scripts/download_data.py
```

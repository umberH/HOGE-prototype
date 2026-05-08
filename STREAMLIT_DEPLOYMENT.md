# Streamlit Community Cloud Deployment Guide

This guide walks you through deploying the HOGE explainability dashboard to Streamlit Community Cloud.

## 🚀 Quick Start

**Prerequisites:**
- GitHub repository: https://github.com/umberH/HOGE-prototype
- Neo4j Aura instance created and credentials saved
- OpenAI API key

**Deployment Steps:**
1. Fork/have repository on GitHub ✅ (already done)
2. Create Streamlit Community Cloud account
3. Deploy from GitHub
4. Configure secrets
5. Verify deployment

---

## Step 1: Create Streamlit Community Cloud Account

1. Go to https://share.streamlit.io/
2. Click **"Sign up"** or **"Continue with GitHub"**
3. Authorize Streamlit to access your GitHub account
4. You'll be redirected to the Streamlit Cloud dashboard

---

## Step 2: Deploy Your App

### From Streamlit Cloud Dashboard:

1. Click **"New app"** button (top right)
2. Fill in the deployment form:

   **Repository:**
   - GitHub URL: `https://github.com/umberH/HOGE-prototype`
   - Branch: `main` (for production) or `develop` (for staging)

   **Main file path:**
   - `frontend/app.py`

   **App URL (optional):**
   - Custom subdomain: `hoge-prototype` or `phd-experiment1`
   - Will be available at: `https://hoge-prototype.streamlit.app`

3. Click **"Advanced settings"** (optional):
   - Python version: `3.10` (recommended)
   - Add custom packages if needed

4. Click **"Deploy!"**

**Initial deployment takes 2-5 minutes.** ⏳

---

## Step 3: Configure Secrets

Once deployed, you need to add your credentials as secrets.

### Navigate to Secrets:

1. In your app dashboard, click **⋮ (three dots)** → **"Settings"**
2. Click **"Secrets"** in the left sidebar
3. Paste the following TOML configuration:

```toml
# Streamlit Secrets Configuration (TOML format)

# Neo4j Aura credentials
[neo4j]
uri = "neo4j+s://xxxxx.databases.neo4j.io"
user = "neo4j"
password = "your-aura-password-here"
database = "neo4j"

# OpenAI API
[openai]
api_key = "sk-your-openai-api-key-here"
model = "gpt-4"

# Deployment mode
USE_REMOTE_NEO4J = "remote"
```

**⚠️ Important:**
- Replace `uri`, `password`, and `api_key` with your actual credentials
- DO NOT commit secrets to GitHub
- Streamlit will automatically load these as `st.secrets`

### Click **"Save"** and the app will automatically restart.

---

## Step 4: Verify Deployment

### Check App Status:

1. **App should be live** at: `https://your-app-name.streamlit.app`
2. Check for errors in the **Logs** section (Settings → Logs)
3. Test the following features:
   - [ ] Home page loads
   - [ ] Can select a loan application
   - [ ] Explanation generates without errors
   - [ ] Visualizations render correctly
   - [ ] Neo4j knowledge graph queries work

### Common Issues:

**❌ "Module not found" errors:**
- Check that `requirements.txt` includes all dependencies
- Redeploy after fixing

**❌ "Connection refused" to Neo4j:**
- Verify Neo4j Aura secrets are correct
- Check that Aura instance is running (not paused)
- Verify `USE_REMOTE_NEO4J = "remote"` in secrets

**❌ "OpenAI API error":**
- Verify API key in secrets
- Check OpenAI account has credits

---

## Step 5: Continuous Deployment

### Auto-Deploy on Push:

Streamlit Cloud automatically redeploys when you push to the watched branch.

**To deploy new changes:**

```bash
# Make changes locally
git add .
git commit -m "feat: add new feature"
git push origin main

# Streamlit Cloud will auto-deploy in ~2 minutes
```

### Manual Reboot:

If needed, manually restart the app:
1. Go to app dashboard
2. Click **⋮ → "Reboot app"**

---

## 📁 Required Files for Deployment

Ensure these files exist in your repository:

### 1. `frontend/app.py` ✅
Main Streamlit application file.

### 2. `requirements.txt` ✅
All Python dependencies (already exists).

**Key dependencies:**
```txt
streamlit>=1.29.0
neo4j>=5.14.0
openai>=1.6.0
pandas>=2.0.0
plotly>=5.18.0
# ... (see full requirements.txt)
```

### 3. `.streamlit/config.toml` (Optional)
Create this for custom Streamlit configuration:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = true
```

---

## 🔒 Security Best Practices

### ✅ DO:
- Store all credentials in Streamlit secrets
- Use `.env` for local development
- Add `.env` to `.gitignore`
- Use environment variables for sensitive data
- Keep `USE_REMOTE_NEO4J=remote` in Streamlit secrets

### ❌ DON'T:
- Commit `.env` file to GitHub
- Hardcode API keys in code
- Share secrets publicly
- Store passwords in code comments

---

## 🛠️ Local Testing Before Deployment

Test your app locally to ensure it works:

```bash
# 1. Activate virtual environment
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

# 2. Set environment to use local Neo4j
echo "USE_REMOTE_NEO4J=local" >> .env

# 3. Run Streamlit locally
streamlit run frontend/app.py

# 4. Test all features work
# Visit http://localhost:8501
```

---

## 📊 Monitoring & Analytics

### View Logs:

1. Go to app dashboard
2. Click **⋮ → "Logs"**
3. Monitor real-time application logs
4. Check for errors or warnings

### Usage Analytics:

Streamlit Community Cloud provides basic analytics:
- Number of viewers
- App uptime
- Resource usage

**To view:** Go to app dashboard → **"Analytics"** tab

---

## 💰 Resource Limits (Free Tier)

**Streamlit Community Cloud Free Tier includes:**
- ✅ 1 private app
- ✅ Unlimited public apps
- ✅ 1 GB RAM per app
- ✅ 1 CPU core
- ✅ Auto-deployment from GitHub
- ✅ Community support

**Limitations:**
- ⚠️ Apps sleep after 7 days of inactivity
- ⚠️ Limited to 1 concurrent user for resource-intensive apps
- ⚠️ No custom domain (uses `*.streamlit.app`)

**For more resources:** Upgrade to Streamlit Teams or Enterprise.

---

## 🔄 Updating the Deployment

### Update from GitHub:

```bash
# Local changes
git add .
git commit -m "Update: ..."
git push origin main

# Streamlit auto-deploys in ~2 minutes
```

### Update Secrets:

1. Go to app settings
2. Edit secrets in TOML format
3. Click "Save"
4. App automatically restarts

### Rollback:

If deployment fails, revert to previous commit:

```bash
git revert HEAD
git push origin main
```

---

## 🌐 Custom Domain (Optional)

**Note:** Custom domains require Streamlit Teams plan.

If you have Teams:
1. Go to app settings
2. Click "Domains"
3. Add your custom domain
4. Update DNS records as instructed

---

## 🎯 Deployment Checklist

Before deploying to production:

- [ ] All tests pass locally (`pytest tests/`)
- [ ] App runs without errors (`streamlit run frontend/app.py`)
- [ ] Neo4j Aura instance is running and accessible
- [ ] Data loaded into Neo4j Aura
- [ ] OpenAI API key is valid and has credits
- [ ] `requirements.txt` is up to date
- [ ] `.env` is in `.gitignore`
- [ ] No secrets in code or comments
- [ ] README updated with deployment URL
- [ ] GitHub repository is public (or Streamlit has access)

---

## 🆘 Troubleshooting

### Issue: App won't start

**Check:**
1. Logs for error messages
2. `requirements.txt` syntax
3. Python version compatibility
4. Main file path is correct (`frontend/app.py`)

**Fix:**
- Fix errors in code
- Update `requirements.txt`
- Reboot app

### Issue: Neo4j connection timeout

**Check:**
1. Aura instance is running (check Neo4j Console)
2. Secrets are correct (URI, password)
3. `USE_REMOTE_NEO4J = "remote"` in secrets

**Fix:**
- Wake up Aura instance if paused
- Verify credentials
- Check firewall settings

### Issue: Import errors

**Check:**
1. All dependencies in `requirements.txt`
2. Correct import paths (use `backend.src.*` prefix)
3. Python version matches (3.9-3.11)

**Fix:**
- Add missing packages to `requirements.txt`
- Fix import statements
- Rebuild app

---

## 📞 Support

**Streamlit Community:**
- Forum: https://discuss.streamlit.io/
- Docs: https://docs.streamlit.io/
- GitHub: https://github.com/streamlit/streamlit

**Neo4j Aura:**
- Support: https://neo4j.com/cloud/support/
- Docs: https://neo4j.com/docs/aura/

---

## ✅ Post-Deployment

Once deployed successfully:

1. **Share the URL:**
   - `https://your-app-name.streamlit.app`

2. **Set up monitoring:**
   - Check logs daily for first week
   - Monitor Neo4j Aura usage

3. **Update documentation:**
   - Add deployment URL to README
   - Document any custom configuration

4. **Keep Neo4j Aura alive:**
   - GitHub Action pings every 2 days (already configured)
   - Or manually visit app weekly

---

**🎉 Your HOGE explainability dashboard is now live on Streamlit Cloud!**

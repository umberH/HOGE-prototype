# Contributing to PhD Experiment 1

Thank you for contributing! This document provides guidelines for contributing to this project.

## Development Workflow

### 1. Branch Strategy

- `main` - Production-ready code, deployed to Streamlit Cloud
- `develop` - Integration branch for features, merge here first
- `feature/*` - New features (e.g., `feature/add-login`)
- `bugfix/*` or `fix/*` - Bug fixes
- `chore/*` - Maintenance tasks
- `docs/*` - Documentation updates
- `test/*` - Test improvements

### 2. Creating a Feature

1. **Create a branch from develop:**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Write tests first or alongside your code:**
   - Add tests in `tests/` directory
   - Follow existing test structure
   - Ensure tests pass: `pytest tests/ -v`

3. **Make your changes:**
   - Follow Python PEP 8 style guidelines
   - Keep functions focused and documented
   - Update docstrings

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create a PR:**
   ```bash
   git push -u origin feature/your-feature-name
   ```
   - Create PR targeting `develop` branch
   - Fill out the PR template
   - Link related issues

### 3. Commit Message Convention

Follow conventional commits:

- `feat:` - New feature
- `fix:` - Bug fix
- `chore:` - Maintenance tasks
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `style:` - Code style changes (formatting)

Example:
```
feat: add mechanistic interpretability visualization

- Add interactive decision tree visualization
- Implement feature interaction charts
- Add tests for visualization components
```

### 4. Testing Requirements

All PRs must include tests:

- **Unit tests** for new functions/classes
- **Integration tests** for API endpoints
- **End-to-end tests** for user workflows

Run tests before committing:
```bash
pytest tests/ -v --cov=backend
```

### 5. Code Review Process

1. PR is created and automated checks run
2. At least one reviewer approval required
3. All tests must pass
4. No merge conflicts with target branch
5. PR merged to `develop`
6. Periodically, `develop` is merged to `main` for deployment

### 6. Project Structure

```
├── backend/              # Backend API and business logic
│   └── src/
│       ├── api/         # DTOs, services, adapters
│       ├── models/      # ML models
│       ├── explainability/  # Explanation generation
│       └── knowledge_graph/ # Neo4j integration
├── frontend/            # Streamlit dashboard
│   └── app.py
├── tests/              # All test files
│   ├── test_api/
│   └── test_explainability/
├── .development/       # Dev tools, scripts, notebooks
├── .resources/         # Data, models, figures
└── .documentation/     # Documentation (local only)
```

### 7. Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/staq-6/Phd-Experiment1.git
   cd Phd-Experiment1
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create `.env` file:
   ```
   OPENAI_API_KEY=your_key_here
   NEO4J_URI=your_neo4j_uri
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   ```

5. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

### 8. Deployment

- **Streamlit Cloud**: Auto-deploys from `main` branch
- **Neo4j**: Deployed on Aura Free Tier
- **Secrets**: Configured in Streamlit Cloud settings

### 9. Getting Help

- Check existing issues and PRs
- Review documentation in `.documentation/`
- Ask questions in issues or discussions
- Follow developer guidelines in `DEVELOPER_GUIDELINES.md`

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on what is best for the project
- Show empathy towards other contributors

Thank you for contributing! 🎉

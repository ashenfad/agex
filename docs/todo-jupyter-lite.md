# TODO: JupyterLite Demo Hosting

## Goal
Host runnable agex notebooks on GitHub Pages using JupyterLite for zero-friction demos.

## Why JupyterLite?
- **Zero backend**: Runs entirely in browser via WebAssembly
- **GitHub Pages native**: Works perfectly with static hosting
- **Real Python environment**: NumPy, Pandas, Plotly all work
- **Perfect for dummy LLM**: No API calls needed, works with DummyLLMClient
- **User can modify**: Actually runnable, not just static viewing

## Two-Notebook Strategy
1. **`demo.ipynb`**: Basic runtime interoperability (human orchestrated)
2. **`orchestration.ipynb`**: Multi-agent coordination with events introspection

## Implementation Plan

### Repository Structure
```
├── docs/                    # GitHub Pages root
│   ├── index.md            # Landing page with demo links
│   ├── lite/               # JupyterLite deployment (auto-generated)
│   └── _config.yml         # Jekyll config
├── examples/notebook/       # Source notebooks
│   ├── demo.ipynb
│   ├── orchestration.ipynb
│   ├── agents.py           # Maybe - or keep in notebooks for education
│   └── helper.py           # Utilities and canned responses
├── requirements.txt        # JupyterLite Python deps
└── .github/workflows/
    └── deploy.yml          # Auto-build and deploy
```

### GitHub Action Setup
```yaml
name: Build and Deploy JupyterLite
on:
  push:
    branches: [ main ]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install jupyterlite-core
        pip install -r requirements.txt
    - name: Build JupyterLite
      run: |
        jupyter lite build --contents examples/notebook --output-dir docs/lite
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: docs
```

### Landing Page Content
- Hero section explaining agex value prop
- Two clear demo links with descriptions
- "No installation required" messaging
- Brief explanation of what users will learn

### Collapsing Cell Strategy
For `orchestration.ipynb`, use HTML details tags for "setup" sections:
```html
<details>
<summary><strong>🔧 Setup: Recreate Specialists (click to expand)</strong></summary>
<p><em>This recreates the agents from our basic demo.</em></p>
</details>
```

## Benefits for 0.0.1 Release
- **Instant gratification**: Users can try agex immediately
- **No API key friction**: Dummy LLM removes barriers
- **Professional presentation**: Shows framework seriously
- **GitHub discoverability**: Works with social sharing, READMEs
- **Educational**: Progressive complexity across two notebooks

## Requirements.txt for JupyterLite
```txt
agex[all-providers]
pandas
plotly
numpy
ipywidgets
jupyter
```

## Backup Plans
- **Binder**: If JupyterLite has issues
- **Colab badges**: As secondary option
- **Static rendered**: nbviewer as fallback

## Post-0.0.1 Enhancements
- Add live LLM variants of notebooks
- Expand to more complex examples
- Consider JupyterBook for richer documentation

---

*Created: Post-discussion of orchestration notebook strategy and hosting options* 
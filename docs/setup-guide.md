# Documentation Site Setup

This guide explains how to work with the agex documentation site.

## Local Development

### Prerequisites
```bash
pip install -r requirements-docs.txt
```

### Preview the docs locally
```bash
# Start MkDocs development server
mkdocs serve

# Site available at http://localhost:8000
```

### Build static site
```bash
# Build the complete site
mkdocs build --site-dir _site

# Build JupyterLite demo
jupyter lite build --contents examples/notebook --output-dir _site/lite
```

## GitHub Pages Deployment

The site automatically deploys to GitHub Pages via GitHub Actions when you push to `main`.

### Setup Steps:
1. Enable GitHub Pages in repository settings
2. Set source to "GitHub Actions"
3. Update `mkdocs.yml` with your actual repository URLs
4. Push to main branch

### Manual URLs to Update:
In `mkdocs.yml`:
- Replace `USERNAME/agex` with your actual repository
- Replace `USERNAME.github.io/agex` with your actual Pages URL

## Site Structure

```
your-site.github.io/agex/
├── /                    # MkDocs Material documentation
│   ├── api/            # API reference
│   ├── examples/       # Examples overview  
│   ├── demo.md         # JupyterLite integration page
│   └── ...
└── /lite/              # JupyterLite interactive demos
    └── lab/index.html  # Jupyter interface
```

## Customization

- **Styling**: Edit `docs/stylesheets/extra.css`
- **JavaScript**: Edit `docs/javascripts/extra.js`  
- **Theme**: Modify `mkdocs.yml` theme section
- **Navigation**: Update `nav` section in `mkdocs.yml`

## Content Updates

The site pulls from your existing docs:
- Your API docs in `docs/api/` work as-is
- Your examples are showcased in the new examples overview
- Your quick-start and big-picture docs are integrated

## Interactive Demo

The `/lite/` section hosts your JupyterLite notebooks:
- Builds from `examples/notebook/`
- Includes your dummy LLM setup
- Allows users to run and modify code 
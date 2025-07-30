# Contributing to agex

Thank you for your interest in contributing to agex! This guide will help you get started with development and understand our contribution process.

## Development Setup

### Prerequisites

This project uses `pyenv` to manage the Python version and `uv` for package management.

- **Python**: Version specified in `.python-version` file
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (recommended) or pip
- **Optional**: [pyenv](https://github.com/pyenv/pyenv) for Python version management

### Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ashenfad/agex.git
   cd agex
   ```

2. **Set up the Python version:**
   If you have `pyenv` installed, it will automatically pick up the version from the `.python-version` file.

3. **Create a virtual environment and install dependencies:**
   ```bash
   # Using uv (recommended)
   uv venv
   uv pip install -e ".[dev,test,all-providers]"
   
   # Or using standard pip
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev,test,all-providers]"
   ```

4. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

5. **Verify your setup:**
   ```bash
   # Run tests to make sure everything works
   pytest
   
   # Try running an example
   python examples/mathy.py
   ```

### LLM Configuration for Development

```bash
export AGEX_LLM_PROVIDER=openai
export AGEX_LLM_MODEL=gpt-4
export OPENAI_API_KEY=your_key_here
```

## Contribution Guidelines

This project is in early development with informal contribution processes. Detailed guidelines are still being developed.

For now:
- **Bug reports**: Please use [GitHub Issues](https://github.com/ashenfad/agex/issues)
- **Feature ideas**: Feel free to open an issue to discuss
- **Code contributions**: Open an issue first to discuss your approach

More formal contribution guidelines will be established as the project and community grow.

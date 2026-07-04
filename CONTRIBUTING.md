# Contributing to March7

Thanks for your interest in improving March7. This document explains how to set
up a development environment and the conventions we follow.

## Development setup

1. Fork and clone the repository.
2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```

3. Install the project in editable mode with development extras:

   ```bash
   pip install -e ".[dev]"
   ```

4. Copy the example environment file and add your API key:

   ```bash
   cp .env.example .env
   ```

5. Run the app:

   ```bash
   streamlit run src/march7/app.py
   ```

## Project layout

```
src/march7/        Application package (app, config, components, models, utils)
data/              Reference datasets and knowledge base
docs/              User and developer documentation
scripts/           Maintenance and setup utilities
tests/             Test suite (pytest)
deploy/            Container and reverse-proxy configuration
```

## Running tests

```bash
pytest
```

## Pull requests

- Keep changes focused and describe the motivation in the PR description.
- Match the surrounding code style; no emojis in source, comments, or docs.
- Add or update tests when you change behavior.
- Ensure `pytest` passes before requesting review.

## Reporting issues

Please include reproduction steps, expected vs. actual behavior, and your
environment (OS, Python version) when filing an issue.

By contributing, you agree that your contributions will be licensed under the
MIT License.

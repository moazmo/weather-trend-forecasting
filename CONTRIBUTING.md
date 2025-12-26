# Contributing to Weather Trend Forecasting

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Git
- (Optional) NVIDIA GPU with CUDA for faster training

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/weather-trend-forecasting.git
   cd weather-trend-forecasting
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements-dev.txt
   pip install torch --index-url https://download.pytorch.org/whl/cu121  # For GPU
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

## 📝 How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Use the bug report template
3. Include:
   - Python version
   - OS and version
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages/logs

### Suggesting Features

1. Check existing issues/discussions
2. Use the feature request template
3. Explain the use case and benefits

### Submitting Changes

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the code style guidelines
   - Add tests for new functionality
   - Update documentation if needed

3. **Run tests**
   ```bash
   pytest tests/
   ```

4. **Run linting**
   ```bash
   ruff check .
   black --check .
   ```

5. **Commit your changes**
   ```bash
   git commit -m "feat: add amazing feature"
   ```

6. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 🎨 Code Style

- **Formatter**: Black (line length: 100)
- **Linter**: Ruff
- **Type hints**: Required for all functions
- **Docstrings**: Google style

### Example

```python
def predict_temperature(
    country: str,
    date: str,
    model: torch.nn.Module,
) -> dict[str, float]:
    """
    Predict temperature for a given country and date.

    Args:
        country: Name of the country.
        date: Date in YYYY-MM-DD format.
        model: Trained PyTorch model.

    Returns:
        Dictionary with predicted temperature.

    Raises:
        ValueError: If country is not found.
    """
    ...
```

## 📁 Project Structure

```
├── app/          # FastAPI application
├── src/          # Core ML modules
├── tests/        # Test files
├── notebooks/    # Jupyter notebooks (exploration only)
├── data/         # Data files (not in git)
├── models/       # Model files (not in git)
└── docs/         # Documentation
```

## ✅ Commit Message Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Adding/updating tests
- `chore:` Maintenance tasks

## 📋 Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] All tests pass
- [ ] No linting errors

## 🙏 Thank You!

Your contributions make this project better for everyone!

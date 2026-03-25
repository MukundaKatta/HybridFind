# Contributing to HybridFind

Thank you for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/MukundaKatta/HybridFind.git
cd HybridFind
pip install -e .
pip install pytest ruff
```

## Running Tests

```bash
make test
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
make lint
make fmt
```

## Pull Requests

1. Fork the repository and create a feature branch.
2. Write tests for any new functionality.
3. Ensure all tests pass and linting is clean.
4. Open a pull request with a clear description of your changes.

## Reporting Issues

Please use [GitHub Issues](https://github.com/MukundaKatta/HybridFind/issues) and include:

- A minimal reproducible example
- Python version and OS
- Expected vs. actual behavior

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

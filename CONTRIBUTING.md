# Contributing

Thanks for your interest. This is a research repository — contributions that
**strengthen rigor** are especially welcome (additional baselines, statistical
tests, leakage checks, alternative cost models).

## Development setup

```bash
git clone <repo>
cd alpha-validation-lab
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

## Before opening a PR

```bash
make lint      # ruff
make test      # pytest (must stay green)
```

- Keep the **model and feature set constant** unless your PR is explicitly about
  changing them — the controlled comparison is the point.
- Any new study that produces a number must come with: an out-of-sample protocol,
  a baseline to beat, and a significance or selection-bias check.
- New target/strategy code should ship with a unit test (see `tests/`).

## Principles

1. **Try to falsify, not to confirm.**
2. **Race every model against a trivial baseline.**
3. **Report negative results plainly.**
4. **No leakage** — features causal, targets forward, last `H` rows dropped.

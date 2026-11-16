# Contributing

Start with a reproducible issue or a small, focused change. For a substantial new
method or benchmark, discuss the intended protocol before running expensive work.

```bash
python -m pip install -e '.[dev]'
python -m ruff check src tests
python -m ruff format --check src tests
python -m pytest -q
trajectorylab run --config configs/smoke.toml
```

Keep the core tests independent of network access and credentials. Cite algorithms,
preserve upstream licenses if adding third-party code, and distinguish reproductions
from original contributions. AI-assisted changes are welcome with reviewed code,
working reproduction steps and an accurate explanation of the contribution.

New result claims need configuration, source provenance, raw per-run returns,
training seeds, compute budget, and failure/uncertainty discussion. Do not submit
private company data, internal integrations, or unverifiable performance numbers.

Hosted CI configuration is provided; a local pass does not establish that hosted
CI has run. Public dataset downloads are excluded from routine CI.

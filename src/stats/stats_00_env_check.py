# -*- coding: utf-8 -*-
"""Environment self-check (run on the external server): prints Python and key package versions to confirm dependencies are ready."""
import sys

print('python:', sys.version.split()[0])
for m in ['numpy', 'scipy', 'pandas', 'statsmodels', 'pymc', 'arviz']:
    try:
        mod = __import__(m)
        print(f'{m}: {getattr(mod, "__version__", "ok")}')
    except Exception as exc:  # noqa: BLE001
        print(f'{m}: MISSING ({exc})')

"""Keep generated result artifacts in their format-specific directories."""
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parents[1] / 'results'
FORMATS = {'json', 'pdf', 'csv', 'npz', 'html', 'md', 'zip', 'png', 'svg', 'log'}


def by_format(path: Path) -> Path:
    path = Path(path)
    try:
        relative = path.resolve().relative_to(RESULTS_ROOT)
    except ValueError:
        return path
    extension = path.suffix.lstrip('.').lower()
    if extension not in FORMATS:
        return path
    if relative.parts[0] in FORMATS:
        relative = Path(*relative.parts[1:])
    target = RESULTS_ROOT / extension / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    return target

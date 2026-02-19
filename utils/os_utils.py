from pathlib import Path


def find_project_root(current: Path, target_files=('replay_buffer.pkl')):
    """
    project_root = find_project_root(Path.cwd())
    """
    for parent in current.resolve().parents:
        if any((parent / f).exists() for f in target_files):
            return parent
    return None  # fallback to current if not found
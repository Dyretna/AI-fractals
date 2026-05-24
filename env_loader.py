import os
import sys
from pathlib import Path
from typing import Optional


def find_project_root(start_path: Optional[Path] = None) -> Optional[Path]:
    if start_path is None:
        start_path = Path.cwd()

    current = start_path.resolve()

    for _ in range(5):
        env_file = current / ".env"
        if env_file.exists():
            return current

        parent = current.parent
        if parent == current:
            break
        current = parent

    return None


def ensure_env_loaded(project_root: Optional[Path] = None) -> bool:
    if "AI_FRACTALS_ENV_LOADED" in os.environ:
        return True

    if project_root is None:
        project_root = find_project_root()
        if project_root is None and "__file__" in globals():
            script_path = Path(__file__).resolve()
            project_root = find_project_root(script_path.parent)

    if project_root is None:
        if Path(".env").exists():
            project_root = Path.cwd()
        else:
            os.environ["AI_FRACTALS_ENV_LOADED"] = "1"
            return True

    env_path = project_root / ".env"
    if not env_path.exists():
        os.environ["AI_FRACTALS_ENV_LOADED"] = "1"
        return True

    try:
        from dotenv import load_dotenv
    except ImportError:
        os.environ["AI_FRACTALS_ENV_LOADED"] = "1"
        return True

    load_dotenv(env_path, override=True)

    new_env = os.environ.copy()
    new_env["AI_FRACTALS_ENV_LOADED"] = "1"

    os.execve(sys.executable, [sys.executable] + sys.argv, new_env)
    return False

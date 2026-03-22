from pathlib import Path


def get_project_base_directory() -> str:
    return str(Path(__file__).resolve().parents[1])

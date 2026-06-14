import os
import zipfile
from pathlib import Path


REQUIRED_JSONDATA_FILES = (
    "action_space.txt",
    "card_type.json",
    "type_card.json",
)


def ensure_rlcard_doudizhu_jsondata():
    import rlcard

    doudizhu_dir = Path(rlcard.__path__[0]) / "games" / "doudizhu"
    jsondata_dir = doudizhu_dir / "jsondata"
    if all((jsondata_dir / filename).is_file() for filename in REQUIRED_JSONDATA_FILES):
        return

    zip_path = doudizhu_dir / "jsondata.zip"
    if not zip_path.is_file():
        return

    lock_path = doudizhu_dir / ".jsondata.extract.lock"
    with open(lock_path, "w") as lock_file:
        _lock_file(lock_file)
        try:
            if all(
                (jsondata_dir / filename).is_file()
                for filename in REQUIRED_JSONDATA_FILES
            ):
                return

            jsondata_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                for member in zip_ref.infolist():
                    target = doudizhu_dir / member.filename
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zip_ref.open(member, "r") as source:
                        with open(target, "wb") as destination:
                            destination.write(source.read())
        finally:
            _unlock_file(lock_file)


def _lock_file(lock_file):
    if os.name != "posix":
        return

    import fcntl

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)


def _unlock_file(lock_file):
    if os.name != "posix":
        return

    import fcntl

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

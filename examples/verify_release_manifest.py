"""Verify every file listed in RELEASE_MANIFEST.json using the standard library."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    errors = []
    for item in manifest["files"]:
        path = ROOT / item["file"]
        if not path.is_file():
            errors.append(item["file"] + ": missing")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size != item["bytes"] or digest.hexdigest() != item["sha256"]:
            errors.append(item["file"] + ": SHA-256/size mismatch")
    if errors:
        raise SystemExit("\n".join(errors))
    print(json.dumps({"release": manifest["release"], "files_verified": len(manifest["files"]), "all_passed": True}))


if __name__ == "__main__":
    main()

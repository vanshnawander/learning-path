import os
import shutil
from pathlib import Path


TEXT_SOURCE_EXTS = {
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".cu": "cuda",
    ".py": "python",
    ".s": "asm",
    ".S": "asm",
    ".txt": "text",
    ".yml": "yaml",
    ".yaml": "yaml",
}


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> None:
    docs_dir = Path(__file__).resolve().parents[1]
    output_root = docs_dir / "source"

    if output_root.exists():
        shutil.rmtree(output_root)

    for file_path in docs_dir.rglob("*"):
        if not file_path.is_file():
            continue

        if _is_within(file_path, output_root):
            continue

        ext = file_path.suffix
        if ext not in TEXT_SOURCE_EXTS:
            continue

        rel = file_path.relative_to(docs_dir)

        if any(part.startswith(".") for part in rel.parts):
            continue

        language = TEXT_SOURCE_EXTS[ext]
        target_rel = Path("source") / rel.with_suffix(rel.suffix + ".md")
        target_path = docs_dir / target_rel

        content = (
            f"# {rel.as_posix()}\n\n"
            f"```{language}\n"
            f"--8<-- \"{rel.as_posix()}\"\n"
            f"```\n"
        )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")


main()

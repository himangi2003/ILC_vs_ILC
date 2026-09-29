import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Save immediate subfolder names to a text file."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", default="folder_names.txt", type=Path)
    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"Folder does not exist: {args.input}")

    folder_names = sorted(
        folder.name
        for folder in args.input.iterdir()
        if folder.is_dir()
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(name + "\n" for name in folder_names),
        encoding="utf-8",
    )

    print(f"Saved {len(folder_names)} folder names to {args.output}")


if __name__ == "__main__":
    main()
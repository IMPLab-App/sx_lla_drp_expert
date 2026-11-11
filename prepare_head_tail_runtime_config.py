import argparse
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a Head/Tail runtime config.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.source.open("r", encoding="utf-8") as source_file:
        config = yaml.safe_load(source_file)

    config["save_dir"] = args.save_dir
    config["seed"] = args.seed

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(config, output_file, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    main()

"""Audit a private manifest without training a model."""
import argparse
import json
from visionseg.data import audit_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    result = audit_manifest(args.root)
    print(json.dumps({"counts": result["counts"], "manifest_sha256": result["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()

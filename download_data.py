"""Download and validate the official UCI source files used by this project."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "raw"

DATASETS = {
    "dia": {
        "url": "https://archive.ics.uci.edu/static/public/1104/"
        "drug_induced_autoimmunity_prediction.zip",
        "archive": "drug_induced_autoimmunity_prediction.zip",
        "required": {
            "DIA_trainingset_RDKit_descriptors.csv",
            "DIA_testset_RDKit_descriptors.csv",
            "RDKit_ChemDes.xlsx",
        },
    },
    "abalone": {
        "url": "https://archive.ics.uci.edu/static/public/1/abalone.zip",
        "archive": "abalone.zip",
        "required": {"abalone.data", "abalone.names"},
    },
}


def fetch_dataset(name: str, spec: dict[str, object]) -> None:
    destination = DATA_ROOT / name
    destination.mkdir(parents=True, exist_ok=True)
    required = {destination / filename for filename in spec["required"]}

    if all(path.is_file() for path in required):
        print(f"{name}: required files already present")
        return

    archive = destination / str(spec["archive"])
    print(f"{name}: downloading official UCI archive")
    urlretrieve(str(spec["url"]), archive)
    with ZipFile(archive) as zip_file:
        zip_file.extractall(destination)

    missing = sorted(path.name for path in required if not path.is_file())
    if missing:
        raise RuntimeError(f"{name}: archive did not contain {missing}")
    print(f"{name}: downloaded and validated")


def main() -> None:
    for name, spec in DATASETS.items():
        fetch_dataset(name, spec)


if __name__ == "__main__":
    main()

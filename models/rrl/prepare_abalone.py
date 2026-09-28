"""Prepare the upstream data/info format; labels remain numeric for regression."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.eda import load_abalone


def main():
    frame = load_abalone()
    folder = Path(__file__).resolve().parent/'dataset'
    frame.to_csv(folder/'abalone.data', header=False, index=False)
    names = [c.replace(' ', '_') for c in frame.columns]
    (folder/'abalone.info').write_text('\n'.join(
        f'{name} {"discrete" if i == 0 else "continuous"}' for i,name in enumerate(names))+'\nlabel 8\n')
    print('4177 rows; 8 inputs; numeric Rings at column 8')


if __name__ == '__main__':
    main()

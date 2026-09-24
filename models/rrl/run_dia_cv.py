"""Run and collect the five DIA folds for one RRL configuration."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='rrl_original')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--structure', default='5@16')
    parser.add_argument('--learning-rate', type=float, default=0.002)
    args = parser.parse_args()

    result_dir = ROOT / 'results' / 'dia' / args.name
    work_dir = HERE / 'log_folder' / 'dia_cv' / args.name
    result_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in range(5):
        fold_work = work_dir / f'fold_{fold}'
        command = [
            sys.executable, 'experiment.py', '-d', 'dia', '-bs', '32',
            '-s', args.structure, '-e', str(args.epochs), '-lrde', '100',
            '-lr', str(args.learning_rate), '-ki', str(fold), '-wd', '0.001',
            '--nlaf', '--alpha', '0.9', '--beta', '3', '--gamma', '3',
            '--temp', '0.1', '--print_rule', '--device', 'cpu',
            '--output_dir', str(fold_work),
        ]
        subprocess.run(command, cwd=HERE, check=True)
        with (fold_work / 'metrics.json').open() as source:
            rows.append(json.load(source))
        for filename in ('predictions.csv', 'rrl.txt'):
            shutil.copyfile(fold_work / filename, result_dir / f'fold_{fold}_{filename}')

    folds = pd.DataFrame(rows)
    folds.to_csv(result_dir / 'fold_metrics.csv', index=False)
    metrics = [column for column in folds if column != 'fold']
    summary = folds[metrics].agg(['mean', 'std']).T
    summary.to_csv(result_dir / 'summary.csv')
    print(summary.round(4).to_string())


if __name__ == '__main__':
    main()

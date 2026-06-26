#!/usr/bin/env bash
set -e
python scripts/make_synthetic_data.py --out data/demo_medication_packaging.csv
python src/run_experiment.py   --data data/demo_medication_packaging.csv   --target 用药依从性   --features 视觉设计 功能性设计 体验设计 感知体验 日常管理体验   --out outputs/demo_run   --epochs 300   --folds 5

# Dataset note

`processed.cleveland.data` is the processed Cleveland subset from the UCI Heart Disease dataset:
https://archive.ics.uci.edu/dataset/45/heart+disease

The dataset is provided with the original ZIP supplied for this project. This project uses the 14-column Cleveland processed file; the final column `num` is the original target and is converted to binary in `src/heart_disease_classification/train_model.py`. A `?` represents a missing value. See `../../reports/report.md` for the feature definitions and experimental setup.


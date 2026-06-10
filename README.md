# Telugu Word Recogniser

This repository implements a CRNN-based word recogniser for Telugu handwritten words. It was built around the IIIT/CVIT IIIT-INDIC-HW-WORDS-Telugu dataset and provides data loading, training, evaluation, and a small inference utility.

## Key features
- CRNN model (CNN + BiLSTM) with CTC loss for end-to-end word recognition
- Data pipeline that resizes images while preserving aspect ratio, optional augmentation, and robust GT path handling
- Training loop with warmup LR schedule, checkpointing, and evaluation metrics (precision/recall/F1/CER)
- Lightweight inference script that consumes saved checkpoints and writes CSV predictions

## Dataset
Dataset: IIIT-INDIC-HW-WORDS-Telugu
Source: Centre for Visual Information Technology (CVIT), IIIT Hyderabad
License: CC BY 4.0
URL: https://ilocr.iiit.ac.in/dataset/23/

Place the dataset in the repository root (or point `--data` to its location). Expected layout (the project expects `IIT_data_set` by default):

- IIT_data_set/
  - train/
	 - images/
	 - train_gt.txt
  - val/
	 - images/
	 - val_gt.txt
  - test/
	 - images/
	 - test_gt.txt

## Files of interest
- `data_model.py` — dataset, model, training loop, validation and checkpointing
- `infer_check.py` — simple inference utility that loads a checkpoint and writes `check_results.csv`
- `checkpoints/` — saved model checkpoints and evaluation metrics

## Quickstart
1. Create and activate a Python environment (recommended):

	`python -m venv .venv`
	`source .venv/Scripts/activate` (Windows PowerShell: `.\.venv\Scripts\Activate.ps1`)

2. Install dependencies:

	`pip install -r requirements.txt`

3. Train (example):

	`python data_model.py --data IIT_data_set --epochs 20 --batch 32`

	Checkpoints and metrics are written to `checkpoints/` by default.

4. Run inference (example):

	`python infer_check.py --checkpoint checkpoints/best_model.pt --input check --out check_results.csv`

## Notes and troubleshooting
- Character map: `char_map` is built from the training GT file; any characters in validation/test not present in the training charset will be dropped (a warning is printed).
- CTC length constraints: the code enforces a per-sample minimum image width to avoid target>input issues; if you see many warnings about invalid samples, inspect image resizing and GT lengths.
- If resuming training from a checkpoint, scheduler and optimizer states are restored when available to preserve LR trajectory.

## Evaluation
The training script writes an `evaluation_metrics.csv` file inside the configured save folder (default `checkpoints/`) that logs per-epoch metrics including train/val loss, accuracy, precision, recall, F1, CER, and blank frequency.

## License and data
The dataset is provided under CC BY 4.0 by IIIT/CVIT (see link above). This repository code is provided as-is; include any project license you prefer.

## Contact
If you want changes, new features, or help running the project, open an issue or reach out.

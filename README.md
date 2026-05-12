# How many images does YOLO need?

An experiment to find the minimum number of labelled images needed for a YOLO object-detection model to become useful.

## Method

The dataset is split into a fixed 20% test set and an 80% training pool. The model is then trained from scratch on increasingly large subsets of the training pool: **100, 250, and 500 images**. Each subset is nested inside the next (the 100-image set is the same 100 images as the first 100 of the 250-image set), keeping the comparison fair.

After each training run the model is evaluated on the same held-out test set and three metrics are recorded:

| Metric | What it measures |
|---|---|
| **mAP@.5** | Detection accuracy at IoU threshold 0.5 |
| **mAP@.5:.95** | Detection accuracy averaged across IoU thresholds 0.5 → 0.95 (stricter) |
| **Inference speed (ms)** | Time to process one image |

Results are saved to `results.json` after each run, so the experiment can be interrupted and resumed.

## Datasets

Each dataset lives in its own file following a common interface:

```python
def load_my_dataset(output_dir) -> {"images_dir": Path, "labels_dir": Path, "classes": list[str]}
```

| File | Dataset |
|---|---|
| `dataset_pothole.py` | [Kaggle — andrewmvd/pothole-detection](https://www.kaggle.com/datasets/andrewmvd/pothole-detection) (665 real images, Pascal-VOC XML → converted to YOLO) |
| `dataset_synthetic.py` | Procedurally generated rail images with random obstacle bounding boxes |

To switch datasets, change one line in `main.py`:

```python
load_dataset = load_synthetic_rails   # or load_kaggle_pothole
```

## Model

- **Architecture:** YOLOv11n (nano) — the smallest and fastest YOLO11 variant
- **Pre-trained weights:** `yolo11n.pt` (COCO)
- **Epochs:** 25
- **Input size:** 640 × 640
- **Device:** MPS (Apple Silicon)

## Setup

```bash
pip install -r dependencies.txt
```

Add a `.env` file with your Kaggle credentials (only needed for the pothole dataset):

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

## Run

```bash
python main.py
```

Training runs are saved under `runs/detect/size_N/`. A summary table is printed at the end and `results.json` contains the full metrics.

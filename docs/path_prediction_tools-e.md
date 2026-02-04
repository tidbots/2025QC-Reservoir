# Validation Tools

ESN path prediction validation and visualization tools using the ETH pedestrian tracking dataset.

## ETH Dataset

### Data Files

| File | Description |
|------|-------------|
| `tools/data/students001_train.txt` | ETH campus pedestrian data |
| `tools/data/biwi_eth.txt` | BIWI dataset |

### Data Format

```
frame  ped_id  x  y
```
- frame: Frame number
- ped_id: Pedestrian ID
- x, y: Position coordinates (meters)

---

## Batch Evaluation (eth_esn_batch.py)

Evaluate ESN prediction accuracy without GUI.

### Usage

```bash
# Default evaluation (5 pedestrians)
python3 tools/eth_esn_batch.py

# Specify pedestrian IDs
python3 tools/eth_esn_batch.py --ped_ids 399 168 269 177 178

# Adjust parameters
python3 tools/eth_esn_batch.py --n_models 10 --future_horizon 20
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| --data | data/students001_train.txt | Dataset path |
| --ped_ids | Auto-select | Pedestrian IDs to evaluate |
| --n_peds | 5 | Number of pedestrians for auto-select |
| --n_models | 10 | Number of ESN models |
| --future_horizon | 20 | Prediction steps |

---

## Visualization (eth_esn_visualizer.py)

Visualize prediction results.

### Usage

```bash
python3 tools/eth_esn_visualizer.py --ped_ids 399 168 269
```

### Output

- Trajectory and predictions for each pedestrian
- Prediction comparisons at multiple frames
- Statistical summary

---

## V1 vs V2 Comparison (eth_v1_v2_comparison.py)

Compare V1 (original ESN) with V2 (Kalman hybrid).

### Usage

```bash
python3 tools/eth_v1_v2_comparison.py --ped_ids 399 168 269 177 178
```

### Output

- Error comparison by pedestrian
- Improvement rate visualization
- Overall summary

---

## Original Script (person_tracking_esn_fx.py)

ESN evaluation script created by colleague. Includes animation feature.

### Usage

```bash
# Save MP4 animation
python3 tools/person_tracking_esn_fx.py --save_mp4 --ped_ids 399
```

---

## PDF Conversion (md2pdf.py)

Convert Markdown documents to PDF.

### Usage

```bash
python3 tools/md2pdf.py docs/path_prediction_eth_evaluation.md
```

---

## File Structure

```
tools/
├── data/
│   ├── students001_train.txt    # ETH dataset
│   └── biwi_eth.txt             # BIWI dataset
├── eth_esn_batch.py             # Batch evaluation
├── eth_esn_visualizer.py        # Visualization
├── eth_v1_v2_comparison.py      # V1 vs V2 comparison
├── person_tracking_esn_fx.py    # Original script
└── md2pdf.py                    # PDF conversion
```

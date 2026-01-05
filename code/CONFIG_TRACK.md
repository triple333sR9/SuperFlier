# Configuration Tracking

## Selected Model
**efficientnet_b3** | AdamW | unfreeze all | lr=0.0005 | medium augmentation | F1=0.9643 | AUC=0.9937

---

## Round 1: Baseline Testing

### Models Tested
resnet50, efficientnet_b3, densenet121, convnext_small, resnext50

### Fixed Parameters
| Parameter | Value |
|-----------|-------|
| Image size | 384 |
| Batch size | 16 |
| Epochs | 15 |
| Learning rate | 0.001 |
| Dropout | 0.3 |
| Weight decay | 0.01 |
| Optimizer | AdamW |
| Freeze strategy | unfreeze last block |
| Augmentation | medium |
| Weighted loss | Yes |

---

## Round 2: Grid Search (144 combinations)

### Parameter Space
| Parameter | Values |
|-----------|--------|
| Models | efficientnet_b3, densenet121, resnet50 |
| Optimizers | AdamW, Adam, SGD, RMSprop |
| Freeze strategy | freeze, last_block, all |
| Learning rate | 0.0005, 0.001 |
| Augmentation | medium, heavy |

### Fixed Parameters
| Parameter | Value |
|-----------|-------|
| Image size | 384 |
| Batch size | 32 |
| Epochs | 15 |
| Dropout | 0.2 |
| Weight decay | 0.01 |

### Augmentation Definitions
- **Medium**: horizontal flip, rotation(15°), color jitter(0.2)
- **Heavy**: horizontal flip, rotation(30°), color jitter(0.4), random erasing(0.2)

### Failed Configurations
- Heavy augmentation with SGD failed due to `RandomErasing` placed before `ToTensor()` (4 configurations affected)
- Fixed by moving `RandomErasing` after `ToTensor()` and `Normalize()`

---

## Round 3: Top 10 Refinement
Re-ran top 10 configurations from Round 2 for validation.
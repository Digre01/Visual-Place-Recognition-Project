## 1. Global Retrieval Baselines 

| Method | Dataset | Metric | R@1 | R@5 | R@10 | R@20 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NetVLAD (512x512)** | SF-XS Test | L2 / Dot Product | 42.2 | 53.7 | 60.8 | 65.3 |
| **NetVLAD** | Tokyo-XS | L2 / Dot Product | 54.0 | 64.4 | 69.5 | 74.6 |
| **NetVLAD (512x512)** | SVOX Night | L2 / Dot Product | 8.5 | 18.2 | 22.7 | 29.5 |
| **NetVLAD (512x512)** | SVOX Sun | L2 / Dot Product | 37.1 | 54.4 | 61.9 | 69.0 |
| **CosPlace (512x512)** | SF-XS Test | L2 / Dot Product | 63.1 | 74.8 | 78.6 | 81.4 |
| **CosPlace (512x512)** | Tokyo-XS | L2 / Dot Product | 65.1 | 79.7 | 86.0 | 89.5 |
| **CosPlace (512x512)** | SVOX Night | L2 / Dot Product | 33.3 | 51.5 | 59.1 | 67.7 |
| **CosPlace (512x512)** | SVOX Sun | L2 / Dot Product | 62.3 | 78.5 | 84.5 | 88.8 |
| **MixVPR (320x320)** | SF-XS Test | L2 / Dot Product | 70.2 | 79.0 | 81.3 | 83.9 |
| **MixVPR (320x320)** | Tokyo-XS | Dot Product | 78.1 | 89.5 | 92.4 | 93.7 |
| **MixVPR (320x320)** | SVOX Night | Dot Product | 62.9 | 79.8 | 84.1 | 88.0 |
| **MixVPR (320x320)** | SVOX Sun | Dot Product | 85.4 | 93.0 | 94.7 | 95.9 |
| **MegaLoc (320x320)** | SF-XS Test | L2 / Dot Product | 86.4 | 89.9 | 90.7 | 91.4 |
| **MegaLoc (320x320)** | Tokyo-XS | L2 | 94.6 | 97.8 | 98.4 | 99.0 |
| **MegaLoc (320x320)** | SVOX Night | L2 | 90.4 | 98.1 | 98.8 | 99.1 |
| **MegaLoc (320x320)** | SVOX Sun | L2 | 96.4 | 99.3 | 99.5 | 99.6 |

---

## 2. Refinement Results (Two-Stage Pipeline)

### Dataset: SF-XS Test
| Method | Refinement | R@1 | R@5 | R@10 | R@20 | Time/Query (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NetVLAD** | SuperGlue | 61.1 | 64.4 | 64.9 | 65.3 | 1.097 |
| **NetVLAD** | LoFTR | 61.5 | 63.9 | 64.6 | 65.3 | 2.059 |
| **NetVLAD** | SuperPoint-LG | 62.7 | 64.5 | 64.9 | 65.3 | 2.696 |
| **CosPlace** | SuperGlue | 76.7 | 79.4 | 80.5 | 81.4 | 1.124 |
| **CosPlace** | LoFTR | 77.4 | 79.7 | 80.6 | 81.4 | 2.095 |
| **CosPlace** | SuperPoint-LG | 77.7 | 80.5 | 80.9 | 81.4 | 2.731 |
| **MixVPR** | SuperGlue | 79.2 | 82.0 | 83.2 | 83.6 | 1.048 |
| **MixVPR** | LoFTR | 79.6 | 82.7 | 83.4 | 83.6 | 2.083 |
| **MixVPR** | SuperPoint-LG | 80.5 | 82.5 | 83.2 | 83.6 | 2.720 |
| **MegaLoc** | SuperGlue | 85.8 | 89.9 | 90.7 | 91.5 | 1.107 |
| **MegaLoc** | LoFTR | 86.5 | 89.7 | 90.8 | 91.5 | 2.092 |
| **MegaLoc** | SuperPoint-LG | 86.7 | 90.7 | 91.3 | 91.5 | 2.722 |

### Dataset: Tokyo-XS
| Method | Refinement | R@1 | R@5 | R@10 | R@20 | Time/Query (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NetVLAD** | SuperGlue | 67.0 | 71.1 | 72.7 | 74.6 | 1.108 |
| **NetVLAD** | LoFTR | 68.9 | 70.8 | 73.0 | 74.6 | 2.051 |
| **NetVLAD** | SuperPoint-LG | 68.9 | 71.4 | 73.0 | 74.6 | 2.692 |
| **CosPlace** | SuperGlue | 82.5 | 86.3 | 88.6 | 89.5 | 1.146 |
| **CosPlace** | LoFTR | 84.8 | 87.9 | 88.6 | 89.5 | 2.089 |
| **CosPlace** | SuperPoint-LG | 82.9 | 86.3 | 87.9 | 89.5 | 2.746 |
| **MixVPR** | SuperGlue | 86.7 | 91.7 | 93.3 | 93.7 | 1.098 |
| **MixVPR** | LoFTR | 89.8 | 92.4 | 93.3 | 93.7 | 2.121 |
| **MixVPR** | SuperPoint-LG | 88.6 | 92.4 | 92.7 | 93.7 | 2.806 |
| **MegaLoc** | SuperGlue | 93.0 | 98.1 | 98.4 | 99.0 | 1.156 |
| **MegaLoc** | LoFTR | 94.3 | 97.8 | 98.7 | 99.0 | 2.076 |
| **MegaLoc** | SuperPoint-LG | 94.3 | 98.4 | 98.7 | 99.0 | 2.721 |

### Dataset: SVOX Night
| Method | Refinement | R@1 | R@5 | R@10 | R@20 | Time/Query (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NetVLAD** | SuperGlue | 24.8 | 27.1 | 28.4 | 29.5 | 1.131 |
| **NetVLAD** | LoFTR | 25.3 | 27.8 | 28.7 | 29.5 | 2.053 |
| **NetVLAD** | SuperPoint-LG | 24.5 | 27.2 | 28.3 | 29.5 | 2.706 |
| **CosPlace** | SuperGlue | 59.4 | 64.5 | 65.9 | 67.6 | 1.158 |
| **CosPlace** | LoFTR | 61.4 | 65.4 | 66.2 | 67.6 | 2.086 |
| **CosPlace** | SuperPoint-LG | 60.5 | 65.1 | 66.3 | 67.6 | 2.736 |
| **MixVPR** | SuperGlue | 81.9 | 86.4 | 87.5 | 88.0 | 1.097 |
| **MixVPR** | LoFTR | 82.5 | 86.8 | 87.6 | 88.0 | 2.153 |
| **MixVPR** | SuperPoint-LG | 82.4 | 86.3 | 87.1 | 88.0 | 2.834 |
| **MegaLoc** | SuperGlue | 90.5 | 97.6 | 98.7 | 99.3 | 1.182 |
| **MegaLoc** | LoFTR | 92.6 | 98.4 | 99.0 | 99.3 | 2.080 |
| **MegaLoc** | SuperPoint-LG | 91.7 | 97.4 | 98.9 | 99.3 | 2.740 |

### Dataset: SVOX Sun
| Method | Refinement | R@1 | R@5 | R@10 | R@20 | Time/Query (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NetVLAD** | SuperGlue | 64.3 | 66.3 | 67.4 | 69.0 | 1.132 |
| **NetVLAD** | LoFTR | 64.6 | 66.9 | 67.8 | 69.0 | 2.068 |
| **NetVLAD** | SuperPoint-LG | 65.0 | 66.5 | 67.7 | 69.0 | 2.712 |
| **CosPlace** | SuperGlue | 81.3 | 85.9 | 87.4 | 88.8 | 1.175 |
| **CosPlace** | LoFTR | 85.1 | 87.1 | 87.9 | 88.8 | 2.107 |
| **CosPlace** | SuperPoint-LG | 83.8 | 86.7 | 87.9 | 88.8 | 2.756 |
| **MixVPR** | SuperGlue | 89.8 | 94.0 | 95.2 | 95.9 | 1.224 |
| **MixVPR** | LoFTR | 93.4 | 95.0 | 95.3 | 95.9 | 2.168 |
| **MixVPR** | SuperPoint-LG | 91.5 | 95.0 | 95.4 | 95.9 | 2.794 |
| **MegaLoc** | SuperGlue | 93.8 | 98.7 | 99.4 | 99.6 | 1.175 |
| **MegaLoc** | LoFTR | 97.3 | 99.3 | 99.5 | 99.6 | 2.101 |
| **MegaLoc** | SuperPoint-LG | 96.1 | 99.1 | 99.4 | 99.6 | 2.739 |

---

### Computation & Matching Times

#### Average Total Time (Global + Refinement)
* **NetVLAD + SuperGlue:** ~1425s
* **NetVLAD + LoFTR:** ~3550s
* **NetVLAD + LightGlue:** ~1120s
* **CosPlace + SuperGlue:** ~1280s
* **CosPlace + LoFTR:** ~3240s
* **CosPlace + LightGlue:** ~920s
* **MixVPR + SuperGlue:** ~1220s
* **MixVPR + LoFTR:** ~3190s
* **MixVPR + LightGlue:** ~890s

### Retrieval Time Per Query (ms)
| Method | SF-XS | Tokyo | Sun | Night | Weighted Avg |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NetVLAD** | 407 ms | 596 ms | 386 ms | - | 354.5 ms |
| **CosPlace** | 207 ms | 212 ms | 105 ms | - | 148.9 ms |
| **MixVPR** | 155 ms | 226 ms | 120 ms | 123 ms | 144.0 ms |
| **MegaLoc** | 1154 ms | 879 ms | 1344 ms | 1404 ms | - |

## 3. Extension 6.1: Threshold Analysis (R@1)

| Method | Refinement | Dataset | Baseline | Th=10 | Th=20 | Th=30 | Th=45 | Th=65 | Th=80 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CosPlace** | LoFTR | SF-XS | 63.1% | 64.4% | 69.7% | 72.9% | 74.9% | 76.4% | 76.7% |
| **CosPlace** | SP-LG | SF-XS | 63.1% | 72.4% | 75.6% | 76.6% | 77.4% | 77.4% | 77.5% |
| **CosPlace** | LoFTR | Tokyo-XS | 65.1% | 73.0% | 83.8% | 84.4% | 84.8% | 84.8% | 84.8% |
| **CosPlace** | SP-LG | Tokyo-XS | 65.1% | 80.3% | 83.2% | 83.2% | 83.2% | 83.2% | 83.2% |
| **CosPlace** | LoFTR | SVOX Sun | 62.3% | 66.5% | 79.5% | 83.4% | 84.2% | 84.5% | 84.7% |
| **CosPlace** | SP-LG | SVOX Sun | 62.3% | 71.8% | 79.5% | 81.3% | 82.8% | 83.3% | 83.6% |
| **CosPlace** | LoFTR | SVOX Night | 33.3% | 45.8% | 59.3% | 61.1% | 61.2% | 61.2% | 61.3% |
| **CosPlace** | SP-LG | SVOX Night | 33.3% | 51.0% | 59.3% | 59.9% | 60.0% | 60.0% | 60.2% |
| **NetVLAD** | LoFTR | SF-XS | 42.2% | 44.5% | 55.0% | 59.4% | 60.9% | 61.2% | 61.5% |
| **NetVLAD** | SP-LG | SF-XS | 42.2% | 56.5% | 61.3% | 62.3% | 62.7% | 62.8% | 62.8% |
| **NetVLAD** | LoFTR | Tokyo-XS | 54.0% | 59.4% | 67.3% | 67.9% | 68.9% | 68.9% | 68.9% |
| **NetVLAD** | SP-LG | Tokyo-XS | 54.0% | 67.3% | 67.9% | 68.6% | 68.9% | 68.9% | 68.9% |
| **NetVLAD** | LoFTR | SVOX Sun | 37.1% | 47.2% | 60.7% | 62.8% | 63.8% | 63.9% | 64.2% |
| **NetVLAD** | SP-LG | SVOX Sun | 37.1% | 52.8% | 58.0% | 63.5% | 64.5% | 64.9% | 64.9% |
| **NetVLAD** | LoFTR | SVOX Night | 8.5% | 18.5% | 24.5% | 25.0% | 25.2% | 25.2% | 25.2% |
| **NetVLAD** | SP-LG | SVOX Night | 8.5% | 21.8% | 24.7% | 24.5% | 24.7% | 24.7% | 24.7% |

---

## Extension 6.1: Cost Savings Analysis

| Method | Refinement | Dataset | Th=10 | Th=20 | Th=30 | Th=45 | Th=65 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CosPlace** | LoFTR | SF-XS | 91.0% | 74.5% | 67.1% | 59.6% | 53.6% |
| **CosPlace** | SP-LG | SF-XS | 71.3% | 62.8% | 59.5% | 53.6% | 45.1% |
| **CosPlace** | LoFTR | Tokyo-XS | 80.5% | 60.6% | 58.5% | 54.6% | 50.1% |
| **CosPlace** | SP-LG | Tokyo-XS | 66.1% | 57.3% | 54.6% | 48.0% | 38.0% |
| **CosPlace** | LoFTR | SVOX Sun | 85.7% | 64.9% | 59.3% | 53.1% | 50.6% |
| **CosPlace** | SP-LG | SVOX Sun | 75.4% | 63.5% | 59.7% | 55.0% | 47.2% |
| **CosPlace** | LoFTR | SVOX Night | 63.4% | 31.7% | 28.4% | 26.1% | 24.1% |
| **CosPlace** | SP-LG | SVOX Night | 48.5% | 32.4% | 28.6% | 25.5% | 21.2% |
| **NetVLAD** | LoFTR | SF-XS | 84.0% | 54.4% | 45.0% | 39.5% | 35.3% |
| **NetVLAD** | SP-LG | SF-XS | 53.5% | 41.9% | 39.2% | 35.4% | 29.4% |
| **NetVLAD** | LoFTR | Tokyo-XS | 76.3% | 51.0% | 49.2% | 46.1% | 43.4% |
| **NetVLAD** | SP-LG | Tokyo-XS | 56.4% | 49.5% | 46.1% | 41.6% | 34.4% |

---

## 4. Extension 6.1: Logistic Regression Analysis

### Performance (R@1 using Logistic RegressorReg)
| Model | Train Set | Pred. Th | Tokyo-XS | SF-XS | SVOX Sun | SVOX Night |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CosPlace + LoFTR** | Svox Night | 30.86 | 84.4 | 73.2 | 83.6 | 61.1 |
| **CosPlace + LoFTR** | Svox Sun | 30.92 | 84.4 | 73.2 | 83.6 | 61.1 |
| **CosPlace + SP-LG** | Svox Night | 21.79 | 83.2 | 76.0 | 81.0 | 59.7 |
| **CosPlace + SP-LG** | Svox Sun | 21.96 | 83.2 | 76.4 | 80.4 | 59.4 |
| **NetVlad + LoFTR** | Svox Night | 41.89 | 68.6 | 60.6 | 63.5 | 25.2 |
| **NetVlad + LoFTR** | Svox Sun | 41.78 | 68.6 | 60.6 | 63.5 | 25.2 |
| **NetVlad + SP-LG** | Svox Night | 35.85 | 68.6 | 62.5 | 63.5 | 24.7 |
| **NetVlad + SP-LG** | Svox Sun | 35.71 | 68.6 | 62.5 | 63.5 | 63.5 |

### Cost Savings (Time Saved % with using Logistic Regressor)
| Model | Train Set | Pred. Th | Tokyo-XS | SF-XS | SVOX Sun | SVOX Night |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CosPlace + LoFTR** | Svox Night | 30.86 | 58.5% | 66.5% | 59.1% | 28.3% |
| **CosPlace + SP-LG** | Svox Night | 21.79 | 56.7% | 61.8% | 61.5% | 30.1% |
| **NetVlad + LoFTR** | Svox Night | 41.89 | 46.8% | 40.2% | 34.7% | 6.9% |
| **NetVlad + SP-LG** | Svox Night | 35.85 | 44.9% | 38.0% | 35.3% | 7.6% |

---


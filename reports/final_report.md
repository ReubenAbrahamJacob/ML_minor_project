# Multimodal, uncertainty-aware predictive maintenance for rolling bearings

*Minor project, NITK. All numbers below are generated from `reports/tables/*.csv` by `scripts/make_report.py`; run configs, seeds and git state are in `reports/runs/*.json`. Costs in the decision layer are **assumptions**, not measurements.*

## 1. Problem
For a rolling-element bearing we ask three questions: (1) which health state is it in (healthy / early / severe), (2) how much life remains, with a calibrated interval, and (3) should we maintain now, schedule, or continue. We test whether fusing signal-derived time-frequency images with engineered features helps (RQ1), whether intervals reach nominal coverage (RQ2), how much is lost across datasets (RQ3) and whether the same pipeline runs on a turbofan (RQ4).

## 2. Mechanics background (course notes)
Rolling bearings fail by contact fatigue (spalling) even when well lubricated; identical bearings show different lives, hence a statistical treatment: rating life L10 is the life 90% of a group reaches, L50 = 5 L10, L = (C/P)^3 for ball bearings and Lh = 10^6 L10 / (60 N). Reliability follows ln(1/R) = (L/a)^b with a = 6.84, b = 1.17, which motivates the Weibull baseline and uncertainty quantification. Nominal life scales as load^-3 and 1/speed, so operating condition is always a covariate or a split variable. Petroff's equation, the Sommerfeld number and McKee's analysis belong to **journal** bearings and are background only; no feature uses them.
**Two source notes** (details in `docs/mechanics_notes.md`): the slide prints 8.85 as the L/L90 coefficient, but a = 6.84 and b = 1.17 give 6.84 (unit-tested); and CLAUDE.md's BSF formula is half of CWRU's published ball-defect value (2 x BSF matches it).

## 3. Data
| dataset | asset | condition | snapshots | lifetime_s | windows_used | channels | sampling_hz | rpm | load |
|---|---|---|---|---|---|---|---|---|---|
| femto | Bearing1_1 | 1.0 | 2803 | 28030.0 | 300 | 2 | 25600.0 | 1800.0 | 4000.0 |
| femto | Bearing1_2 | 1.0 | 871 | 8710.0 | 300 | 2 | 25600.0 | 1800.0 | 4000.0 |
| femto | Bearing1_3 | 1.0 | 2375 | 23750.0 | 300 | 2 | 25600.0 | 1800.0 | 4000.0 |
| femto | Bearing1_4 | 1.0 | 1428 | 14280.0 | 300 | 2 | 25600.0 | 1800.0 | 4000.0 |
| femto | Bearing1_5 | 1.0 | 2463 | 24630.0 | 300 | 2 | 25600.0 | 1800.0 | 4000.0 |
| femto | Bearing1_6 | 1.0 | 2448 | 24480.0 | 300 | 2 | 25600.0 | 1800.0 | 4000.0 |
| femto | Bearing1_7 | 1.0 | 2259 | 22590.0 | 300 | 2 | 25600.0 | 1800.0 | 4000.0 |
| femto | Bearing2_1 | 2.0 | 911 | 9110.0 | 300 | 2 | 25600.0 | 1650.0 | 4200.0 |
| femto | Bearing2_2 | 2.0 | 797 | 7970.0 | 300 | 2 | 25600.0 | 1650.0 | 4200.0 |
| femto | Bearing2_3 | 2.0 | 1955 | 19550.0 | 300 | 2 | 25600.0 | 1650.0 | 4200.0 |
| femto | Bearing2_4 | 2.0 | 751 | 7510.0 | 300 | 2 | 25600.0 | 1650.0 | 4200.0 |
| femto | Bearing2_5 | 2.0 | 2311 | 23110.0 | 300 | 2 | 25600.0 | 1650.0 | 4200.0 |
| femto | Bearing2_6 | 2.0 | 701 | 7010.0 | 300 | 2 | 25600.0 | 1650.0 | 4200.0 |
| femto | Bearing2_7 | 2.0 | 230 | 2300.0 | 230 | 2 | 25600.0 | 1650.0 | 4200.0 |
| femto | Bearing3_1 | 3.0 | 515 | 5150.0 | 300 | 2 | 25600.0 | 1500.0 | 5000.0 |
| femto | Bearing3_2 | 3.0 | 1637 | 16370.0 | 300 | 2 | 25600.0 | 1500.0 | 5000.0 |
| femto | Bearing3_3 | 3.0 | 434 | 4340.0 | 300 | 2 | 25600.0 | 1500.0 | 5000.0 |
| xjtu | Bearing1_1 | 1.0 | 123 | 7380.0 | 123 | 2 | 25600.0 | 2100.0 | 12.0 |
| xjtu | Bearing1_2 | 1.0 | 161 | 9660.0 | 161 | 2 | 25600.0 | 2100.0 | 12.0 |
| xjtu | Bearing1_3 | 1.0 | 158 | 9480.0 | 158 | 2 | 25600.0 | 2100.0 | 12.0 |
| xjtu | Bearing1_4 | 1.0 | 122 | 7320.0 | 122 | 2 | 25600.0 | 2100.0 | 12.0 |
| xjtu | Bearing1_5 | 1.0 | 52 | 3120.0 | 52 | 2 | 25600.0 | 2100.0 | 12.0 |
| xjtu | Bearing2_1 | 2.0 | 491 | 29460.0 | 300 | 2 | 25600.0 | 2250.0 | 11.0 |
| xjtu | Bearing2_2 | 2.0 | 161 | 9660.0 | 161 | 2 | 25600.0 | 2250.0 | 11.0 |
| xjtu | Bearing2_3 | 2.0 | 533 | 31980.0 | 300 | 2 | 25600.0 | 2250.0 | 11.0 |
| xjtu | Bearing2_4 | 2.0 | 42 | 2520.0 | 42 | 2 | 25600.0 | 2250.0 | 11.0 |
| xjtu | Bearing2_5 | 2.0 | 339 | 20340.0 | 300 | 2 | 25600.0 | 2250.0 | 11.0 |
| xjtu | Bearing3_1 | 3.0 | 2538 | 152280.0 | 300 | 2 | 25600.0 | 2400.0 | 10.0 |
| xjtu | Bearing3_2 | 3.0 | 2496 | 149760.0 | 300 | 2 | 25600.0 | 2400.0 | 10.0 |
| xjtu | Bearing3_3 | 3.0 | 371 | 22260.0 | 300 | 2 | 25600.0 | 2400.0 | 10.0 |
| xjtu | Bearing3_4 | 3.0 | 1515 | 90900.0 | 300 | 2 | 25600.0 | 2400.0 | 10.0 |
| xjtu | Bearing3_5 | 3.0 | 114 | 6840.0 | 114 | 2 | 25600.0 | 2400.0 | 10.0 |
| cwru | Normal_3 | 3.0 | 60 |  | 60 | 1 | 12000.0 | 1730.0 | 3.0 |
| cwru | IR007_0 | 0.0 | 59 |  | 59 | 1 | 12000.0 | 1797.0 | 0.0 |
| cwru | IR007_1 | 1.0 | 59 |  | 59 | 1 | 12000.0 | 1772.0 | 1.0 |
| cwru | IR007_2 | 2.0 | 59 |  | 59 | 1 | 12000.0 | 1750.0 | 2.0 |
| cwru | IR007_3 | 3.0 | 60 |  | 60 | 1 | 12000.0 | 1730.0 | 3.0 |
| cwru | B007_0 | 0.0 | 59 |  | 59 | 1 | 12000.0 | 1797.0 | 0.0 |
| cwru | B007_1 | 1.0 | 59 |  | 59 | 1 | 12000.0 | 1772.0 | 1.0 |
| cwru | B007_2 | 2.0 | 59 |  | 59 | 1 | 12000.0 | 1750.0 | 2.0 |
| cwru | B007_3 | 3.0 | 59 |  | 59 | 1 | 12000.0 | 1730.0 | 3.0 |
| cwru | OR007@6_0 | 0.0 | 59 |  | 59 | 1 | 12000.0 | 1797.0 | 0.0 |
| cwru | OR007@6_1 | 1.0 | 59 |  | 59 | 1 | 12000.0 | 1772.0 | 1.0 |
| cwru | OR007@6_2 | 2.0 | 59 |  | 59 | 1 | 12000.0 | 1750.0 | 2.0 |
| cwru | OR007@6_3 | 3.0 | 59 |  | 59 | 1 | 12000.0 | 1730.0 | 3.0 |
| cwru | OR007@3_0 | 0.0 | 59 |  | 59 | 1 | 12000.0 | 1797.0 | 0.0 |
| cwru | OR007@3_1 | 1.0 | 59 |  | 59 | 1 | 12000.0 | 1772.0 | 1.0 |
| cwru | OR007@3_2 | 2.0 | 59 |  | 59 | 1 | 12000.0 | 1750.0 | 2.0 |
| cwru | OR007@3_3 | 3.0 | 59 |  | 59 | 1 | 12000.0 | 1730.0 | 3.0 |
| cwru | OR007@12_0 | 0.0 | 59 |  | 59 | 1 | 12000.0 | 1797.0 | 0.0 |
| cwru | OR007@12_1 | 1.0 | 59 |  | 59 | 1 | 12000.0 | 1772.0 | 1.0 |
| cwru | OR007@12_2 | 2.0 | 59 |  | 59 | 1 | 12000.0 | 1750.0 | 2.0 |
| cwru | OR007@12_3 | 3.0 | 59 |  | 59 | 1 | 12000.0 | 1730.0 | 3.0 |
| cwru | IR014_0 | 0.0 | 59 |  | 59 | 1 | 12000.0 | 1797.0 | 0.0 |
| cwru | IR014_1 | 1.0 | 59 |  | 59 | 1 | 12000.0 | 1772.0 | 1.0 |
| cwru | IR014_2 | 2.0 | 59 |  | 59 | 1 | 12000.0 | 1750.0 | 2.0 |
| cwru | IR014_3 | 3.0 | 59 |  | 59 | 1 | 12000.0 | 1730.0 | 3.0 |
| cwru | B014_0 | 0.0 | 59 |  | 59 | 1 | 12000.0 | 1797.0 | 0.0 |
| cwru | B014_1 | 1.0 | 59 |  | 59 | 1 | 12000.0 | 1772.0 | 1.0 |
| cwru | B014_2 | 2.0 | 59 |  | 59 | 1 | 12000.0 | 1750.0 | 2.0 |
| cwru | B014_3 | 3.0 | 59 |  | 59 | 1 | 12000.0 | 1730.0 | 3.0 |
| cwru | OR014@6_0 | 0.0 | 59 |  | 59 | 1 | 12000.0 | 1797.0 | 0.0 |
| cwru | OR014@6_1 | 1.0 | 59 |  | 59 | 1 | 12000.0 | 1772.0 | 1.0 |
| cwru | OR014@6_2 | 2.0 | 59 |  | 59 | 1 | 12000.0 | 1750.0 | 2.0 |
| cwru | OR014@6_3 | 3.0 | 59 |  | 59 | 1 | 12000.0 | 1730.0 | 3.0 |
| cwru | IR021_0 | 0.0 | 59 |  | 59 | 1 | 12000.0 | 1797.0 | 0.0 |
| cwru | IR021_1 | 1.0 | 59 |  | 59 | 1 | 12000.0 | 1772.0 | 1.0 |
| cwru | IR021_2 | 2.0 | 59 |  | 59 | 1 | 12000.0 | 1750.0 | 2.0 |
| cwru | IR021_3 | 3.0 | 59 |  | 59 | 1 | 12000.0 | 1730.0 | 3.0 |
| cwru | B021_0 | 0.0 | 59 |  | 59 | 1 | 12000.0 | 1797.0 | 0.0 |
| cwru | B021_1 | 1.0 | 59 |  | 59 | 1 | 12000.0 | 1772.0 | 1.0 |
| cwru | B021_2 | 2.0 | 59 |  | 59 | 1 | 12000.0 | 1750.0 | 2.0 |
| cwru | B021_3 | 3.0 | 59 |  | 59 | 1 | 12000.0 | 1730.0 | 3.0 |
| cwru | OR021@6_0 | 0.0 | 59 |  | 59 | 1 | 12000.0 | 1797.0 | 0.0 |
| cwru | OR021@6_1 | 1.0 | 59 |  | 59 | 1 | 12000.0 | 1772.0 | 1.0 |
| cwru | OR021@6_2 | 2.0 | 59 |  | 59 | 1 | 12000.0 | 1750.0 | 2.0 |
| cwru | OR021@6_3 | 3.0 | 59 |  | 59 | 1 | 12000.0 | 1730.0 | 3.0 |
| cwru | OR021@3_0 | 0.0 | 59 |  | 59 | 1 | 12000.0 | 1797.0 | 0.0 |
| cwru | OR021@3_1 | 1.0 | 59 |  | 59 | 1 | 12000.0 | 1772.0 | 1.0 |
| cwru | OR021@3_2 | 2.0 | 59 |  | 59 | 1 | 12000.0 | 1750.0 | 2.0 |
| cwru | OR021@3_3 | 3.0 | 59 |  | 59 | 1 | 12000.0 | 1730.0 | 3.0 |
| cwru | OR021@12_0 | 0.0 | 59 |  | 59 | 1 | 12000.0 | 1797.0 | 0.0 |
| cwru | OR021@12_1 | 1.0 | 59 |  | 59 | 1 | 12000.0 | 1772.0 | 1.0 |
| cwru | OR021@12_2 | 2.0 | 59 |  | 59 | 1 | 12000.0 | 1750.0 | 2.0 |
| cwru | OR021@12_3 | 3.0 | 59 |  | 59 | 1 | 12000.0 | 1730.0 | 3.0 |
| cwru | IR028_0 | 0.0 | 58 |  | 58 | 1 | 12000.0 | 1797.0 | 0.0 |
| cwru | IR028_1 | 1.0 | 59 |  | 59 | 1 | 12000.0 | 1772.0 | 1.0 |
| cwru | IR028_2 | 2.0 | 59 |  | 59 | 1 | 12000.0 | 1750.0 | 2.0 |
| cwru | IR028_3 | 3.0 | 59 |  | 59 | 1 | 12000.0 | 1730.0 | 3.0 |
| cwru | B028_0 | 0.0 | 58 |  | 58 | 1 | 12000.0 | 1797.0 | 0.0 |
| cwru | B028_1 | 1.0 | 59 |  | 59 | 1 | 12000.0 | 1772.0 | 1.0 |
| cwru | B028_2 | 2.0 | 59 |  | 59 | 1 | 12000.0 | 1750.0 | 2.0 |
| cwru | B028_3 | 3.0 | 59 |  | 59 | 1 | 12000.0 | 1730.0 | 3.0 |
| cwru | Normal_0 | 0.0 | 60 |  | 60 | 1 | 12000.0 | 1797.0 | 0.0 |
| cwru | Normal_1 | 1.0 | 60 |  | 60 | 1 | 12000.0 | 1772.0 | 1.0 |
| cwru | Normal_2 | 2.0 | 60 |  | 60 | 1 | 12000.0 | 1750.0 | 2.0 |
| cmapss FD001 | 200 engines |  | 7005 |  | 7005 | 21 |  |  |  |
| cmapss FD002 | 519 engines |  | 18271 |  | 18271 | 21 |  |  |  |
| cmapss FD003 | 200 engines |  | 8372 |  | 8372 | 21 |  |  |  |
| cmapss FD004 | 497 engines |  | 20756 |  | 20756 | 21 |  |  |  |

*Source: `reports/tables/data_audit.csv`*

* **CWRU** 12 kHz drive-end: 64 files (4 normal, 60 faulty), fault-labelled only (not run-to-failure). Health state is an assumption: Normal = healthy, 0.007 in = early, 0.014/0.021 in = severe.
* **FEMTO/PRONOSTIA**: 17 full run-to-failure bearings (6 learning set + 11 full test set), 2 channels, 25.6 kHz, 0.1 s snapshot every 10 s. Each bearing is thinned to at most 300 evenly spaced windows to keep CPU cost low (whole life kept). Temperature files were **not** used (limitation).
* **XJTU-SY**: 15 bearings, 25.6 kHz, 1.28 s per minute; the centre 2560 samples of each snapshot are used so windows match FEMTO.
* **C-MAPSS** FD001-FD004, 30-cycle windows.
* **Bearing geometry**: only CWRU's 6205 geometry is used (envelope energy at BPFO/BPFI/BSF/FTF and harmonics), flagged `verified: false` in `configs/datasets.yaml` because the source page could not be fetched in this session. FEMTO and XJTU geometry is `null` (not found in retrievable sources); those datasets use generic envelope bands instead.

## 4. Methods
Sensor branch: RMS, std, skew, kurtosis, crest factor, peak-to-peak, 8 band energies, spectral entropy/centroid, envelope-spectrum energy per channel. Image branch: STFT (FEMTO, XJTU) or CWT (CWRU) image at 224x224 -> frozen ResNet18 (ImageNet weights, no fine-tuning) -> PCA (16 components, fit on training bearings only). **These images are derived from the vibration signal itself ("multi-representation fusion"), not photographs.** Fusion = concatenation -> LightGBM. Controls: sensor-only, image-only, Random Forest, mean-lifetime and Weibull baselines, and fused with *shuffled* embeddings (images no longer belong to their window).
RUL target: remaining life fraction, flat at 0.4 until 60% of life (onset defined by a life-fraction threshold, not by RMS first-prediction-time) and linear afterwards. Health states: <60% / 60-90% / >90% of life. Uncertainty: quantile LightGBM (alpha/2, 1-alpha/2), split-conformal with absolute residuals, and conformalised quantile regression (CQR); the calibration set is 4 bearings held out of both training and testing. Decision: expected-cost rule over maintain-now / schedule / continue using a normal approximation of the interval (`configs/costs.yaml`).

## 5. Evaluation protocol
Splits are by group, never by window: leave-one-load-out (CWRU), leave-one-bearing-out (FEMTO), source-vs-target dataset (cross-dataset), official train/test engines (C-MAPSS). Scalers and PCA are fit on training rows only, and unit tests fail if a group id appears in two splits (`tests/`). Metrics: macro-F1, per-class recall, confusion matrix; RMSE, MAE, mean NASA score (late penalised more, computed on percent-of-life); empirical vs nominal coverage and width. Five seeds; per-bearing means are compared with paired Wilcoxon tests.

## 6. Results

### 6.1 RQ1: does fusion help?
**Classification (CWRU, leave-one-load-out).**
![](figures/cwru_f1.png)
| task | model | macro_f1_mean | macro_f1_std | acc_mean |
|---|---|---|---|---|
| ftype | lgbm_fused | 0.999 | 0.001 | 0.999 |
| ftype | lgbm_fused_shuffled | 0.998 | 0.002 | 0.998 |
| ftype | lgbm_image | 0.964 | 0.028 | 0.963 |
| ftype | lgbm_sensor | 0.998 | 0.002 | 0.998 |
| ftype | rf_sensor | 0.999 | 0.001 | 0.999 |
| state | lgbm_fused | 0.998 | 0.003 | 0.998 |
| state | lgbm_fused_shuffled | 0.998 | 0.004 | 0.997 |
| state | lgbm_image | 0.965 | 0.021 | 0.964 |
| state | lgbm_sensor | 0.998 | 0.004 | 0.997 |
| state | rf_sensor | 0.997 | 0.005 | 0.996 |

*Source: `reports/tables/cwru_classification_summary.csv`*
| task | comparison | mean_diff | wilcoxon_p |
|---|---|---|---|
| state | fused vs lgbm_sensor | 0.0 | 0.18 |
| state | fused vs lgbm_image | 0.033 | 0.0 |
| state | fused vs lgbm_fused_shuffled | 0.001 | 0.18 |
| ftype | fused vs lgbm_sensor | 0.0 | 0.22 |
| ftype | fused vs lgbm_image | 0.035 | 0.0 |
| ftype | fused vs lgbm_fused_shuffled | 0.0 | 0.017 |

*Source: `reports/tables/cwru_classification_wilcoxon.csv`*
Every sensor-based model is near ceiling (macro-F1 about 0.998-0.999). Fusion is not distinguishable from sensor-only or from fused-with-shuffled-images (p = 0.18 for health state). Image-only is clearly worse (about 0.965, p < 1e-4). Caveat: CWRU holds the same physical specimens at every load, so leave-one-load-out does not test unseen bearing instances; this saturation is a known property of the dataset.

**RUL (FEMTO, leave-one-bearing-out, 17 bearings x 5 seeds).**
![](figures/femto_rmse.png)
| model | rmse_mean | rmse_std | mae_mean | mae_std | nasa_mean | nasa_std |
|---|---|---|---|---|---|---|
| rf_sensor | 0.102 | 0.019 | 0.075 | 0.019 | 1.582 | 0.56 |
| lgbm_fused | 0.105 | 0.021 | 0.078 | 0.02 | 1.631 | 0.646 |
| lgbm_fused_shuffled | 0.106 | 0.022 | 0.079 | 0.021 | 1.655 | 0.653 |
| lgbm_sensor | 0.107 | 0.021 | 0.08 | 0.021 | 1.669 | 0.658 |
| weibull | 0.111 | 0.041 | 0.062 | 0.022 | 3.07 | 2.091 |
| lgbm_image | 0.112 | 0.016 | 0.084 | 0.015 | 2.095 | 0.682 |
| mean_lifetime | 0.148 | 0.05 | 0.093 | 0.038 | 3.916 | 1.691 |

*Source: `reports/tables/femto_rul_summary.csv`*
| comparison | mean_rmse_diff | wilcoxon_p |
|---|---|---|
| lgbm_fused vs lgbm_sensor | -0.002 | 0.008 |
| lgbm_fused vs lgbm_image | -0.007 | 0.132 |
| lgbm_fused vs lgbm_fused_shuffled | -0.001 | 0.051 |
| lgbm_fused vs rf_sensor | 0.003 | 0.225 |
| lgbm_fused vs weibull | -0.006 | 0.579 |
| lgbm_fused vs mean_lifetime | -0.043 | 0.008 |

*Source: `reports/tables/femto_rul_wilcoxon.csv`*
Fused LightGBM beats sensor-only by 0.002 RMSE (p = 0.008): statistically detectable but practically negligible. It is not better than the sensor-only Random Forest (p = 0.22) or the Weibull baseline in RMSE (p = 0.58), and the shuffled-image control is almost as good (p = 0.051), so most of the fused model's skill comes from the sensor features and operating-condition covariates, not from the image content. The Weibull baseline has the lowest MAE but a much higher NASA score (the asymmetric score penalises its errors more). All learned models clearly beat the mean-lifetime baseline. **Conclusion: on these data fusion gives at best a marginal gain; the hypothesis that images add substantial information is not supported.**

![](figures/femto_rul_bands.png)
| life_bin | mae |
|---|---|
| 0.0-0.2 | 0.025 |
| 0.2-0.4 | 0.061 |
| 0.4-0.6 | 0.1 |
| 0.6-0.8 | 0.077 |
| 0.8-1.0 | 0.13 |

*Source: `reports/tables/femto_error_by_life.csv`*
Error grows late in life: FEMTO bearings often fail abruptly, so degradation is hard to see far ahead (see per-bearing plots above).

**Explainability.** Summed mean |SHAP|: sensor features dominate; the image embedding and operating condition contribute roughly an order of magnitude less.
| modality | mean_abs_shap |
|---|---|
| image (CNN embedding) | 0.015 |
| operating condition | 0.013 |
| sensor features | 0.203 |

*Source: `reports/tables/shap_modality.csv`*
![](figures/shap_top.png)

### 6.2 RQ2: do intervals reach nominal coverage?
| method | nominal | coverage | coverage_std | width |
|---|---|---|---|---|
| conformal_abs | 0.8 | 0.823 | 0.126 | 0.235 |
| conformal_abs | 0.9 | 0.9 | 0.081 | 0.274 |
| conformal_abs | 0.95 | 0.943 | 0.057 | 0.302 |
| cqr | 0.8 | 0.797 | 0.099 | 0.166 |
| cqr | 0.9 | 0.88 | 0.078 | 0.215 |
| cqr | 0.95 | 0.941 | 0.054 | 0.3 |
| raw_quantile | 0.8 | 0.716 | 0.144 | 0.157 |
| raw_quantile | 0.9 | 0.798 | 0.121 | 0.194 |
| raw_quantile | 0.95 | 0.875 | 0.103 | 0.266 |

*Source: `reports/tables/femto_coverage_summary.csv`*
![](figures/femto_calibration.png)
Raw quantile regression under-covers (about 0.80 at nominal 0.90). Split-conformal with absolute residuals reaches nominal coverage (0.82 / 0.90 / 0.94 at 0.80 / 0.90 / 0.95) and CQR is close with narrower intervals at 80% and 90% (0.80 / 0.88 / 0.94). Spread across bearings is large (std 0.05-0.13): coverage is marginal, not per-bearing, and windows within a bearing are autocorrelated, so the exchangeability assumption is only approximate.

**Decision layer (assumed costs).**
| policy | mean_cost | failure_rate |
|---|---|---|
| fixed_interval | 3.49 | 0.235 |
| point_estimate | 7.384 | 0.706 |
| run_to_failure | 10.0 | 1.0 |
| uncertainty_aware | 1.321 | 0.0 |

*Source: `reports/tables/femto_decision_summary.csv`*
![](figures/decision_cost.png)
With the assumed costs (failure 10x a planned stop), the interval-aware rule avoids all failures in this replay and is cheapest; a point-estimate version of the same rule fails on 12 of 17 bearings, which shows why the interval matters. The absolute numbers depend entirely on the assumed costs.

### 6.3 RQ3: cross-dataset (FEMTO <-> XJTU-SY)
| norm | setting | model | rmse_mean | rmse_std | mae_mean | mae_std | nasa_mean | nasa_std |
|---|---|---|---|---|---|---|---|---|
| per_condition_norm | femto in-domain LOBO | fused | 0.103 |  | 0.072 |  | 1.502 |  |
| per_condition_norm | femto in-domain LOBO | image | 0.11 |  | 0.081 |  | 2.04 |  |
| per_condition_norm | femto in-domain LOBO | sensor | 0.102 |  | 0.073 |  | 1.506 |  |
| per_condition_norm | femto->xjtu | fused | 0.134 | 0.004 | 0.112 | 0.004 | 2.165 | 0.081 |
| per_condition_norm | femto->xjtu | image | 0.132 | 0.0 | 0.1 | 0.0 | 3.226 | 0.02 |
| per_condition_norm | femto->xjtu | sensor | 0.136 | 0.002 | 0.113 | 0.002 | 2.181 | 0.035 |
| per_condition_norm | xjtu in-domain LOBO | fused | 0.121 |  | 0.085 |  | 2.107 |  |
| per_condition_norm | xjtu in-domain LOBO | image | 0.116 |  | 0.088 |  | 2.173 |  |
| per_condition_norm | xjtu in-domain LOBO | sensor | 0.123 |  | 0.087 |  | 2.142 |  |
| per_condition_norm | xjtu->femto | fused | 0.115 | 0.002 | 0.086 | 0.001 | 2.154 | 0.023 |
| per_condition_norm | xjtu->femto | image | 0.128 | 0.0 | 0.102 | 0.0 | 2.684 | 0.018 |
| per_condition_norm | xjtu->femto | sensor | 0.112 | 0.001 | 0.084 | 0.001 | 2.025 | 0.036 |
| raw | femto in-domain LOBO | fused | 0.105 |  | 0.077 |  | 1.584 |  |
| raw | femto in-domain LOBO | image | 0.113 |  | 0.084 |  | 2.166 |  |
| raw | femto in-domain LOBO | sensor | 0.108 |  | 0.08 |  | 1.627 |  |
| raw | femto->xjtu | fused | 0.191 | 0.006 | 0.164 | 0.008 | 3.796 | 0.181 |
| raw | femto->xjtu | image | 0.132 | 0.001 | 0.105 | 0.001 | 2.76 | 0.025 |
| raw | femto->xjtu | sensor | 0.206 | 0.005 | 0.181 | 0.004 | 4.402 | 0.26 |
| raw | xjtu in-domain LOBO | fused | 0.116 |  | 0.086 |  | 2.064 |  |
| raw | xjtu in-domain LOBO | image | 0.118 |  | 0.089 |  | 2.284 |  |
| raw | xjtu in-domain LOBO | sensor | 0.117 |  | 0.086 |  | 2.096 |  |
| raw | xjtu->femto | fused | 0.115 | 0.002 | 0.092 | 0.005 | 2.132 | 0.091 |
| raw | xjtu->femto | image | 0.124 | 0.0 | 0.095 | 0.0 | 2.632 | 0.028 |
| raw | xjtu->femto | sensor | 0.114 | 0.002 | 0.092 | 0.003 | 2.177 | 0.029 |

*Source: `reports/tables/generalisation.csv`*
![](figures/generalisation.png)
Training on FEMTO and testing on XJTU-SY with raw features degrades the sensor and fused models strongly (RMSE about 0.19-0.21 vs about 0.12 in-domain), while the image model degrades less (0.13). Per-condition normalisation (label-free z-scoring within each dataset/condition) removes most of the loss for the sensor and fused models (about 0.134-0.136). The reverse direction (XJTU to FEMTO) is close to in-domain in both settings (about 0.112-0.129). Note the cross-dataset models have no operating-condition covariates, since conditions do not overlap between datasets.
| setting | calibration | coverage | coverage_std | width | nominal |
|---|---|---|---|---|---|
| femto->xjtu | 3 labelled target bearings | 0.898 | 0.086 | 0.334 | 0.9 |
| femto->xjtu | source bearings (no recalibration) | 0.834 | 0.048 | 0.296 | 0.9 |
| xjtu->femto | 3 labelled target bearings | 0.896 | 0.025 | 0.266 | 0.9 |
| xjtu->femto | source bearings (no recalibration) | 0.911 | 0.081 | 0.288 | 0.9 |

*Source: `reports/tables/generalisation_coverage.csv`*
![](figures/generalisation_coverage.png)
Under shift, conformal intervals calibrated on source bearings undercover FEMTO->XJTU (0.83 vs 0.90 nominal); recalibrating with 3 labelled target bearings restores about 0.90 in both directions (at the cost of wider intervals for FEMTO->XJTU).

### 6.4 RQ4: C-MAPSS through a new adapter
Only `build_features.cmapss` (the adapter: 30-cycle sensor windows -> features and a sensor-heatmap image) differs; the featurizer, models, conformal and evaluation code are reused. Target: RUL capped at 125 cycles, official test last-window labels.
| fd | model | rmse_mean | rmse_std | mae_mean | mae_std | nasa_mean | nasa_std |
|---|---|---|---|---|---|---|---|
| 1 | lgbm_fused | 12.719 | 0.468 | 9.575 | 0.242 | 2.612 | 0.438 |
| 1 | lgbm_fused_shuffled | 12.837 | 0.526 | 9.665 | 0.412 | 2.602 | 0.359 |
| 1 | lgbm_image | 26.161 | 0.614 | 20.324 | 0.597 | 37.73 | 4.171 |
| 1 | lgbm_sensor | 12.762 | 0.505 | 9.563 | 0.409 | 2.545 | 0.322 |
| 2 | lgbm_fused | 18.11 | 0.105 | 14.065 | 0.101 | 8.997 | 0.402 |
| 2 | lgbm_fused_shuffled | 17.988 | 0.156 | 13.906 | 0.201 | 9.211 | 0.647 |
| 2 | lgbm_image | 43.561 | 0.223 | 37.499 | 0.185 | 340.107 | 26.09 |
| 2 | lgbm_sensor | 17.898 | 0.15 | 13.845 | 0.206 | 9.04 | 0.304 |
| 3 | lgbm_fused | 13.094 | 0.31 | 9.267 | 0.113 | 4.29 | 1.174 |
| 3 | lgbm_fused_shuffled | 13.199 | 0.339 | 9.367 | 0.27 | 4.112 | 0.711 |
| 3 | lgbm_image | 24.11 | 1.062 | 18.091 | 0.705 | 43.294 | 19.93 |
| 3 | lgbm_sensor | 13.212 | 0.335 | 9.375 | 0.267 | 3.992 | 0.45 |
| 4 | lgbm_fused | 19.443 | 0.148 | 14.689 | 0.126 | 10.361 | 0.565 |
| 4 | lgbm_fused_shuffled | 19.615 | 0.149 | 14.771 | 0.124 | 11.26 | 0.684 |
| 4 | lgbm_image | 44.026 | 0.18 | 36.828 | 0.075 | 543.464 | 34.565 |
| 4 | lgbm_sensor | 19.513 | 0.344 | 14.629 | 0.178 | 11.07 | 0.705 |

*Source: `reports/tables/cmapss_summary.csv`*
| fd | coverage | width |
|---|---|---|
| 1 | 0.93 | 48.639 |
| 2 | 0.924 | 65.801 |
| 3 | 0.88 | 40.734 |
| 4 | 0.898 | 65.5 |

*Source: `reports/tables/cmapss_coverage.csv`*
![](figures/cmapss_rmse.png)
| Component | Transfers? |
|---|---|
| Feature/fusion/LightGBM code, split-by-engine, metrics | unchanged |
| Split-conformal 90% intervals | unchanged, coverage 0.88-0.93 |
| Sensor features | retuned (per-sensor last/mean/std/slope replaces vibration statistics) |
| Image branch | **failed**: image-only RMSE is 1.8-2.4x worse than sensors, and fusion adds nothing (fused ≈ sensor ≈ shuffled). Per-window min-max heatmaps discard absolute sensor level. |
| Decision layer | not run (RUL in cycles; the decision units are life fractions) |

## 7. Limitations
* Images are signal-derived; the image branch is a re-encoding of the same vibration, so little new information is expected. No surface-defect images were used.
* Costs in the decision layer are invented; the policy ranking is not evidence about real savings.
* FEMTO/XJTU geometry unavailable, so fault-frequency features exist only for CWRU, and CWRU geometry is flagged unverified.
* Windows were thinned to at most 300 per bearing; temperature data unused; one CNN (ResNet18) and one image type per dataset were tried.
* Accelerated-test conditions are far above rating loads, so L10 relations are only a qualitative prior. XJTU operating conditions and FEMTO conditions are taken from dataset descriptions (`conditions_verified: false`).
* RUL targets are in life fractions (needed for cross-dataset comparison); converting to time requires an estimate of total life.
* Calibration uses 4 bearings; coverage varies a lot between bearings. Wilcoxon tests use 17 (FEMTO) paired bearings; CWRU tests pair (load, seed) combinations and are therefore optimistic.

## 8. Future work
Verify geometry from the original papers and add FEMTO/XJTU fault-frequency features; use temperature; tune the onset with an RMS first-prediction time; try fine-tuned CNNs or LSTM baselines; conformal methods robust to autocorrelation; calibrate costs with a maintenance partner.

## 9. Reproduce
`make setup data features eval test app` (see README). Run records: `reports/runs/`.

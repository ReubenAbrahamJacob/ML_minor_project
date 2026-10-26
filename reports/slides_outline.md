# 12-slide outline
1. Problem and three questions (state, RUL + interval, action)
2. Bearing mechanics: contact fatigue, L10, Weibull; why uncertainty (and the 8.85 vs 6.84 slide note)
3. Data: CWRU, FEMTO, XJTU-SY, C-MAPSS (reports/tables/data_audit.csv)
4. Pipeline diagram: sensor branch + signal-derived image branch -> fusion -> conformal -> decision
5. Leakage-safe protocol (group splits, train-only PCA, tests)
6. RQ1 classification: near-ceiling, image-only worse (figures/cwru_f1.png)
7. RQ1 RUL: fused ~ sensor, shuffled control, Weibull baseline (figures/femto_rmse.png, femto_rul_bands.png)
8. Explainability: SHAP per modality (figures/shap_modality.png)
9. RQ2 coverage: raw quantile vs conformal vs CQR (figures/femto_calibration.png)
10. Decision layer with assumed costs (figures/decision_cost.png)
11. RQ3 cross-dataset + recalibration (figures/generalisation.png, generalisation_coverage.png)
12. RQ4 C-MAPSS adapter, limitations, future work

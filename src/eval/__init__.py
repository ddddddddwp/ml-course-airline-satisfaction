"""P3 评估层：指标 / 主评估 / 稳定性 / 可解释性 / 误差分析。

模块（都可独立 `python -m src.eval.<module> --config configs/base.yaml` 运行）：
    metrics    binary_metrics / reliability_curve / bootstrap_auc_ci
    evaluate   OOF -> metrics.json + reliability.csv + segment_metrics.csv + summary.md + 3 张图
    stability  多 seed × 5 折 -> stability.json + p3_seed_variance.png + stability.md
    interpret  LightGBM pred_contrib TreeSHAP（OOF 目标编码，不依赖 shap 包）
    badcase    高置信 FP/FN top-50 + 分群错误率

口径：ROC AUC 是主指标；阈值指标与混淆矩阵只作描述，提交必须给概率。
"""

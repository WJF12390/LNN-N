# Improved LNN for Medication Packaging and Adherence Prediction

本仓库提供一套面向“药品定制包装—用药依从性”研究的改进型液态神经网络（Improved Liquid Neural Network, Improved LNN）代码模板。

> 重要说明：如果你的数据是横截面问卷数据，LNN 不能被解释为真实动态因果模型。本仓库将 LNN 用作探索性预测模型，用于与传统机器学习模型进行比较，并辅助识别设计变量的预测贡献。

## 1. 适用场景

- 以问卷变量预测连续型结果变量，例如“用药依从性”；
- 比较 LNN、MLP、Ridge、随机森林、ExtraTrees、梯度提升树等模型；
- 使用 K 折交叉验证输出 R²、MAE、RMSE；
- 进行 permutation importance 变量重要性分析；
- 生成结果 CSV 和预测散点图，方便论文复现。

## 2. 推荐变量设置

输入变量：视觉设计、功能性设计、体验设计、感知体验、日常管理体验。

目标变量：用药依从性。

如果你有更细的题项数据，也可以在配置文件中将题项列作为输入变量。

## 3. 环境安装

建议使用 Python 3.10 或以上版本。

```bash
git clone https://github.com/your-name/improved-lnn-medication-packaging.git
cd improved-lnn-medication-packaging
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 4. 数据格式

将 CSV 文件放入 `data/` 文件夹，例如：

```text
data/medication_packaging.csv
```

数据应至少包含：

```text
视觉设计, 功能性设计, 体验设计, 感知体验, 日常管理体验, 用药依从性
```

注意：不要将包含隐私信息的原始问卷数据直接公开到 GitHub。

## 5. 快速运行

生成模拟数据：

```bash
python scripts/make_synthetic_data.py --out data/demo_medication_packaging.csv
```

运行完整实验：

```bash
python src/run_experiment.py   --data data/demo_medication_packaging.csv   --target 用药依从性   --features 视觉设计 功能性设计 体验设计 感知体验 日常管理体验   --out outputs/demo_run   --epochs 300   --folds 5
```

Windows PowerShell 可写成一行：

```bash
python src/run_experiment.py --data data/demo_medication_packaging.csv --target 用药依从性 --features 视觉设计 功能性设计 体验设计 感知体验 日常管理体验 --out outputs/demo_run --epochs 300 --folds 5
```

## 6. 输出结果

运行后会生成：

```text
outputs/demo_run/
├── metrics_all_models.csv
├── metrics_summary.csv
├── predictions_lnn.csv
├── permutation_importance_lnn.csv
├── prediction_scatter_lnn.png
├── training_curve_fold_1.png
└── config_used.json
```

## 7. 方法说明

本仓库中的 LNN 是一种适配表格问卷数据的“特征步进式液态网络”。它将输入变量按照理论路径顺序组织为伪序列：

```text
视觉设计 → 功能性设计 → 体验设计 → 感知体验 → 日常管理体验
```

这种顺序反映研究者的理论假设，而不是用户行为的真实时间序列。因此，模型结果只能解释为探索性预测结果，不能解释为动态因果关系。

## 8. 论文中推荐写法

> 由于本文数据来源于横截面问卷，并不具备连续时间序列特征，因此本文不将液态神经网络解释为动态因果模型，而仅将其作为探索性预测补充。具体而言，本文依据“视觉设计—功能性设计—体验设计—感知体验—日常管理体验”的理论顺序，将输入变量构造为特征步进序列，并通过液态时间常数单元对变量组合关系进行建模。模型输出用于与传统机器学习方法进行预测比较，并辅助判断各设计变量对用药依从性的预测贡献。

## 9. 仓库结构

```text
improved_lnn_medication_packaging/
├── configs/default_config.json
├── data/README.md
├── scripts/make_synthetic_data.py
├── src/run_experiment.py
├── src/data_utils.py
├── src/metrics.py
├── src/train_utils.py
├── src/models/improved_lnn.py
├── src/models/baselines.py
├── tests/test_shapes.py
├── requirements.txt
├── LICENSE
└── README.md
```

## 10. 许可证

MIT License.

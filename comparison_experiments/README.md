# Comparison Experiments

这个目录用于做基线对比实验，当前包含三类模型：

- `LSTM`
- `GRU`
- `GCN`

## 运行方式

在根目录执行：

```bash
python comparison_experiments/train_baseline.py --model lstm
python comparison_experiments/train_baseline.py --model gru
python comparison_experiments/train_baseline.py --model gcn
```

## 输出目录

结果会保存在：

- `comparison_experiments/outputs/lstm/`
- `comparison_experiments/outputs/gru/`
- `comparison_experiments/outputs/gcn/`

每个模型都会生成：

- `best_model.pt`
- `loss_history.csv`
- `loss_history.json`
- `loss_curves.png`
- `pred_vs_true.png`
- `test_metrics.json`

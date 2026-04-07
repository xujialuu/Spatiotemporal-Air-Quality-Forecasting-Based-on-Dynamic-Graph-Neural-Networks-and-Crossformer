# Dynamic-GNN + Crossformer Air Quality Forecasting

这个项目基于整理好的北京多站点空气质量与气象数据，构建了一个可训练的 PyTorch 时空预测框架：

- 空间分支：风场驱动动态邻接矩阵 + 两层 GCN
- 时间分支：简化可运行版 Crossformer，包含 patch embedding + two-stage attention
- 融合分支：并行预测后的特征级融合
- 监督方式：`L = alpha * L_space + beta * L_time + gamma * L_fusion`

## 项目结构

```text
project/
├── configs/
│   └── default.yaml
├── data/
│   ├── dataset.py
│   ├── preprocess.py
│   └── graph_utils.py
├── models/
│   ├── crossformer_blocks.py
│   ├── fusion.py
│   ├── graph_layers.py
│   ├── model.py
│   ├── spatial_branch.py
│   └── temporal_branch.py
├── utils/
│   ├── losses.py
│   ├── metrics.py
│   ├── seed.py
│   └── trainer.py
├── evaluate.py
├── train.py
└── README.md
```

## 数据说明

默认读取：

- `../station_csv_no_empty_cols`
- `../北京市站点列表-2021.01.23起.xlsx`

默认输入特征：

- 污染物：`PM2.5`, `PM10`, `NO2`, `SO2`, `O3`, `CO`
- 气象：`temperature`, `dew_point`, `pressure`, `wind_speed`, `wind_direction`, `precipitation`

`feature_columns` 实现了逻辑特征名到实际 CSV 列名的映射，因此模型代码本身没有把原始列名写死。

## 环境依赖

```bash
pip install torch pyyaml numpy tqdm
```

项目数据读取没有依赖 `pandas/openpyxl`。

## 训练

在 `project/` 目录下运行：

```bash
python train.py --config configs/default.yaml
```

输出包括：

- 训练/验证日志
- 最优模型权重：`outputs/best_model.pt`
- 测试集指标：`MAE / RMSE / R2`

## 评估

```bash
python evaluate.py --config configs/default.yaml
```

## 关键张量

- 输入 `x`: `[B, T, N, F]`
- 目标 `y`: `[B, K, N, C]`
- 风速序列 `wind_speed_seq`: `[B, T, N]`
- 风向序列 `wind_dir_seq`: `[B, T, N]`
- 坐标 `coords`: `[N, 2]`

## 动态邻接矩阵

静态邻接：

```text
A_static[i,j] = exp( -d(i,j)^2 / sigma^2 )
```

动态邻接：

```text
A_ij(t) = A_static_ij * max(0, cos(theta_i(t) - phi_ij)) * v_i(t)
```

其中：

- `theta_i(t)` 是站点 `i` 在时刻 `t` 的风向
- `phi_ij` 是从站点 `i` 指向 `j` 的方位角
- `v_i(t)` 是站点 `i` 的风速

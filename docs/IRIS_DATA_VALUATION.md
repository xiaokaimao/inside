# Iris data valuation：OFA 与 Frame-OFA 实验

## 1. RMSE 越低越好吗？

是。本文使用

\[
\operatorname{RMSE}
=
\sqrt{\frac1n\sum_{i=1}^n
(\widehat\phi_i-\phi_i^{\mathrm{GT}})^2}.
\]

它与 utility 使用相同单位，零是最好结果。在 ground truth 足够精确、
utility-call 预算相同的前提下，RMSE 越低表示整个 Shapley 向量越准确。

## 2. 实验设置

- 环境：conda `svmsv`，Python 3.13.2，scikit-learn 1.6.1。
- 数据：sklearn Iris；复现官方 OFA 的 seed=2024、80% train split、
  标准化和类别平衡抽取。
- 玩家：24 个 valued training points，每类 8 个。
- performance set：24 个不同样本，每类 8 个。
- 模型：`sklearn.svm.SVC(kernel="rbf", probability=False)`。
- utility：performance classification accuracy。
- 空 coalition：performance set 上最佳常数类别准确率，本实验为 \(1/3\)。
- 单类 coalition：恒预测 coalition 中的唯一类别。
- 其他 coalition：正常拟合 RBF-SVM。

这些规则对所有 estimator 完全相同。该 game 与官方 PyTorch
one-epoch logistic/cross-entropy game 不同；OFA 本身与底层模型无关。

比较四种方法：

1. `official_ofa_fixed_ratio`：官方 Shapley-specific \(q_s\) 和 ratio
   conditional-mean 聚合；
2. `iid_linear_ofa`：同一 \(q_s\) 下的 IID 线性无偏估计器；
3. `frame_coupled`：保持相同逐样本边际的 coupled linear Frame-OFA；
4. `frame_orbit_ratio`：完整 cyclic-orbit frame 加官方 ratio 聚合。

每个 estimator 都计入官方边界的 \(2n+2=50\) 次 utility calls。

## 3. High-budget MC ground truth

ground truth 使用 permutation 和 reverse-permutation antithetic pairing：

- 40,000 个独立 antithetic pairs；
- 80,000 条 permutations；
- 复用空集和全集后，等价 utility calls 为
  \[
  2+80{,}000(24-1)=1{,}840{,}002;
  \]
- 16 个 spawn workers，耗时 91.2 秒；
- ground-truth SE-RMSE：\(2.969\times10^{-4}\)；
- half-split RMSE 诊断：\(2.946\times10^{-4}\)。

SE-RMSE 约为最佳 estimator RMSE 的 12.5%。Bonferroni simultaneous
interval 的最大半宽仍约为最佳 RMSE 的 43.3%，所以这些 samples 足以
支持总体 RMSE 比较，但不应把非常接近的单个玩家排序当成精确真值。

## 4. 结果

每个点包含 20 个 estimator repeats；各方法的 bootstrap 95% 区间
保存在结果 JSON 中，表中列出 aggregate RMSE。

| Inner calls | Total calls | Official ratio OFA | IID linear | Coupled linear | Orbit ratio Frame-OFA |
|---:|---:|---:|---:|---:|---:|
| 1,200 | 1,250 | 0.006486 | 0.011559 | 0.010421 | **0.005208** |
| 2,400 | 2,450 | 0.004444 | 0.008159 | 0.006617 | **0.003516** |
| 4,800 | 4,850 | 0.002965 | 0.005381 | 0.004864 | **0.002380** |

Orbit-ratio Frame-OFA 相对官方 ratio OFA 的 RMSE 降幅分别为：

- 1,200 calls：19.70%；
- 2,400 calls：20.89%；
- 4,800 calls：19.72%。

对相同 repeat seed 做 paired bootstrap，`official RMSE - orbit RMSE`
的 95% 区间分别为：

- \([8.43\times10^{-4},1.69\times10^{-3}]\)；
- \([6.20\times10^{-4},1.26\times10^{-3}]\)；
- \([4.03\times10^{-4},7.56\times10^{-4}]\)。

三个区间都严格大于零。

Coupled-linear 相对 IID-linear 的 RMSE 下降为 9.85%、18.91% 和
9.62%。这证明 geometry-aware coupling 本身有效，但 coupled-linear
仍不如官方 ratio estimator。

## 5. 为什么 orbit 版本更好？

Iris accuracy utility 有很强的 coalition-size 成分。

- 直接 linear/HT estimator 即使使用 endpoint-linear baseline，仍会
  承受未被 baseline 捕获的 slice-level 波动；
- 官方 ratio estimator 逐 size 估计 in/out conditional means，能够
  自动抵消 size-only 水平；
- cyclic orbit 在保留 ratio 聚合的同时，让每个玩家在每个 size 上的
  in/out counts 固定，并严格实现一阶方向平衡。

因此，本实验支持的改进不是“用 coupled-linear 取代 OFA”，而是：

\[
\boxed{
\text{官方 ratio OFA}
+
\text{cyclic-orbit frame-balanced sampling}.
}
\]

## 6. 多进程

现在代码支持持久化进程池：

- 每个 worker 只构造一个私有 game；
- 使用 `spawn`，不共享 Torch/sklearn model；
- 保序返回 utilities；
- worker 内 Torch、OpenBLAS、MKL 都限制为单线程；
- 不缓存重复 coalition，因此 query accounting 不变。

40,000-pair ground truth 达到约 20,173 calls/s。相同 RBF-SVM utility
单线程 microbenchmark 为约 1,281 calls/s，对应约 15.7 倍吞吐提升。

## 7. 复现

```bash
conda run -n svmsv python -m experiments.iris_data_valuation \
  --game rbf_svm_accuracy \
  --gt-pairs 40000 \
  --budgets 1200 2400 4800 \
  --repeats 20 \
  --jobs 16 \
  --chunksize 32 \
  --candidate-pool 16 \
  --output results/iris_rbf_svm_frame_ofa.json
```

完整 ground truth、每次 repeat 的 Shapley vectors、bootstrap 区间、
数据索引和计时均保存在结果 JSON 中。

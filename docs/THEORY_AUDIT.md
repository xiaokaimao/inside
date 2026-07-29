# Frame-OFA：公式、定理与算法严格核验

## 结论先行

原几何思路成立，但必须区分三种不同估计器：

1. **全尺寸线性版**：采样 \(s=1,\ldots,n-1\)，frame 目标是 \(P\)。
2. **官方 OFA**：精确计算 \(s=0,1,n-1,n\)，只采样
   \(s=2,\ldots,n-2\)，并使用随机分母的条件均值。
3. **本文实现的 boundary-exact 线性版**：保留官方边界层，但将内层改为
   严格无偏的 coupled-HT 或分层线性估计器。

若修改官方代码，内层 frame 目标是

\[
\alpha P,\qquad
\alpha=\frac{n-3}{n-1},
\]

而不是 \(P\)。此前把这三种形式混写，是最重要的系数错误来源。

---

## 1. 记号

令玩家集合为 \(N=[n]\)，coalition \(S\subseteq N\) 的指示向量为
\(z_S\in\{0,1\}^n\)，\(s=|S|\)。定义

\[
d_S=z_S-\frac{s}{n}\mathbf 1,\qquad
P=I-\frac1n\mathbf1\mathbf1^\top,
\]

以及

\[
u_S=\sqrt{\frac{n}{s(n-s)}}\,d_S.
\]

直接计算得到

\[
\mathbf1^\top u_S=0,\qquad \|u_S\|_2=1.
\]

所以 \(u_S\) 是效率子空间 \(\mathbf1^\perp\) 上的单位方向。

---

## 2. Shapley 的 fixed-size 几何分解

### 定理 1

对任意 cooperative game \(v\)，

\[
\boxed{
\phi=
\frac{v(N)-v(\varnothing)}n\mathbf1+
\sum_{s=1}^{n-1}
\frac{n}{s(n-s)}
\mathbb E_{|S|=s}[v(S)d_S].
}
\]

等价地，

\[
\boxed{
\phi=
\frac{v(N)-v(\varnothing)}n\mathbf1+
\sum_{s=1}^{n-1}
\sqrt{\frac{n}{s(n-s)}}
\mathbb E_{|S|=s}[v(S)u_S].
}
\]

### 证明摘要

固定一个非空真子集 \(S\)，其 utility \(v(S)\) 在玩家 \(i\) 的 Shapley
值中的系数为

\[
\begin{cases}
\dfrac{1}{s\binom ns},&i\in S,\\[4pt]
-\dfrac{1}{(n-s)\binom ns},&i\notin S.
\end{cases}
\]

该系数向量等于

\[
\frac{n}{\binom ns\,s(n-s)}d_S.
\]

对同一 size 的 \(\binom ns\) 个 coalition 收集系数，即得结论。端点
\(S=\varnothing,N\) 合并为 efficiency direction。

---

## 3. 每个 Boolean slice 都是总体 tight frame

### 定理 2

固定 \(s\in\{1,\ldots,n-1\}\)，若 \(S\) 在 size-\(s\) slice 上均匀，
则

\[
\mathbb E[u_S]=0,
\qquad
\boxed{
\mathbb E[u_Su_S^\top]=\frac1{n-1}P.
}
\]

### 证明摘要

超几何包含概率给出

\[
\mathbb E[z_i]=\frac sn,\qquad
\mathbb E[z_iz_j]=\frac{s(s-1)}{n(n-1)}\quad(i\ne j).
\]

因此

\[
\mathbb E[d_Sd_S^\top]
=\frac{s(n-s)}{n(n-1)}P.
\]

乘以 \(n/[s(n-s)]\) 即得结论。

补集满足

\[
u_{N\setminus S}=-u_S,
\]

所以补集提供同一条直线的反方向，而不是新的角向维度。

---

## 4. 对齐官方 OFA

官方 Algorithm 1 只随机采样

\[
D_{\mathrm{int}}=\{2,\ldots,n-2\},
\]

并用 \(2n+2\) 次 utility 精确计算空集、全集、所有 singleton 和所有
leave-one-out coalition。

定义

\[
Z_{\mathrm{int}}
=\sum_{s=2}^{n-2}\frac1{\sqrt{s(n-s)}},
\qquad
q_s^*
=\frac{1}{Z_{\mathrm{int}}\sqrt{s(n-s)}}.
\]

官方精确边界向量为

\[
\boxed{
\begin{aligned}
(b_{\mathrm{exact}})_i
=\frac1n\Big[
&v(\{i\})+v(N)
+\frac1{n-1}\sum_{j\ne i}v(N\setminus\{j\})\\
&-v(\varnothing)-v(N\setminus\{i\})
-\frac1{n-1}\sum_{j\ne i}v(\{j\})
\Big].
\end{aligned}
}
\]

于是 boundary-exact 线性表示为

\[
\boxed{
\phi=b_{\mathrm{exact}}
+Z_{\mathrm{int}}\sqrt n\,
\mathbb E_{Q_{\mathrm{int}}}
[(v(S)-b_{|S|})u_S],
}
\]

其中 \(b_s\) 可以是任意只依赖 size 的 baseline。

由于内部共有 \(n-3\) 个 size，

\[
\mathbb E[F_{\mathrm{int}}]
=\frac{n-3}{n-1}P.
\]

精确处理的 \(s=1,n-1\) 各提供 \(P/(n-1)\)，总和才是 \(P\)。

---

## 5. \(q_s^*\) 的准确含义

对线性 importance-weighted 估计器，单样本系数为

\[
\frac{\sqrt{n/[s(n-s)]}}{q_s}.
\]

若使用统一的 worst-case residual 上界，二阶矩上界要求最小化

\[
\sum_s\frac1{s(n-s)q_s}.
\]

Cauchy–Schwarz 给出唯一最优解

\[
q_s^*\propto\frac1{\sqrt{s(n-s)}}.
\]

这意味着：

> \(q_s^*\) 是指定 worst-case 二阶矩/query-complexity 目标下的最优
> 边际，并不是对每一个具体 utility 都是 MSE 最优。

若已知每层 residual 方差，game-specific 分配一般为

\[
q_s\propto
\sqrt{\frac{n}{s(n-s)}}
\sqrt{\mathbb E_s[(v(S)-b_s)^2]}.
\]

---

## 6. 算法 A：保持 OFA 边际的 coupled-HT

### 采样

1. 取 \(U\sim\operatorname{Unif}(0,1)\)。
2. 令
   \[
   r_t=(U+t/T)\bmod 1,
   \quad s_t=F_{q^*}^{-1}(r_t).
   \]
   每个 \(s_t\) 的边际仍为 \(q^*\)，但 batch size counts 更稳定。
3. 给定 size 序列，选择 base coalitions \(B_t\)，最小化经验 frame
   discrepancy。
4. 取独立均匀排列 \(\pi\)，实际评估 \(S_t=\pi(B_t)\)。

对任意固定 \(B_t\)，\(\pi(B_t)\) 在对应 slice 上均匀，因此每个
\(S_t\) 的边际分布严格为 \(Q_{\mathrm{int}}^*\)。

### 估计器

\[
\boxed{
\widehat\phi
=b_{\mathrm{exact}}
+\frac{Z_{\mathrm{int}}\sqrt n}{T}
\sum_{t=1}^T[v(S_t)-b_{s_t}]u_{S_t}.
}
\]

该估计器对任意 game 无偏；样本之间不需要独立。

### additive 误差恒等式

令

\[
v(S)=c+a^\top z_S,
\qquad
\ell(s)=v(\varnothing)+\frac{s}{n}[v(N)-v(\varnothing)].
\]

则 \(v(S)-\ell(s)=a^\top d_S\)。定义

\[
F_T
=\frac{Z_{\mathrm{int}}}{T}
\sum_t\sqrt{s_t(n-s_t)}u_{S_t}u_{S_t}^\top.
\]

有严格恒等式

\[
\boxed{
\widehat\phi-\phi
=\left(
F_T-\frac{n-3}{n-1}P
\right)a.
}
\]

进一步，固定 size 序列和随机重标号之前的 base batch，记

\[
\bar E=\bar F_T-\frac{n-3}{n-1}P.
\]

只对最后一个共同的均匀玩家排列 \(\pi\) 取条件期望，则

\[
\boxed{
\mathbb E_{\pi\mid\mathrm{base}}
\|\widehat\phi-\phi\|_2^2
=\frac{\|Pa\|_2^2}{n-1}\|\bar E\|_F^2.
}
\]

如果 greedy base 本身也随机，无条件公式的右侧还必须再取
\(\mathbb E_{\mathrm{base}}\)，不能把一个随机的 realized discrepancy
直接写成无条件 MSE。

---

## 7. 算法 B：逐 size 预分配的 stratified Frame-OFA

在约束 \(B_s\ge1\)、\(\sum_sB_s=T\) 下，选择与 \(Tq_s^*\)
平方距离最小的整数预算

\[
B_s\approx Tq_s^*,\qquad B_s\ge1.
\]

在每个 slice 单独构造 batch，并使用

\[
\boxed{
\widehat\phi
=b_{\mathrm{exact}}
+\sum_{s=2}^{n-2}
\sqrt{\frac n{s(n-s)}}
\frac1{B_s}
\sum_{b=1}^{B_s}
[v(S_{s,b})-b_s]u_{S_{s,b}}.
}
\]

只要每个 block 经独立于 utility 的均匀随机重标号，该估计器对任意
game 无偏。\(q_s^*\) 只决定预算，不进入最终 target 权重，因此整数
舍入不会造成径向偏差。

定义

\[
H_s=\frac1{B_s}\sum_bu_{S_{s,b}}u_{S_{s,b}}^\top.
\]

对 additive game \(v(S)=c+a^\top z_S\)，再定义

\[
\mu_s=c+\frac{s}{n}\mathbf1^\top a,\qquad
\bar u_s=\frac1{B_s}\sum_bu_{S_{s,b}}.
\]

任意 size-only baseline \(b_s\) 下的完整误差恒等式是

\[
\boxed{
\widehat\phi-\phi
=
\left[
\sum_{s=2}^{n-2}H_s
-\frac{n-3}{n-1}P
\right]a
+
\sum_{s=2}^{n-2}
\sqrt{\frac n{s(n-s)}}(\mu_s-b_s)\bar u_s.
}
\]

所以只有在使用 endpoint-linear baseline \(b_s=\ell(s)=\mu_s\)，
或每个 slice 都严格一阶平衡 \(\bar u_s=0\) 时，才化简为

\[
\boxed{
\widehat\phi-\phi
=
\left[
\sum_{s=2}^{n-2}H_s
-\frac{n-3}{n-1}P
\right]a.
}
\]

固定随机重标号前的 base batches，令

\[
\bar E_s=\bar H_s-\frac1{n-1}P.
\]

若使用 endpoint-linear baseline，并让每个 slice 使用相互独立的均匀
玩家排列，则对这些排列取条件期望有精确随机化 MSE：

\[
\boxed{
\mathbb E_{\{\pi_s\}\mid\mathrm{base}}
\|\widehat\phi-\phi\|_2^2
=\frac{\|Pa\|_2^2}{n-1}
\sum_s\|\bar E_s\|_F^2.
}
\]

若连 greedy base 的随机性一起平均，右侧还需外层
\(\mathbb E_{\mathrm{base}}\)。

因此逐 slice 的 Frobenius frame loss 直接等于 additive class 上的
条件随机化 MSE 系数，而不只是几何启发式。

---

## 8. 算法 C：与官方 ratio 聚合兼容的 cyclic orbit

固定 size \(s\)，选择 base block \(B\subseteq\mathbb Z_n\)，并生成

\[
\mathcal O(B)=\{B+r:r\in\mathbb Z_n\}.
\]

每条完整 orbit 有 \(n\) 个 coalition，并满足：

- 每个玩家恰出现 \(s\) 次；
- 每个玩家恰缺席 \(n-s\) 次；
- 一阶方向均值严格为零。

给 size \(s\) 分配 \(L_s\) 条 orbit 后，

\[
T_{i,s}^+=sL_s,\qquad
T_{i,s}^-=(n-s)L_s
\]

对所有玩家固定。因此官方条件均值的随机分母变成确定分母。

若总 block 数为 \(B_s=nL_s\)，则

\[
\frac1n
\left[
\overline v(i\in S)-\overline v(i\notin S)
\right]
=
\sqrt{\frac n{s(n-s)}}
\frac1{B_s}\sum_bv(S_b)(u_{S_b})_i.
\]

所以 cyclic orbit 下，官方 ratio 聚合与分层线性估计严格相等。

实现用循环自相关计算每个 candidate orbit 的 operator：

\[
O(B)=\frac1n\sum_{r=0}^{n-1}
u_{B+r}u_{B+r}^\top.
\]

候选选择不展开这个 \(n\times n\) 矩阵，而只保存 circulant first
row。每个候选的 first row 可用 FFT 在 \(O(n\log n)\) 时间求得，
greedy 增量利用
\(\langle\operatorname{Circ}(a),\operatorname{Circ}(b)\rangle
=n\langle a,b\rangle\) 在 \(O(n)\) 时间评分：

\[
2\langle A,O(B)\rangle+\|O(B)\|_F^2.
\]

输出 \(L\) 条完整 orbit 仍然必须物化 \(Ln\) 个长度为 \(n\) 的
coalitions，成本为 \(O(Ln^2)\)。最终 diagnostics 从累计 first rows
构造每层的小 operator，不再对全部 materialized rows 做
\(D^\top D\)。

---

## 9. exact design 与低阶 interaction

### BIBD

若固定 size 的 blocks 构成 \(2\)-design/BIBD，则

\[
\frac1B\sum_bu_{S_b}=0,\qquad
\frac1B\sum_bu_{S_b}u_{S_b}^\top=\frac1{n-1}P.
\]

于是所有 additive games 有限样本精确恢复。

但 BIBD 有必要整除条件：

\[
\frac{Bs}{n}\in\mathbb Z,\qquad
\frac{Bs(s-1)}{n(n-1)}\in\mathbb Z.
\]

不能假设任意 \((n,s,B)\) 都存在 exact design。

### 低阶充分条件

若 utility 是最高 \(r\) 阶 square-free polynomial，

\[
v(z)=\sum_{|A|\le r}h_A\prod_{j\in A}z_j,
\]

则 fixed-size 积分对象

\[
v(z)\left(z_i-\frac{s}{n}\right)
\]

最高为 \(r+1\) 阶。size-\(s\) slice 上，strength
\(\min(r+1,s)\) 的 combinatorial design 是精确积分的充分条件：

\[
\begin{array}{c|c}
\text{game degree}&\text{充分 design strength}\\
\hline
1&2\\
2&3\\
r&r+1
\end{array}
\]

这是充分条件，不一定必要。仅有二阶 tight frame 不能保证 pairwise
game 精确。

---

## 10. 二阶 frame 之外还要控制一阶均值

tight frame 不推出

\[
\sum_bu_{S_b}=0.
\]

反例：\(n=4,s=2\)，取

\[
\{1,2\},\{1,3\},\{1,4\}.
\]

三个方向构成 \(\mathbf1^\perp\) 的正交基，所以 \(H=P/3\)，但方向
均值不为零。

端点线性 baseline 使 additive identity 仍成立；对一般 utility，
slice 常数分量会通过非零一阶均值产生额外误差。因此 approximate
design 使用联合 loss：

\[
\gamma_1
\left\|\frac1B\sum_bu_{S_b}\right\|_2^2
+
\gamma_2
\left\|H_s-\frac1{n-1}P\right\|_F^2.
\]

当前实现的 greedy candidate score 同时包含 frame-potential 项和可调
的一阶平衡项。完整 cyclic orbit 自动实现一阶平衡。

二阶 frame 更好仍不保证一般 game 的 MSE 更小。一个精确反例是
\(n=5,T=4\)，令

\[
v(S)=
\begin{cases}
-1,&|S|=2,\\
+1,&|S|=3,\\
0,&\text{otherwise}.
\end{cases}
\]

真实 Shapley 向量为零。取 base batch

\[
\{2,4\},\quad\{1,4,5\},\quad\{2,5\},\quad\{3,4,5\},
\]

再对整个 batch 施加一个共同的均匀重标号。该 estimator 仍无偏，
但精确 MSE 为 \(5/4\)；同预算 IID OFA 的精确 MSE 为 \(5/6\)，
比值是 \(3/2\)。原因是二阶 frame 目标没有消除跨 size 的一阶方向
和相关性。这同时说明一阶平衡项是必要的，但仅靠软惩罚不能给出
universal dominance。

---

## 11. Baseline 的准确边界

因为 \(\mathbb E[u_S\mid |S|=s]=0\)，减去任意 size-only baseline
不改变线性估计器的期望。

端点线性 baseline

\[
\ell(s)=v(\varnothing)+\frac{s}{n}[v(N)-v(\varnothing)]
\]

对 additive game 特别合适，但不保证对一般 game 降低方差。

反例是 grand-unanimity game：

\[
v(S)=0\;(S\ne N),\qquad v(N)=M.
\]

不减 baseline 时所有 proper-coalition 更新均为零；减去 \(\ell(s)\)
后反而产生非零随机更新。

理想的 slice 常数 baseline 是

\[
b_s^*=\mathbb E[v(S)\mid |S|=s],
\]

但通常未知。实现允许选择 `baseline="linear"` 或 `"none"`。

---

## 12. 不能继承或夸大的结论

### 不能直接继承官方 IID concentration theorem

官方 OFA 的证明使用独立 coalitions。frame/orbit batch 中样本相关，
随机重标号只保证边际正确，不会恢复独立性。因此不能原样宣称相同的
Bernstein bound 或 query complexity。

严格做法是：

- 把一次完整 frame/orbit 看作一个统计单元；
- 使用多个独立随机重标号 batch；
- 在 batch 层重新证明 MSE 或 concentration。

### 不能声称对所有 games 普遍优于 IID OFA

在 endpoint-linear baseline（或严格一阶平衡）条件下，frame
discrepancy 严格控制 additive component。若

\[
v=v_{\mathrm{add}}+r,
\]

则

\[
\widehat\phi-\phi
=(F-\alpha P)a+
[\widehat\phi(r)-\phi(r)].
\]

高阶 residual 必须单独控制。对抗性的高阶 game 可以让任何固定
dependent design 比 IID 更差。

随机重标号还只保证**分布等变性**。若把一个已经 realization 的
design 和 game 同时按同一排列映射，估计值路径级等变；但对重标号后
的 game 用相同 seed 重新生成另一份未映射 design，一般不会得到同一
条路径的排列结果。

### 原官方 ratio estimator 不是有限样本严格无偏

一般 dependent design 不能只替换 `_generator()` 后继续宣称无偏。
若某个玩家/size stratum 没有样本，官方零填充会造成有限样本偏差。
在 IID sampling 下，条件于 stratum count \(>0\)，条件样本均值本身
仍然无偏，随机分母并不自动意味着 ratio bias。对任意 dependent
coupling，分母与被选 utility 的相关性可能再引入额外 ratio bias，
除非另外证明，或让 counts 固定。

以下两种方式是严格的：

1. 改为本文显式线性 coupled/stratified estimator；
2. 使用 1-balanced/BIBD/cyclic-orbit 设计，使所有分母固定。

---

## 13. 对官方代码的实现映射

官方 Shapley 配置实际使用 `OFA_fixed`；`OFA_optimal` 禁止 Shapley。

若直接做兼容扩展，建议新增 `OFA_frame(OFA_fixed)` 并重写
`sampling()`，而不是只改 `_generator()`，因为 dependence 是整个
batch 的性质。

- 一般 approximate frame：同时重写 `_process()`/`_estimate()` 为
  显式线性权重。
- 完整 cyclic orbit：可复用官方 `_process()`/`_estimate()`，因为
  in/out counts 固定。
- checkpoint 应落在完整独立 batch 或完整 orbit 后。

当前仓库采用 clean-room 实现：本地官方快照只用于核验，保持不变且
被版本控制排除。原因是官方仓库在固定 commit 上没有 LICENSE，
不能默认允许公开再分发，即便是未修改快照。

## 14. 核验来源

- Li and Yu, *One Sample Fits All*, NeurIPS 2024:  
  https://papers.nips.cc/paper_files/paper/2024/hash/6b295b08549c0441914e391651423477-Abstract-Conference.html
- 官方代码（核验 commit `f5e9da3f6f35f5fee2de1c89dbd779ed25c505cd`）:  
  https://github.com/watml/one-for-all
- Li et al., *Adalina: Adaptive Linear Approximation for the Shapley Value
  and Beyond*, 2026 preprint:  
  https://arxiv.org/abs/2604.08438

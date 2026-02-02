## Physical Analysis (Bullets)

* 水对殖民地是**库存型生命维持资源**：决定性不是“日消耗”，而是**(i) 入住前缓冲库存门槛**与**(ii) 回收效率<1带来的不可避免净损失**。这更适合用“存量–流量”守恒写清楚，而不是只写年需求数。
* 水与运输系统的耦合机制应明确为：水作为 Tier-3 大宗货物占用弧容量，挤占 Tier-1/2 “seed”从而降低你们 Layer-2 的增长斜率（机会成本/追加工期）。
* 你文稿里“85,000 t/yr”需要自洽：若居民总用水 (w_{\rm use}=20) L/(cap·d) 且回收 (\eta=0.98)，仅生活净补给约 (0.4) kg/(cap·d) ⇒ (14{,}600) t/yr（10万人）。要到 85k t/yr，必须来自农业/生态蒸散或更低回收率；建议把农业作为独立项参数化，而非一句话“dominant”。
* “Phase II 500,000 t 启动水”应作为**设计变量 (W_{\rm gate})**（安全性/生态需求不确定），做敏感性：不同 (W_{\rm gate}) 对入住年、追加成本的弹性。这样故事更“可证伪”。

---

## Model Formulation

### Governing Equations

**1) 需求分解（吨/日）**
设入住后人口 (N(t))，工业建设进度质量流 (\dot M_{\rm base}(t))（来自 Model I/II 的最优运输–建造结果）。定义三类“毛需求”：
[
C_{\rm dom}(t)= N(t),w_{\rm dom},\qquad
C_{\rm ag}(t)= N(t),w_{\rm ag},\qquad
C_{\rm ind}(t)= \kappa,\dot M_{\rm base}(t),
]
其中 (w_{\rm dom},w_{\rm ag}) 以 t/(cap·d)（或 L/(cap·d) 乘 (10^{-3})）计，(\kappa) 以 t(*{\rm water})/t(*{\rm processed}) 计（你原式应把“净需求”写成 ((1-\eta)) 形式，才量纲一致）。

**2) 回收–泄漏下的水库存守恒（吨/日）**（把“水是库存”写成白盒）
令 (W(t)) 为可用水库存（吨），回收效率 (\eta_{\rm bio},\eta_{\rm ind})，结构性泄漏/挥发为一阶损失 (\lambda W)。外部供给来自地球运输 (q_E(t)) 与月面 ISRU (q_L(t))：
[
\frac{dW}{dt}= q_E(t)+q_L(t);-;\Big[(1-\eta_{\rm bio})\big(C_{\rm dom}(t)+C_{\rm ag}(t)\big);+;(1-\eta_{\rm ind})C_{\rm ind}(t);+;\lambda W(t)\Big].
]
括号内即“不可避免净损失” (\ell(t))。**任务 3 的一年保障**等价于：在入住起始年 ([t_{\rm in},t_{\rm in}+365]) 内保证 (W(t)\ge 0) 且不中断。

**3) 入住门槛（生态充能 + 安全缓冲）**
把你文稿的“逻辑门”形式化为库存下界：
[
W(t_{\rm in}) ;\ge; W_{\rm gate},
]
其中 (W_{\rm gate}=W_{\rm eco}+W_{\rm buf})。你原来把土壤体积、含水率等写进公式是可以的，但建议收敛为两个可解释块，并把细节放附录（否则正文变量爆炸）。

**4) 与 TDMCNF（Model I）容量耦合：水挤占弧容量**
在时间展开网络的任一运输弧 ((i,j)) 与时段 (t)，令 (x^k_{ij,t}) 为货物 (k)（含水 (k=w)）的流量（吨/期）：
[
\sum_{k\in{\text{Tier1, Tier2, Tier3}, w}} x^k_{ij,t};\le;U_{ij,t},
]
且月面水到达量给定为网络流汇入：
[
q_E(t)= \sum_{(i\to \text{Moon})} x^{w}*{i,\text{Moon},t}.
]
这样“追加工期”的机制变成：若为满足 (W*{\rm gate}) 或一年净损失而提高 (x^w)，则同一容量约束下 Tier-1/2 的 (x^{\text{seed}}) 必然下降。

**5) 与 Layer-2 工业增长的“机会成本”桥（你现在叙事最缺的白盒）**
用一个最小耦合写清“水导致增长变慢”即可：令“seed 有效输入”
[
S_{\rm seed}(t)=S_{\rm seed}^{(0)}(t)-\chi,q_E^{\rm (pre)}(t),
]
其中 (q_E^{\rm (pre)}) 是入住前为攒库存而运的水，(\chi) 为“每吨水挤占的 seed 等效吨位”（通常 (\chi\approx1)，若有装载结构差异再修正）。把它代入你们已有的自复制增长 ODE（此处不重写细节，只标明耦合口）：
[
\dot{\mathcal I}(t)=f!\left(\mathcal I(t),,S_{\rm seed}(t)\right),\qquad
q_L(t)\le g!\left(\mathcal I(t)\right).
]
**追加工期**用“时间平移”定义最干净：找 (\Delta T) 使
[
M_{\rm base}^{(\text{water})}(t)\approx M_{\rm base}^{(0)}(t-\Delta T).
]

**6) 追加成本（任务 3 输出口）**
在你们成本结构上，任务 3 的“额外成本”拆成两项最清晰：
[
\Delta C = \sum_t c_E(t),q_E(t);+;F_{\rm ISRU},z;-;\text{(baseline 对应项)},
]
其中 (z\in{0,1}) 表示是否为水配置/提前部署 ISRU（固定费用 + 能力上限），与文献里“fixed-charge make-vs-buy”一致。

### Boundary/Initial Conditions

* 建造期：(N(t)=0)（robot-first），可取 (W(2050)=0)。
* 入住：(N(t)=10^5,\mathbf 1_{t\ge t_{\rm in}})，并施加 (W(t_{\rm in})\ge W_{\rm gate})。
* 一年保障窗口：对 (t\in[t_{\rm in},t_{\rm in}+365]) 强制 (W(t)\ge 0)（或 (W(t)\ge W_{\rm safety})）。

---

**你这段文字要“讲通故事”的三处关键润色**：第一，把 Phase I/II/III 的核心统一成“库存方程 + 门槛约束 + 容量挤占”三件事；第二，把 500,000 t 与 85,000 t/yr 都改为**参数化并校验量纲**（给出由 (w_{\rm dom},w_{\rm ag},\eta,\lambda) 推导出的数，而不是直接报数）；第三，图的叙事顺序改为：需求剖面 → 库存门槛可行性 → 挤占导致的 (\Delta T) 与 (\Delta C)（最后再谈鲁棒性）。

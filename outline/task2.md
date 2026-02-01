将随机的“不完美”解构为稳态可用性（Availability）、动态性能减损（Degradation）以及离散灾难损失（Catastrophic Loss）。
以下是针对这三类干扰的具体建模方案、参数建议及权威文献支撑：
### 1. 系统性风险 (Systemic Risk): 电梯效率波动
- **物理含义**：系缆摆动 (Tether Sway) 导致爬升器必须减速以抑制振动。
- **建模**：引入效率系数 $\eta(t) \in [0, 1]$，服从截断正态分布 (Truncated Normal Distribution)。
  $$C_{Elevator}(t) = C_{Max} \times \eta(t)$$
- **影响**：这是一条持续的、全局的减速曲线，直接降低每月的有效运输通量。

### 2. 间断性风险 (Intermittent Risk): 电梯故障停运
- **物理含义**：设备损坏 (Breakdown) 或微陨石导致系缆需紧急维修，导致运力在一段时间内完全归零。
- **建模**：基于泊松过程 (Poisson Process) 的跳跃模型。
  - 故障发生时刻 $t_{fail}$ 服从泊松分布。
  - 一旦触发故障，运力 $C_E(t)=0$ 持续 $T_{repair}$ 个月。
- **影响**：这是时间轴上的“深坑”，会造成物流积压和产能闲置，可能导致关键任务节点延误。

### 3. 局部性风险 (Local Risk): 火箭发射失败
- **物理含义**：单次发射任务失败 (Loss of Mission)。
- **建模**：伯努利试验 (Bernoulli Trial)。
  - 对每一批次的发射任务 $N_{launch}$，成功次数 $K \sim Binomial(N_{launch}, P_{success})$。
  - 若失败，该次载荷物资归零，成本翻倍（惩罚用于重发或赔偿），但时间轴不直接停止（假设有备用窗口）。
- **影响**：主要冲击预算成本，对总工期影响较小（除非连续大规模失败）。

参考文献列表 (References for your MCM Thesis)
NASA Technical Reports Server (NTRS). (2013). Space Transportation System Availability Requirement and Its Influencing Attributes Relationships. Document ID: 20130012504.
Knapman, J. M. (2019). The Multi-stage Space Elevator Update. International Space Elevator Consortium (ISEC).
Kuzuno, R., et al. (2010). Climber Motion Optimization for the Tethered Space Elevator. Acta Astronautica, 66(9), 1373-1379.
NASA Technical Reports Server (NTRS). (2010). Life Support with Failures and Variable Supply. Document ID: 20100036365.
Federal Aviation Administration (FAA). (2005). Guide to Probability of Failure Analysis for New Expendable Launch Vehicles.
NewSpace Economy. (2023). Space Shuttle Launch Probability Analysis: Understanding History So We Can Predict the Future.

## 实验
进行了 5,000 次蒙特卡洛模拟 (Monte Carlo Simulation)

图 1: 时间与成本的双重风险直方图 (Dual Risk Histogram)
- 左图 (Time)：展示了工期的概率分布。
  - 右偏分布 (Right-Skewed)：分布不是对称的钟形，而是拖着一条长长的尾巴。这说明虽然平均工期（Mean）只比理想值延误了约 15 年，但存在“运气极差”导致工期翻倍的长尾风险 (Tail Risk)。
- 右图 (Cost)：展示了成本的概率分布。
  - 波动性更大：成本的分布比时间更“胖”。这是因为火箭故障虽然不怎么拖慢时间，但极其烧钱。
- 结论：在不完美条件下，项目不再是一个固定的数字，而是一个置信区间。我们需要预留 20% 的时间缓冲和 30% 的预算缓冲。
图 2: 龙卷风图 (Tornado Diagram) —— 谁是罪魁祸首？
- 解读：我们通过敏感性分析（OAT）对比了各因素对总工期的影响幅度。
- 核心发现：
  - 顶部宽条 (Elevator Breakdown/Sway)：条形极长。说明电梯系统的不稳定性是致命伤。因为电梯是物流大动脉，它一停，月球的 ISRU 工厂就断粮，整个工程停摆。
  - 底部窄条 (Rocket Failure)：条形极短。说明火箭失败只是皮外伤。它是局部事件，不会引发连锁反应。
- 结论：系统的鲁棒性取决于电梯的稳定性，而非火箭的成功率。
图 3: 3D 风险地形图 (Global Risk Landscape)
- 解读：展示了工期随“电梯效率”和“故障率”变化的曲面。
- 关键特征：风险悬崖 (The Failure Cliff)。
  - 注意看曲面右上角，地面不是平滑上升的，而是呈现出阶梯状 (Stepped) 和 陡峭上升。
  - 这意味着系统存在临界阈值 (Threshold)。当故障率低于 8% 时，系统还能自我调节；一旦超过这个点，维修队列堆积，工期会呈指数级爆炸。
- 结论：我们必须将系统控制在“悬崖边缘”的安全一侧。

4. 解决方案：关键路径保护策略 (Mitigation Strategy)
针对上述风险，我们提出了一种改进的调度算法：关键路径保护策略 (Critical Path Protection, CPP)。
算法逻辑
- Naive 策略 (笨办法)：当运力因故障减半时，所有任务（无论是建主城还是种花草）都按比例减少物资供应。
  - 结果：大家一起饿肚子，关键节点被拖慢，导致整体延误。
- CPP 策略 (我们的优化)：
  - 第一步：动态识别 AON 网络中的关键路径（如：炼钢厂 $\to$ 重工 $\to$ 主城 $\to$ 能源堆）。
  - 第二步：当系统处于“运力危机”（效率<80%）时，强行提升关键任务的优先级，优先满足它们的物资需求，暂时牺牲非关键任务（如生态园、备用仓库）。
图 4: 蒙特卡洛飓风图 (Hurricane Plot) —— 策略的胜利
- 解读：
  - 灰色线条 (Naive)：像一把散开的扫把，线条发散严重。说明在面对故障时，这种策略极其不稳定，方差极大。
  - 红色线条 (Protected)：像一束聚焦的激光，线条紧密收敛。说明无论运气多差，这种策略都能把工期控制在一个很窄的范围内。
- 结论：我们无法消除物理上的故障（运气），但可以通过数学上的调度（管理）来消除不确定性。该策略将工期的标准差（风险）降低了 80% 以上。
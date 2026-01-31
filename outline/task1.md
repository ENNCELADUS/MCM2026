
第一问部分：
1. 基础的AON网络建模：
  形式化为：多模式资源受限项目调度问题 (Multi-Mode RCPSP)。
  全局属性： $$\vec{S}(t) = [P_{rod}(t), V_{build}(t), Pop_{cap}(t)]$$，分别表示月球基地的本地生产力、本地建造力和本地人口上限。
  1. AON网络节点 $$i$$任务包括的固有参数：
    1. 任务ID，任务内容，前导和后继；
    2. 物资需求量 $$M_i$$：分为地球 $$M^{Earth}_i$$、弹性 $$M^{Flexible}_i$$、月球 $$M^{Moon}_i$$三类
    3. 建设时间：是总物资量和安装施工人力的函数
    4. 运输时间（运力分配的函数）： $$T^{Trans}_i = \frac{M^{Earth}_i + k ·M^{Flexible}_i}{R_i}$$，其中 $$R_i$$是分配给该任务的总运力（包括了火箭和电梯），其中 $$k \in [0,1]$$决定了弹性物资多少由地球运过去。
    5. 属性增益向量（完成这个任务会对月球基地产生哪些正面buff）： $$\vec{\Delta}_i = [\Delta P, \Delta V, \Delta Pop]$$
  2. 节点包括的动态属性这些是模型求解出来的变量。
    - $$T_i$$：总完成时间。是三部分的最大值： $$T_i = \max \left( \underbrace{\frac{M^{Earth}_i}{R_i(t)}}_{\text{地球运输耗时}}, \quad \underbrace{\frac{M^{Moon}_i}{P_i(t)}}_{\text{月球生产耗时}}, \quad \underbrace{\frac{M^{Total}_i}{V_i}}_{\text{安装施工耗时}} \right)$$，包括地球运输时间、建设时间和月球生产时间。R, P, V分别表示分配给该任务的地球运力、月球生产力和月球施工能力(t/day)
    - $$ES_i$$ / $$EF_i$$：最早开始/结束时间。
    - $$LS_i$$ / $$LF_i$$：最晚开始/结束时间（用于计算总浮动时间 Total Float）。
    - $$Mode_i$$：运输模式（0=全电梯，1=全火箭，0~1=混合）。
      - 任务i的运力 $$R_i$$是运输模式、火箭全局运力、电梯全局运力的一个约束下的函数。
  - 约束条件：
    紧前关系约束 (Precedence Constraints)：
    $$S_i \ge S_j + D_j \quad \forall j \in Predecessors(i)$$
    (任务 $i$ 必须等前置任务 $j$ 做完才能开始)
    全球运力约束 (Global Transport Capacity)：
    在任意时刻 $t$，所有正在进行运输的任务占用的运力不能超过总运力。
    $$\sum_{i \in Active(t)} \text{R}_i(t) \le C_{R}(t) + C_{E}(t)$$
  月球产力约束
    在任意时刻 $t$，所有正在进行月球生产的任务占用的生产力不能超过总生产力。
    $$\sum_{i \in Active(t)} \text{P}_i(t) \le C_{M}(t)$$

2. 定义关键的AON链路：
  - 根据官方内容，提供一个恰当简化的AON网络（前导图法），作为月球基地建设的工程计划
  - （具体任务内容待研究，可以做成像游戏的成就树那样）。
[Image]
[Image]
[Image]
3. 求解最优的时间表
  - 算法待定，可能是一个启发式算法；
  - 待确定的：每个任务的弹性物资运输比例 $$k$$，运输模式 $$Mode$$，开始时间和结束时间
  - 它们受限于这些变量：时间 $$t$$时刻的总火箭运力、总电梯运力、月球产量。
  - 评价函数：
    - 时间；
    - 钱；
    - 环境保护；
    - 或者上面三者的某种加权。
  - 分别在强制全用电梯、强制全用火箭、混合状态的情况下求解。

 4. 一些细节
这部分是一些不那么关键的细节，以下是gemini完成的
以下是详细的公式化定义和参数列表：
3.1 全球火箭总运力 $C_R(t)$
假设：随着可重复使用技术（如 Starship）的成熟和发射场扩建，运力呈指数增长。
公式：
$$C_R(t) = C_{R\_0} \cdot (1 + r_{growth})^{(t - t_0)}$$
- $$C_{R\_0}$$：2050 年的初始火箭年运力（所有发射场总和）。
- $$r_{growth}$$：火箭运力年增长率。
- $$t_0$$：起始年份（2050）。
3.2 单位质量火箭运输成本 $Cost_{unit}^{Rocket}(t)$
假设：成本随时间下降，符合“学习曲线”模型，但存在一个物理底线（燃料和硬件折旧极限）。我们采用 指数衰减模型。
公式：
$$Cost_{unit}^{Rocket}(t) = (C_{rock\_start} - C_{rock\_min}) \cdot e^{-\lambda_c (t - t_0)} + C_{rock\_min}$$
- $$C_{rock\_start}$$：2050 年的单位运输成本（如 $100/kg）。
- $$C_{rock\_min}$$：理论最低成本底线（燃料费+最低维护，如 $10/kg）。
- $$\lambda_c$$：成本衰减系数（技术成熟速度）。
3.3 电梯运力 $C_E(t)$ 与物理上限
假设：电梯运力符合 S 形 Logistic 增长，但其上限 $C_{E\_max}$ 受到缆绳物理属性的硬约束。
公式：
$$C_E(t) = \frac{C_{E\_max}}{1 + e^{-k_E (t - t_{mid})}}$$
- $$k_E$$：电梯建设扩容速率。
- $$t_{mid}$$：运力快速增长的中点年份。
物理上限 $C_{E\_max}$ 的推导（物理约束细节）：
电梯的年最大吞吐量受限于缆绳的安全应力和爬升器运行模式。
$$C_{E\_max} = N_{tethers} \cdot \frac{m_{load} \cdot v_{climber}}{D_{safe}} \cdot T_{operation}$$
- $$N_{tethers}$$：缆绳总数（题目提及 3 个港口，每个港口 2 根，共 6 根）。
- $$m_{load}$$：单个爬升器的有效载荷。
- $$v_{climber}$$：爬升器平均速度。
- $$D_{safe}$$：安全间距（防止科里奥利力导致缆绳碰撞或共振的最小距离）。
- $$T_{operation}$$：年运营时间（秒）。
3.4 电梯成本 $Cost_{unit}^{Elevator}$
假设：运营成本主要为电力和固定维护，边际成本极低且视为常数。
公式：
$$Cost_{unit}^{Elevator} = C_{elev\_const}$$
3.5 总成本计算公式与参数表
总成本由三部分组成：火箭运输费、电梯运输费、月球生产/建设费。
总成本目标函数 (Total Cost)：
$$Z_{Cost} = \sum_{i \in Tasks} \left( Cost_i^{Trans} + Cost_i^{Prod} \right)$$
其中任务 $i$ 的运输成本拆解：
我们需要根据 $$k$$ 值和 $$Mode$$ 确定多少物资走了火箭，多少走了电梯。
令 $$M_i^{Rocket}$$ 为任务 $$i$$ 分配给火箭的质量，$M_i^{Elev}$ 为分配给电梯的质量。
- 若 $Mode=1$ (全火箭)：$M_i^{Rocket} = M_i^{Earth} + k \cdot M_i^{Flexible}$
- 若 $Mode=0$ (全电梯)：$M_i^{Elev} = M_i^{Earth} + k \cdot M_i^{Flexible}$
- 若 $Mode \in (0,1)$ (混合)：需要引入变量 $x_i \in [0,1]$ 代表运输分流比例。
为了简化，假设 $k$ 决定了弹性物资的去向，而 $M^{Earth}$ 强制走最优或指定路径。这里我们建议定义详细的质量分流公式：
$$Cost_i^{Trans} = \int_{S_i}^{F_i} \left[ \dot{m}_i^{R}(t) \cdot Cost_{unit}^{Rocket}(t) + \dot{m}_i^{E}(t) \cdot Cost_{unit}^{Elevator} \right] dt$$
(注：由于火箭成本随时间变化，严谨计算需要对时间积分。工程上可用任务开始时刻 $S_i$ 的成本近似)
近似公式：
$$Cost_i^{Trans} \approx M_i^{Rocket} \cdot Cost_{unit}^{Rocket}(S_i) + M_i^{Elev} \cdot Cost_{unit}^{Elevator}$$
现实中可获取的参数表 (Parameter Collection Table)：
This content is only supported in a Feishu Docs
3.6 月球生产力 $C_M(t)$
假设：月球总生产力不是时间的直接函数，而是已完成的工业类任务的函数。
我们定义一个集合 $I_{ind} \subset Tasks$，包含所有能提升产能的任务（如“建设冶炼厂”、“部署3D打印群”）。每个任务 $$j \in I_{ind}$$ 完成后，会贡献 $$\Delta P_j$$ 的产能。
公式：
$$C_M(t) = C_{M\_base} + \sum_{j \in I_{ind}} \mathbb{I}(t \ge EF_j) \cdot \Delta P_j$$
- $$\mathbb{I}(\cdot)$$：示性函数，任务 $j$ 结束 ($EF_j$) 后为 1，否则为 0。
- $$\Delta P_j$$：任务 $j$ 带来的产能增量（吨/天）。
- $$C_{M\_base}$$：初始产能（极小，可能是携带的微型设备）。
与建设进度 $\Phi(t)$ 的关联：
如果你希望用一个连续变量 $$\Phi \in [0,1]$$ 来简化：
$$C_M(t) = C_{M\_max} \cdot \Phi(t)$$
其中 $\Phi(t) = \frac{\sum_{j \in I_{ind}, Finished} Weight_j}{\sum_{All} Weight_j}$，即工业设施的加权完成度。
3.7 其他需要处理的细节 (Refinements)
A. 环境影响评价函数 $E(t)$
需要区分火箭和电梯的污染机制。
$$Env = \sum_{i} \left( M_i^{Rocket} \cdot e_{rock} + M_i^{Elev} \cdot e_{elev} \right)$$
- $$e_{rock}$$：火箭单位质量排放因子（高）。$CO_2$ + 黑碳 + 氧化铝颗粒。
- $$e_{elev}$$：电梯单位质量排放因子（极低）。主要是地面发电的碳足迹。
B. 弹性物资比例 $k$ 的决策变量化
在优化中，$k$ 不应该是一个固定常数，而是一个决策变量 $k_i$。
对于每一个任务 $i$，模型需要决定：
- 如果此时火箭运力有空余且便宜（后期），$k_i \to 1$（多运点，省月球产能）。
- 如果此时月球产能过剩，$k_i \to 0$（少运点，多生产）。
C. “安装施工耗时”的修正
你原来的公式中：$T = \max(\dots, \frac{M^{Total}}{V_i})$。
这里的 $$V_i$$ (施工速度) 建议也设为动态的：
$$V_i(t) = V_{base} \cdot (1 + \beta_{robot} \cdot \text{RobotCount}(t))$$
即：施工速度取决于运上去了多少个建筑机器人。这体现了滚雪球效应。

---
总结：你的公式清单 (Cheat Sheet)
在你的论文“模型建立”章节，可以直接罗列这组方程组：
1. 目标函数：Min $W_1 \cdot T_{total} + W_2 \cdot Z_{Cost} + W_3 \cdot Env$
2. 时间约束：AON 拓扑排序 + $Max$ 瓶颈公式。
3. 资源约束 I (Earth)：$\sum R_i(t) \le C_{R\_0}(1+r)^t + \frac{C_{Emax}}{1+e^{-k(t-t_{mid})}}$
4. 资源约束 II (Moon)：$\sum P_i(t) \le \sum_{finished\_factories} \Delta P$
5. 成本积分：$Cost = \int (Mass_{flow} \cdot UnitCost(t)) dt$
这套定义完全覆盖了题目要求，并且在物理和经济逻辑上都非常严谨。
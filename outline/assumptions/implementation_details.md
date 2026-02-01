# 模型实施细节参数 (Implementation Details & Parameters)

> **Warning**: 这些参数基于当前（2025-2026）的航天工程研究与技术预测，作为模型输入的“基准假设” (Baseline Assumptions)。在实际仿真中，建议对关键变量（如 ISRU 产率、发射成本）进行灵敏度分析。

### 1. 物料清单 (BOM) & 需求映射
*   **物资分类 (Tier System)**:
    *   **Tier 1 (精密种子 Class 1)**: 芯片、传感器 ($\chi \approx 1$). 初始 15% $\to$ <1%. **100% 地球运输**.
    *   **Tier 2 (工业中枢 Class 2)**: 电机、工具 ($\chi \in [0.3, 0.7]$). 初始 35% $\to$ ~4%. **混合来源**.
    *   **Tier 3 (基础资源 Class 3)**: 结构、氧气、水 ($\chi \to 0$). 初始 50% $\to$ >95%. **100% ISRU**.
*   **总量约束**: 总需求 $10^8$ 吨 (Cumulative).

### 2. 物流网络定义 (Logistics)
*   **火箭运力 ($U_a$)**: 逻辑回归增长. 2050年单次 $L_{cap} \approx 100 \sim 150$ MT. 极限 $L_{max} \approx 250$ MT.
*   **发射成本 ($c_{a,r,t}$)**: 固定弧成本 2050年 $\approx 100,000$ USD/kg；若启用学习曲线则按年衰减 (见 §5 Wright's Law).
*   **太空电梯成本**: 2050年 $\approx 20,000$ USD/kg.
*   **太空电梯容量**: 固定 $F_{elevator} = 179,000$ 吨/年 (等效 $C_E \approx 0.00567607$ 吨/秒).
*   **环境约束**: 加权排放指数 (WEI)，黑碳 (BC) 权重 500.

## 1. 产能增长模型 (Continuous Capacity Growth Model)

为了简化离散的任务调度复杂度，本模型采用**连续增长曲线 (Continuous Growth Curve)** 来近似月球基地的产能扩张。产能 $P(t)$ 定义为每年通过 ISRU 生产建设材料的能力 (吨/年)。

### 1.1 阶段 I：引导期 (Bootstrapping Phase)
*   **特征**: 依赖地球补给，产能呈线性增长。此阶段月球仅仅是“接收端”，机器人负责组装来自地球的预制件。
*   **时间跨度**: 初期 (2050 - ~2055/2060)。
*   **数学模型 (离散步长，按月)**: 
    $$P_t \le P_{t-1} + \beta \cdot M_{earth,t} - \phi \cdot P_{t-1}$$
*   **参数**:
    *   $P_0$: 初始产能 (默认 $0.1$ t/yr)。
    *   $\beta$: 地球设备转化为产能的系数 (t/yr per ton)。
    *   $M_{earth,t}$: 当期到达月面的 Tier-1 设备质量 (tons/step)。
    *   $\phi$: 年度折旧率，按 $\phi/steps\_per\_year$ 折算到月度。
    *   **瓶颈**: 运力成本与地球发射频率。

### 1.2 阶段 II：指数爆发期 (Self-Replication Phase)
*   **特征**: 机器人开始建造机器人（利用月壤提取铝、铁、硅）。此时产能进入指数增长。
*   **触发条件**: 当本地生产成本 < 地球运输成本，且具备完整的 ISRU 闭环能力。
*   **数学模型 (当前 MILP 口径，离散步长)**:
    $$P_t \le P_{t-1} + \beta \cdot M_{earth,t} + \beta \eta \cdot \Delta G_{t-1} - \phi \cdot P_{t-1}$$
*   **参数**:
    *   $\beta$: 设备产能转化率 (Equipment Leverage) $= 50.0$ (t/yr capacity per ton equipment).
    *   $\eta$: 资源转化效率 (ISRU Efficiency) $= 0.90$.
    *   $\Delta G_{t-1}$: 上一期用于扩产的“本地设备投资” (tons/step).
    *   **说明**: 经典连续形式 $P(t) \propto e^{(\eta \alpha)t}$ 在当前 MILP 中未显式使用，$\alpha$ 仅作为未来替代方案保留。

### 1.3 阶段 III：环境极限期 (Saturation Phase)
*   **特征**: 受限于月球南极采光面、散热极限或特定稀有元素（如挥发物）的枯竭。
*   **数学模型** (修正后的逻辑斯谛方程):
    $$\frac{dP}{dt} = r P \left( 1 - \frac{P}{K(A)} \right) - \phi(D)$$
*   **参数**:
    *   $K(A)$: 动态环境承载力 (Carrying Capacity)，随技术进步 $A$ 移动。
    *   $\phi(D)$: 损耗项 (Depreciation)，代表宇宙射线损伤、月尘磨损导致的报废率。
*   **离散化说明**: 实现中采用线性折旧 $P_t = P_{t-1} - (\phi/steps\_per\_year) P_{t-1}$，并施加硬上限 $P_t \le K$。

## 2. 物流网络定义 (Logistics Network Parameters)

### 2.1 节点集合 $\mathcal{N}$
*   `Node_Earth`: 地球表面/低地轨道 (汇集点)
*   `Node_LEO`: 若有中转需求
*   `Node_LLO`: 低月球轨道 (Gateway/Depot)
*   `Node_Moon`: 月球表面 (Shackleton Crater)

### 2.2 弧段参数 $\mathcal{A}$

| 弧段 (Arc) | 起点 $\to$ 终点 | 典型载具 | 提前期 ($L_a$) | 单次载荷 ($U_a$) | 成本估算 ($C_{a,t}$) | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Arc_Launch` | Earth $\to$ LLO | Starship (Heavy) | 5 Days | 150t | \$100,000/kg (2050) | 直飞月球轨道模式 |
| `Arc_Landing` | LLO $\to$ Moon | Starship HLS / Lander | 1 Day | 100t (Downmass) | Inc. in Launch | 着陆消耗大量 $\Delta v$ |
| `Arc_Elevator` | Earth $\to$ Geo $\to$ Moon | Space Elevator | 7 Days | Cont. Flow | \$20,000/kg (Ops) | 高固定投入，低变动成本 |
| `Arc_ISRU_Log` | Moon $\to$ Moon | Surface Rovers | 0 Days | N/A | Local Energy | 本地短途运输 |

*   **注**: 模型可简化为 `Earth -> Moon` 的单一等效弧段，Lead Time 取 **5 Days** (Rocket) 或 **12 Days** (Elevator+Transfer).

## 3. 处理能力 (Handling Capacity)

为了配合连续增长模型，处理能力 $H(t)$ (挖掘/移动/着陆场吞吐量) 被假定为与生产能力 $P(t)$ 线性相关。
*   **假设**: 每 1 吨生产能力的维持，需要配比 $k_H$ 吨的处理能力。
*   **公式**: $H(t) \approx k_H \cdot P(t)$。
*   **挖掘机效率**: 1吨月面挖掘设备 = ~6,000吨/月 挖掘能力。

## 4. 全局 BOM 映射 (BOM Mapping)

针对 1 亿吨 ($10^8$ MT) 的总需求，我们在模型中进行如下分配：

1.  **基础设施 (Static Mass)**: 40%
    *   主要为辐射屏蔽层 (Regolith Sintering)、着陆垫、道路。
    *   **来源**: 99.9% ISRU (Tier 3)。
2.  **结构与外壳 (Structures)**: 30%
    *   加压居住舱外壳、工厂厂房。
    *   **来源**: 初期 50% Earth (Tier 2/3)，后期 90% ISRU (Al/Fe/Glass)。
3.  **耗材与流体 (Consumables)**: 20%
    *   水、氧气、推进剂、农业用土。
    *   **来源**: 100% ISRU (Ice mining / Regolith $O_2$).
4.  **高精设备 (High-Tech)**: 10%
    *   芯片、机器人、医疗设备、核反应堆芯。
    *   **来源**: 90% Earth (Tier 1)，10% ISRU (Tier 2 assembly)。
5.  **水资源需求 (Water Demand)**:
    *   人均需求: $0.5$ 吨/人/月 (包含循环损耗补充).
    *   **来源**: 100% ISRU (Ice mining).

**建模线性化建议 (Linearization for Optimization)**:
*   $M_{total} = \sum W_i + \sum M_{i,r}$。
*   绝大部分质量 ($W_i$) 是本地的土方工程 (Regolith Moving)。
*   真正的物流瓶颈在于 Tier 1 (设备) 和 Tier 2 (精密结构) 的运输。

## 5. 变量域与优化参数 (Optimization Bounds)

*   **Big-M**: 取 $10^9$ (大于总资源量)。
*   **Time Horizon**: $T_{max} = 1200$ Months (100 Years, 2050-2150).
*   **Integer Cuts**: 任务完成 $u_{i,t}$ 为 Binary；火箭发射次数 $y_{a,t}$ 为 Integer。

## Additional parameters

1. Time Discretization
Assumption: 1-Month Steps ($t \in [0, 1200]$).
Reasoning: Logistics lead times for Rockets (~5 days) and Elevators (~7 days) are sub-monthly. We aggregate them into monthly buckets. Any flow $x_{t}$ initiated in month $t$ arrives in month $t$ (if lead < 15 days) or $t+1$. This keeps the MILP solvable.
2. Arc Selection
Assumption: Simplified Single Arc (Earth $\to$ Moon).
Reasoning: The multi-node graph (Earth $\to$ LLO $\to$ Moon) adds unnecessary variables. We define Arc_Rocket and Arc_Elevator directly from Earth to Moon, with effective costs and lead times that account for the intermediate staging.
3. Elevator Payload Model (Stream Model)
Assumption: Continuous Throughput Limit (Flow Constraint).
Formula: $\sum_{r} x_{elev, r, t} \le C_E(t) \cdot \Delta t$, where $C_E(t)$ is in tons/second and $\Delta t$ is seconds/step.
Reasoning: Elevators operate as a "pipeline". We do not track individual integer climbers ($y_{a,t}$) in the global optimization, as they number in the thousands. Capacity $C_E(t)$ is the dynamic constraint.
4. Rocket Capacity Growth (Launch-Rate + Logistic Payload Model)
Assumption: Decoupled Capacity and Frequency.
Reasoning: Physical constraints limit individual rocket size ($L_{cap}$), while operational capabilities limit launch frequency ($N_{rate}$).
Model:
*   **Payload per Launch ($L_{cap}(t)$)**: Follows a Logistic Growth curve (as defined in `rocket-lift.md`), approaching a physical limit ($L_{max} \approx 250$t) due to $I_{sp}$ and material limits.
    *   $$L_{cap}(t) = \frac{L_{max}}{1 + A \cdot e^{-k(t - t_0)}}$$
    *   $L_{max} = 250$ MT, $L_{cap}(2050) \approx 150$ MT.
*   **Launch Rate ($N_{rate}(t)$)**: Modeled separately as the operational bottleneck (e.g., launches/year), which may grow linearly or similarly curve off.
    *   $N_{rate}(t)$ acts as an integer decision variable or controlled parameter (e.g., Max 10000 launches/year).
*   **Total Capacity**: $Total(t) = N_{rate}(t) \times L_{cap}(t)$.
5. Wright’s Law Cost Curve
Assumption: Annual Power Decay (Wright's Law, yearly).
Parameters:
$C_{base} = $870,600/kg$ (2024 Baseline).
$C_{min} = $20,000/kg$.
$d = 0.055$ (annual_decay_rate).
Formula: $Cost(y) = \\max\\{C_{min},\\; C_{base} \\cdot (1-d)^{(y-2024)}\\}$.
6. ISRU Bootstrapping (Continuous Input)
Assumption: Linear & Endogenous Replication.
Logic: We replace the "Stepwise Function" with a continuous flow model.
*   **Phase I**: $\Delta P \propto \Delta M_{Earth}$. Global capacity rises linearly as machinery arrives.
*   **Phase II**: $\Delta P \propto \Delta G_{t-1}$. Local investment drives capacity expansion via $\beta \eta$.
*   Transition: Occurs when $P(t)$ reaches critical mass for self-replication (e.g., capable of producing Tier 2 components).

7. Phase Durations
Assumption: Endogenous Transition.
Logic: We do not set fixed task durations (e.g., "6 months"). Instead, the duration of each phase is determined by the *Growth Rate*.
*   $T_{Phase\_I} = P_{critical} / Rate_{shipping}$.
*   $T_{Phase\_II}$: No closed-form in the discrete MILP; use simulation outputs or define an effective growth rate $g_{eff}$ if applying a continuous approximation.

8. Handling Capacity ($H(t)$)
Assumption: Proportional to Production.
Logic: $H(t) = k \cdot P(t)$. We assume the ratio of "Diggers" to "Processors" remains optimal throughout the growth curve.

9. Inventory Policies
Assumption: Continuous Flow (JIT Approximation).
Logic: Since we model long-term capacity ($P(t)$), we assume material buffers ($I(t)$) are transient. The constraint is on *Flow Rate* rather than *Stockpile*.

10. Energy vs. Transport Pivot
Assumption: Pivot Condition = Cost Cross-over.
Quantification: The pivot occurs when Local Production Cost < Transport Cost.
*   Transport Cost: $Cost_{Trans}(t)$ decreases via Wright's Law.
*   Local Cost: Fixed Energy Cost ($\approx \$0.10/kWh \times 500 kWh/kg \approx \$50/kg$).
*   **Result**: When Transport Cost < \$50/kg (e.g., Space Elevator era), Earth inputs might surge again, or conversely, if Transport > \$50/kg, ISRU is preferred. This dynamic decides the mix.

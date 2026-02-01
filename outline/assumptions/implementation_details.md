# 模型实施细节参数 (Implementation Details & Parameters)

> **Warning**: 这些参数基于当前（2025-2026）的航天工程研究与技术预测，作为模型输入的“基准假设” (Baseline Assumptions)。在实际仿真中，建议对关键变量（如发射成本）进行灵敏度分析。

### 1. 物料清单 (BOM) & 需求映射
*   **物资分类 (Tier System)**:
    *   **Tier 1 (精密种子 Class 1)**: 芯片、传感器 ($\chi \approx 1$). 初始 15% $\to$ <1%. **100% 地球运输**.
    *   **Tier 2 (工业中枢 Class 2)**: 电机、工具 ($\chi \in [0.3, 0.7]$). 初始 35% $\to$ ~4%. **混合来源**.
    *   **Tier 3 (基础资源 Class 3)**: 结构、氧气、水 ($\chi \to 0$). 初始 50% $\to$ >95%. **100% ISRU**.
*   **总量约束**: 总需求 $10^8$ 吨 (Cumulative).

### 2. 物流网络定义 (Logistics)
*   **火箭运力 ($U_a$)**: 逻辑回归增长. 2050年单次 $L_{cap} \approx 100 \sim 150$ MT. 极限 $L_{max} \approx 250$ MT.
*   **发射成本 ($c_{a,r,t}$)**: 2050年 $C_{base} = 88.90$ USD/kg，若启用学习曲线则按月度衰减 (见 §5 Wright's Law).
*   **太空电梯成本**: 指数衰减模型。2050年初始成本 $C_{start} = 50.0$ USD/kg，最低成本 $C_{min} = 5.0$ USD/kg，月度衰减率 $\lambda = 0.005$ (年度 $\approx 0.06$)。
    *   公式: $C(t) = (C_{start} - C_{min}) \cdot e^{-\lambda t} + C_{min}$
*   **太空电梯容量**: Logistic S 曲线增长模型，受缆绳物理约束。
    *   初始容量 (2050): $C_{E,ref} = 537,000$ 吨/年 (179,000 × 3 个港口)
    *   物理上限: $C_{E,max} \approx 1,000,000$ 吨/年 (1 Mt/y)
    *   公式: $C_E(t) = \frac{C_{E,max}}{1 + A \cdot e^{-k_E (year - t_0)}}$，其中 $A$ 由初始条件求解
    *   **物理约束推导** (Physical Upper Limit):
        *   $C_{E,max} = N_{tethers} \cdot \frac{m_{load} \cdot v_{climber}}{D_{safe}} \cdot T_{operation}$
        *   $N_{tethers} = 6$ (3 个港口 × 2 根缆绳)
        *   $m_{load} = 100$ 吨 (单个爬升器载荷)
        *   $v_{climber} = 200$ km/h $\approx 55.56$ m/s (爬升器平均速度)
        *   $D_{safe} = 1000$ km (安全间距，防止科里奥利力导致碰撞)
        *   $T_{operation} = 0.95 \times 365 \times 24 \times 3600$ 秒/年 (95% 运营时间)
        *   计算: $C_{E,max} = 6 \times \frac{100 \times 55.56}{10^6} \times (0.95 \times 31536000) \approx 1,000,000$ t/y

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
    $$P_t \le P_{t-1} + \beta \cdot M_{earth,t} + \alpha \cdot \Delta G_{t-1} - \phi \cdot P_{t-1}$$
*   **参数**:
    *   $\beta$: 设备产能转化率 (Equipment Leverage) (t/yr capacity per ton equipment).
    *   $\alpha$: 本地扩产转化率 (t/yr capacity per ton local investment).
    *   $\Delta G_{t-1}$: 上一期用于扩产的“本地设备投资” (tons/step).
    *   **说明**: 当前 MILP 使用线性近似（不显式建模指数增长）。

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
*   `E_port_Agg`: 地球港口汇聚点（电梯起点）
*   `Launch_Agg`: 地球发射汇聚点（火箭起点）
*   `Moon`: 月球表面

### 2.2 弧段参数 $\mathcal{A}$

| 弧段 (Arc) | 起点 $\to$ 终点 | 提前期 ($L_a$) | 单次载荷 ($U_a$) | 成本估算 ($C_{a,t}$) | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Arc_Elevator` | E_port_Agg $\to$ Moon | 7 Days | 由电梯年吞吐量折算 | \$500,000/kg (2050) | 固定吞吐流模型 |
| `Arc_Rocket` | Launch_Agg $\to$ Moon | 5 Days | 150t (2050) | \$88.90/kg (2050) | 载荷与成本按学习曲线 |

## 3. 处理能力 (Handling Capacity)

为了配合连续增长模型，处理能力 $H(t)$ 被假定为与生产能力 $P(t)$ 线性相关。
*   **公式**: $H(t) \approx k_H \cdot P(t)$。

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
Assumption: Monthly exponential decay (learning curve).
Parameters:
$C_{base} = 88.90$ USD/kg (2050 Baseline).
$C_{min} = 80.82$ USD/kg.
$\\lambda = 0.015001$ (monthly decay rate).
Formula: $Cost(t) = \\max\\{C_{min},\\; (C_{base}-C_{min}) e^{-\\lambda t} + C_{min}\\}$.
6. ISRU Bootstrapping (Continuous Input)
Assumption: Linear & Endogenous Replication.
Logic: We replace the "Stepwise Function" with a continuous flow model.
*   **Phase I**: $\Delta P \propto \Delta M_{Earth}$. Global capacity rises linearly as machinery arrives.
*   **Phase II**: $\Delta P \propto \Delta G_{t-1}$. Local investment drives capacity expansion via $\alpha$.
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

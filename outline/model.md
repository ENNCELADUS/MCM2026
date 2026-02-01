# 模型建模 (Model Formulation)

## 1. 物理分析 (Physical Analysis)

*   **系统架构**：这是一个**混合系统动力学（System Dynamics）与网络流（Network Flow）**模型。
*   **核心耦合**：
    *   **上层（产能演化）**：基于微分方程（或其离散化形式）的连续增长模型，决定月球基地的**生产能力 (P)**、**处理能力 (H)** 和 **承载力 (Pop)**。
    *   **下层（物流网络）**：受制于能力约束的时空网络流，决定物资运输分配以维持或加速上层的增长。

---

## 2. 符号定义 (Notation)

所有参数与变量均对应代码中的实现 (`src/optimization.py`, `src/config/constants.yaml`)。

### 索引集合 (Sets)
| 符号 | 代码对应 | 定义 |
| :--- | :--- | :--- |
| $t \in \mathcal{T}$ | `m.T` | 时间步索引 $[0, T_{horizon}-1]$ |
| $n \in \mathcal{N}$ | `m.N` | 节点集合 (Earth, Moon) |
| $a \in \mathcal{A}$ | `m.A` | 弧段集合 (Rocket, Elevator) |
| $r \in \mathcal{R}$ | `m.R` | 物资资源集合 (Tier 1/2/3) |
| $\Phi \in \{I, II, III\}$ | - | 增长阶段集合 (Bootstrapping, Replication, Saturation) |

### 关键参数 (Parameters)
| 符号 | 代码对应 | 定义 |
| :--- | :--- | :--- |
| $\eta$ | `isru_efficiency` | ISRU 闭环转化效率 (每吨设施产出多少吨新设施/年) |
| $\alpha$ | `replication_factor` | 自我复制因子 |
| $K$ | `carrying_capacity` | 环境/能源承载力上限 (t/yr) |
| $H_{ratio}$ | `handling_ratio` | 单位生产能力所需的配套处理能力 ($H \approx k \cdot P$) |
| $Cost_{Trans}(t)$ | `arc_cost` | 地月运输成本 (随 Wright's Law 衰减) |
| $Cost_{Local}$ | - | 本地生产边际成本 (能源 + 损耗) |

### 决策变量 (Decision Variables)
| 符号 | 类型 | 定义 |
| :--- | :--- | :--- |
| $x_{a,r,t}$ | Continuous | 物流网络流量 (运输量) |
| $P_t, H_t$ | Continuous | **状态变量**: $t$ 时刻的月面生产与处理能力 |
| $I^M_{r,t}$ | Continuous | 月面物资库存 |
| $\delta^{Growth}_t$ | Continuous | $t$ 时刻投入产能扩建的资源量 (Reinvestment) |
| $\delta^{City}_t$ | Continuous | $t$ 时刻投入城市建设的资源量 (Consumption) |

---

## 3. 模型构建 (Model Formulation)

### 3.1 产能演化动力学 (Capacity Dynamics)

模型不再通过离散任务 ($u_{i,t}$) 升级能力，而是遵循以下**状态转移方程**：

#### 阶段 I：引导期 (Bootstrapping)
在此阶段，产能增长完全依赖地球输入的“种子设施” ($M_{Earth}$)。
$$ P_{t+1} = P_t + \beta \cdot \text{Inflow}^{Tier1}_{t} $$
*   $\text{Inflow}^{Tier1}_{t}$：从地球运抵的高精设备量。
*   $\beta$：单位设备转化系数。

#### 阶段 II：指数爆发期 (Self-Replication)
当具备本地闭环能力后，产能增长来源于**再投资** ($\delta^{Growth}$)。
$$ P_{t+1} = P_t + \eta \cdot \delta^{Growth}_t $$
约束：
1.  **原料限制**：$\delta^{Growth}_t \le I^M_{Tier3, t}$ (需消耗本地建材)
2.  **产能限制**：再投资量不能超过当前产能的剩余部分。

#### 阶段 III：饱和期 (Saturation)
引入环境阻尼项 $\phi(P)$：
$$ P_{t+1} = P_t + \eta \cdot \delta^{Growth}_t - \text{Decay}(P_t) $$
且 $P_t \le K$ (总上限)。

### 3.2 供需平衡 (Supply & Demand)

#### 生产侧
$$ \text{Total\_Output}_t \le P_t \cdot \Delta t $$
产出被分配为两部分：
$$ \text{Total\_Output}_t = \delta^{Growth}_t (\text{扩产}) + \delta^{City}_t (\text{城建}) $$

*   $\delta^{Growth}_t$：进入正反馈循环，加速下期 $P$ 增长。
*   $\delta^{City}_t$：用于满足 10 万人城市的建设需求 (最终目标)。

### 3.3 目标函数

$$ \min \ \left( w_C \cdot \sum \text{TransportCost} + w_T \cdot T_{completion} \right) $$

*   **Completion**: 定义为累积 $\sum \delta^{City}_t \ge 10^8$ (一亿吨城市建成) 的时刻。
*   **Trade-off**:
    *   早期多运设备 ($\uparrow$ Cost) $\to$ 缩短 Phase I $\to$ 早进入指数增长 $\to$ $\downarrow$ Time。
    *   晚期多运成品 ($\uparrow$ Cost) $\to$ 直接增加 $\delta^{City}$。

### 3.4 线性化处理 (optimization.py 实现注记)
为了保留 MILP 的求解优势：
1.  **分段线性化**: 将 Phase II 的指数增长 $P_{t+1} = (1+r)P_t$ 建模为线性约束。
2.  **状态变量**: $P_t$ 作为显式连续变量。
3.  **模式切换**: 引入 Binary 变量 $z_{PhaseII, t}$ 指示当前是否处于自我复制模式（通常当 $Cost_{Local} < Cost_{Trans}$ 时激活）。
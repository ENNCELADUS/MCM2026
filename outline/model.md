# 模型建模 (Model Formulation)

## 1. 物理分析 (Physical Analysis)

*   **系统架构**：这是一个**离散时间（Discrete-Time）**、**多货类（Multi-Commodity）**、**混合整数线性规划（MILP）**模型。
*   **核心耦合**：
    *   **上层（任务网）**：AON/RCPSP 任务网络，决定能力（Capacity）的阶跃式增长。
    *   **下层（物流网）**：时空网络流（Time-Expanded Network），决定物资运输与分配，受限于上层提供的能力。
*   **时间离散化**：时间步长 $\Delta t$（默认 1 个月），总时长 $T_{max}$。
*   **物流守恒**：严格遵守质量守恒，物资从地球出发 -> 运输延迟 -> 月面缓冲区 -> handling 验收 -> 可用库存 -> 任务消耗。

---

## 2. 符号定义 (Notation)

所有参数与变量均对应代码中的实现 (`src/optimization.py`, `src/config/constants.yaml`)。

### 索引集合 (Sets)
| 符号 | 代码对应 | 定义 |
| :--- | :--- | :--- |
| $t \in \mathcal{T}$ | `m.T` | 时间步索引 $[0, T_{horizon}-1]$ |
| $n \in \mathcal{N}$ | `m.N` | 节点集合 (Earth Launch Sites, Earth Ports, Moon) |
| $a \in \mathcal{A}$ | `m.A` | 弧段集合 (Rocket, Elevator, Transfer, Ground) |
| $r \in \mathcal{R}$ | `m.R` | 物资资源集合 (Structure, Equipment, Fuel, etc.) |
| $i \in \mathcal{I}$ | `m.I` | 任务集合 (WBS Tasks) |
| $\mathcal{I}_V \subset \mathcal{I}$ | `m.I_V` | 能力构建任务集 (Capability Tasks) |
| $\mathcal{I}_{nonV} \subset \mathcal{I}$ | `m.I_nonV` | 普通建设任务集 (Construction Tasks) |

### 参数 (Parameters)
| 符号 | 代码对应 | 定义 |
| :--- | :--- | :--- |
| $\Delta t$ | `delta_t` | 单步时长 (s) |
| $L_a$ | `arc_lead[a]` | 弧段运输提前期 (steps) |
| $cost_{a,t}$ | `arc_cost[a,t]` | 弧段 $a$ 在 $t$ 时刻的单位运输成本 ($/kg) |
| $W_i$ | `W[i]` | 任务 $i$ 的总工作量 (kg) |
| $M^{Earth}_{i,r}$ | `M_earth[i,r]` | 任务 $i$ 必需的地球物资需求总量 (kg) |
| $M^{Moon}_{i,r}$ | `M_moon[i,r]` | 任务 $i$ 必需的月面物资需求总量 (kg) |
| $M^{Flex}_{i,r}$ | `M_flex[i,r]` | 任务 $i$ 的弹性物资需求总量 (kg) |
| $\Delta P_i, \Delta V_i, \dots$ | `delta_P[i]`, etc. | 任务 $i$ 完成后的能力增量 |
| $d_{install}$ | `d_install` | 重资产安装调试延迟 (steps) |
| $Rate^{Rocket}_{max}(t)$ | `rocket_launch_rate[t]` | 火箭最大发射频次限制 (launches/step) |
| $Cap^{Rocket}(t)$ | `rocket_payload[t]` | 单枚火箭有效载荷 (kg) |
| $Cap^{Elev}_{max}$ | `elevator_capacity_upper_kg_s` | 电梯系统运力上限 (kg/s) |
| $H_0, V_0, P_0, \dots$ | `initial_capacities` | 初始能力水平 |

### 决策变量 (Decision Variables)
| 符号 | 代码对应 | 类型 | 定义 |
| :--- | :--- | :--- | :--- |
| $x_{a,r,t}$ | `m.x[a,r,t]` | Continuous | $t$ 时刻进入弧段 $a$ 的物资 $r$ 流量 (kg) |
| $y_{a,t}$ | `m.y[a,t]` | Integer | $t$ 时刻弧段 $a$ 启用的运输工具数量 (count) |
| $N_{rate}(t)$ | `m.N_rate[t]` | Integer | $t$ 时刻的实际火箭发射总次数 |
| $u_{i,t}$ | `m.u[i,t]` | Binary | 任务 $i$ 在 $t$ 时刻**是否已完成** (Cumulative status) |
| $u^{done}_{i,t}$ | `m.u_done[i,t]` | Binary | 任务 $i$ **刚好在** $t$ 时刻完成 (Pulse) |
| $z_{i,t}$ | `m.z[i,t]` | Binary | 任务 $i$ 在 $t$ 时刻是否正在进行 (Active) |
| $v_{i,t}$ | `m.v[i,t]` | Continuous | 任务 $i$ 在 $t$ 时刻完成的工作量 (kg) |
| $q^E_{i,r,t}, q^M_{i,r,t}$ | `m.q_E`, `m.q_M` | Continuous | 任务 $i$ 在 $t$ 时刻消耗的 地球/月面 物资 (kg) |
| $Q_{r,t}$ | `m.Q[r,t]` | Continuous | 月面 ISRU 在 $t$ 时刻的产量 (kg) |
| $A^E_{r,t}$ | `m.A_E[r,t]` | Continuous | 月面在 $t$ 时刻验收 (Handling) 的地球物资量 (kg) |
| $I^E_{n,r,t}, I^M_{r,t}$ | `m.I_E`, `m.I_M` | Continuous | 各节点库存水平 (kg) |
| $B^E_{r,t}$ | `m.B_E[r,t]` | Continuous | 月面到货缓冲区库存 (kg) |
| $P_t, V_t, H_t, \dots$ | `m.P`, `m.V`, ... | Continuous | $t$ 时刻的各项实际能力水平 (State) |

---

## 3. 模型构建 (Model Formulation)

### 3.1 任务网络与能力演化 (Level 1)

#### 任务完成逻辑
*   **状态关联**：$u_{i,t} = \sum_{\tau=0}^t u^{done}_{i,\tau}$ (累积完成状态)
*   **唯一完成**：$\sum_t u^{done}_{i,t} = 1$ (每个任务必须且只能完成一次)
*   **全部完成**：$u_{i, T_{end}} \ge 1$ (所有任务必须在规划期结束前完成)

#### 能力动态更新 (Capacity Evolution)
能力值由**已完成**的任务累积决定，考虑安装延迟 $d_{install}$：
$$
\begin{aligned}
P_t &= P_0 + \sum_{i \in \mathcal{I}} \Delta P_i \cdot u_{i, t-d_{install}} \\
V_t &= V_0 + \sum_{i \in \mathcal{I}_V} \Delta V_i \cdot u_{i, t-d_{install}} \\
H_t &= H_0 + \sum_{i \in \mathcal{I}} \Delta H_i \cdot u_{i, t-d_{install}} \\
Pop_t &= Pop_0 + \sum_{i \in \mathcal{I}} \Delta Pop_i \cdot u_{i, t-d_{install}} \\
Power_t &= Power_0 + \sum_{i \in \mathcal{I}} \Delta Power_i \cdot u_{i, t-d_{install}}
\end{aligned}
$$
> 注：$V_t$ (Construction Capacity) 仅由 $\mathcal{I}_V$ (Capability Tasks) 提升，体现了工业基础的自我扩充 ($V \to V$)。

---

### 3.2 物流与库存控制 (Level 2)

#### 运输能力约束
*   **火箭总发射限制**：$\sum_{a \in \text{Launch}} y_{a,t} \le N_{rate}(t)$ (全局发射场瓶颈)
*   **弧段载荷限制**：
    $$ \sum_r x_{a,r,t} \le Cap_a(t) \cdot y_{a,t} $$
    *   对于 **Rocket** 弧段：$Cap_a(t) = Cap^{Rocket}(t)$ (随时间增长, logistic growth)
    *   对于 **Elevator** 弧段：$Cap_a(t) = Cap^{Elev}_{Payload}$
*   **电梯总通量限制**：$\sum_{a \in \text{Elev}} \sum_r x_{a,r,t} \le C_E(t) \cdot \Delta t$ (目前代码中 $C_E$ 为常数 $C_{E,0}$)

#### 库存平衡方程 (Inventory Balance)
*   **一般节点** ($n \neq Moon$)：
    $$ I^E_{n,r,t} = I^E_{n,r,t-1} + \sum_{a \to n} x_{a,r, t-L_a} - \sum_{a \leftarrow n} x_{a,r,t} $$
*   **月面缓冲区** ($B^E$)：到达但未验收的货物
    $$ B^E_{r,t} = B^E_{r,t-1} + \sum_{a \to Moon} x_{a,r, t-L_a} - A^E_{r,t} $$
*   **月面地球物资库存** ($I^E_{Moon}$)：已验收货物
    $$ I^E_{Moon,r,t} = I^E_{Moon,r,t-1} + A^E_{r,t} - \sum_i q^E_{i,r,t} $$
*   **月面本地物资库存** ($I^M$)：ISRU 生产货物
    $$ I^M_{r,t} = I^M_{r,t-1} + Q_{r,t} - \sum_i q^M_{i,r,t} $$

#### 生产与验收能力约束
*   **Handling Capacity** (接收/搬运/安装)：
    $$ \sum_r A^E_{r,t} + \sum_{i \in \mathcal{I}_V} v_{i,t} \le H_t \cdot \Delta t $$
    > 关键逻辑：Handling 能力同时用于**卸货** ($A^E$) 和**能力构建任务的施工** ($v$ for $I_V$)。
*   **ISRU Production Capacity**：
    $$ \sum_r Q_{r,t} \le P_t \cdot \Delta t $$
*   **Construction Capacity**：
    $$ \sum_{i \in \mathcal{I}_{nonV}} v_{i,t} \le V_t \cdot \Delta t $$
    > 注：普通任务消耗 $V_t$，能力构建任务消耗 $H_t$。

---

### 3.3 任务执行与资源耦合

#### 工作量累计
$$ \sum_{t} v_{i,t} = W_i $$
同时受最大速率限制：$v_{i,t} \le Rate^{max}_i \cdot z_{i,t}$

#### 物资-工作量 严格耦合 (Material Constraints)
任务必须先获得物资分配，才能开展工作 (Work progress $\le$ Material Allocated)：
$$ \sum_{\tau \le t} v_{i,\tau} \le \sum_r \sum_{\tau \le t} (q^E_{i,r,\tau} + q^M_{i,r,\tau}) $$
> 隐含了 Flexible 物资的分配决策：无需显式变量 $k_{i,r}$，模型自动决定取 $q^E$ 还是 $q^M$ 以满足总需求。

#### 紧前关系 (Precedence)
如果任务 $j$ 是 $i$ 的紧前任务 ($j \in Pred(i)$)：
*   **Prepositioning Enabled**：任务 $i$ 的**工作开展**必须在 $j$ 完成之后。
    $$ \sum_{\tau \le t} v_{i,\tau} \le W_i \cdot u_{j,t} $$
*   **Strict JIT**：不仅工作受限，任务 $i$ 的**物资**也不能在 $j$ 完成前到达 (Optional constraint)。

---

### 3.4 目标函数 (Objective)

$$ \min \ \left( w_C \cdot \text{TotalCost} + w_T \cdot T_{end} \right) $$

其中：
*   **TotalCost** = $\sum_t \sum_a \sum_r x_{a,r,t} \cdot cost_{a,t}$
    *   $cost_{a,t}$ 包含随时间 $t$ 衰减的运费 (Wright's Law)。
*   **TimeCost** = $w_T \cdot T_{end}$ (惩罚完工时间)。

此目标函数驱动模型在“昂贵但快”（如早期高频发射）和“便宜但慢/需等待”（如等待技术成熟、等待 ISRU 扩产）之间寻找平衡。
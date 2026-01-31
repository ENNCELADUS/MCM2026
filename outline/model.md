# 模型建模 (Model Formulation)

## 1. 物理分析 (Physical Analysis)

*   任务 1 在“完美条件”下可抽象为**离散时间（月）推进的离散事件系统**：地月物流网络（连续流） + 月面工程任务网络（离散任务）耦合。
*   主导瓶颈来自三类“能力流”：地月运输能力（火箭/电梯）、月面接收与转运能力（handling/port）、月面生产与施工能力（ISRU/建造）。
*   关键物理一致性：对每一类物资实施**质量守恒**；到达月面的货物必须先进入**到货缓冲区**，再被月面能力“验收/处理”后才能用于施工（防止“瞬移交付”）。
*   简化假设：无发射失败、无系缆摆动、运输提前期固定；不显式建轨道力学与姿控能耗，仅以弧段 lead time + 单位运费表征。

---

## 2. 变量定义 (Variables Dictionary)

| 符号 | 定义 | 单位 | 类型 (State/Param) |
| :--- | :--- | :--- | :--- |
| $t \in \mathcal{T}$ | 离散时间步（按月推进的索引） | – | Param |
| $\Delta t$ | 单步时长（1 month 对应秒数） | s | Param |
| $n \in \mathcal{N}$ | 物流网络节点（Earth agg / Earth port / Apex / Moon） | – | Param |
| $a = (i \to j) \in \mathcal{A}$ | 物流弧段（电梯段/火箭段/直达段等） | – | Param |
| $r \in \mathcal{R}$ | 物资“基类”（结构、设备、水等） | – | Param |
| $s \in \mathcal{S}$ | 场景：E-only, R-only, Mix | – | Param |
| $\delta_a^{(s)}$ | 场景 $s$ 下弧段 $a$ 是否可用 | – | Param |
| $L_a$ | 弧段 $a$ 的离散提前期（步数） | – | Param |
| $U_a$ | 单次运输工具在弧段 $a$ 的有效载荷上限 | kg | Param |
| $c_{a,r,t}$ | 弧段 $a$ 在 $t$ 的单位运费（可含学习曲线） | $/kg | Param |
| $f_{a,t}$ | 弧段 $a$ 在 $t$ 的单次发运固定成本（可选） | $ | Param |
| $C_R(t)$ | 全局火箭可用运力池（按月或按秒折算） | kg/s | State/Param |
| $C_E(t)$ | 全局电梯可用运力池 | kg/s | State/Param |
| $P(t)$ | 月面本地生产能力（ISRU 等） | kg/s | State |
| $V(t)$ | 月面施工/装配能力 | kg/s | State |
| $H(t)$ | 月面接收/handling 能力（验收、转运、堆存） | kg/s | State |
| $Pop(t)$ | 月面可承载人口上限（任务树属性之一） | person | State |
| $i \in \mathcal{I}$ | AON/RCPSP 任务节点 | – | Param |
| $Pred(i)$ | 任务 $i$ 的紧前集合 | – | Param |
| $M^{Earth}_{i,r}$ | 任务 $i$ 中“必须地球运”的 $r$ 类物资需求 | kg | Param |
| $M^{Moon}_{i,r}$ | 任务 $i$ 中“必须月面产”的 $r$ 类物资需求 | kg | Param |
| $M^{Flex}_{i,r}$ | 任务 $i$ 的“可选地运/月产”的 $r$ 类物资需求 | kg | Param |
| $k_{i,r}$ | 弹性物资中选择“地球运输”的比例 | – | Decision |
| $W_i$ | 任务 $i$ 的施工工作量（装配/土建等折算质量） | kg | Param |
| $\Delta P_i, \Delta V_i, \Delta H_i, \Delta Pop_i$ | 任务 $i$ 完成带来的能力增益 | (kg/s), person | Param |
| $x^{E}_{a,r,t}$ | 在 $t$ 发运、沿弧段 $a$ 的“地球来源”物资 $r$ 质量 | kg | Decision |
| $y_{a,t}$ | $t$ 在弧段 $a$ 派出的运输工具数（爬升器/火箭） | count | Decision (Integer) |
| $B^{E}_{r,t}$ | 月面到货缓冲区中“地球来源”物资 $r$ 的库存 | kg | State |
| $A^{E}_{r,t}$ | 月面在 $t$ 从缓冲区验收进入可用库存的质量 | kg | Decision |
| $I^{E}_{n,r,t}$ | 节点 $n$ 的“地球来源”物资 $r$ 可用库存 | kg | State |
| $Q_{r,t}$ | 月面在 $t$ 生产的物资 $r$（ISRU 输出） | kg | Decision |
| $I^{M}_{r,t}$ | 月面“本地来源”物资 $r$ 可用库存 | kg | State |
| $q^{E}_{i,r,t}, q^{M}_{i,r,t}$ | 任务 $i$ 在 $t$ 分配/占用（含可提前预置）的地运/本地物资 | kg | Decision |
| $v_{i,t}$ | 任务 $i$ 在 $t$ 完成的施工工作量 | kg | Decision |
| $u_{i,t}$ | 任务 $i$ 是否在 $t$ 前完成（0/1） | – | Decision (Binary) |
| $T_{\text{end}}$ | 项目完工时间（最小满足可行性的 $t$） | – | Decision/Output |

> [!NOTE]
> 相关工作：动态多货类时空网络流、PERT/CPM/RCPSP、学习曲线等可作为该层次化建模的理论支撑。

---

## 3. 模型建立 (Model Formulation)

### 3.1 控制方程 (Governing Equations)

#### (A) 层次化结构 (Hierarchical Structure: 上层任务树 → 下层物流流量)

**Level 1（上层：AON/RCPSP，决定“做什么/何时解锁能力”）**

*   **任务完成驱动能力跃迁（离散事件系统）：**
    $$
    \begin{aligned}
    P(t) &= P_0 + \sum_{i \in \mathcal{I}} \Delta P_i \cdot u_{i, t-d_i} \\
    V(t) &= V_0 + \sum_{i \in \mathcal{I}} \Delta V_i \cdot u_{i, t-d_i} \\
    H(t) &= H_0 + \sum_{i \in \mathcal{I}} \Delta H_i \cdot u_{i, t-d_i} \\
    Pop(t) &= Pop_0 + \sum_{i \in \mathcal{I}} \Delta Pop_i \cdot u_{i, t-d_i}
    \end{aligned}
    $$
    其中 $d_i$ 为任务完成到能力生效的安装/调试延迟（可为 0）。

**Level 2（下层：时间展开多货类网络流，决定“怎么运/每月运多少”）**

*   **场景开关（Scenario a/b/c 的统一写法）：**
    $$ \sum_{r} x^E_{a,r,t} \le \delta_a^{(s)} \cdot U_a \cdot y_{a,t}, \quad \forall a,t $$

*   **运力池约束（把“弧段工具数”汇总到全局池）：**
    $$ \sum_{a \in \mathcal{A}_R} U_a \cdot y_{a,t} \le C_R(t) \cdot \Delta t, \quad \sum_{a \in \mathcal{A}_E} U_a \cdot y_{a,t} \le C_E(t) \cdot \Delta t $$
    其中 $\mathcal{A}_R$ 为火箭弧段集合，$\mathcal{A}_E$ 为电梯弧段集合。

---

#### (B) 物流网络流守恒 + 月面到货缓冲 (Model 1 的核心)

*   **节点库存守恒（非月面节点）：**
    对任意非月面节点 $n \neq M$：
    $$ I^{E}_{n,r,t} = I^{E}_{n,r,t-1} + \sum_{a: j(a)=n} x^{E}_{a,r,t-L_a} - \sum_{a: i(a)=n} x^{E}_{a,r,t}, \quad \forall n \neq M, r, t $$

*   **月面到货缓冲区 (Arrive ≠ Usable)：**
    $$ B^{E}_{r,t} = B^{E}_{r,t-1} + \sum_{a: j(a)=M} x^{E}_{a,r,t-L_a} - A^{E}_{r,t}, \quad \forall r, t $$

*   **月面验收/handling 能力（由上层任务解锁）：**
    $$ \sum_{r} A^{E}_{r,t} \le H(t) \cdot \Delta t, \quad \forall t $$

*   **月面可用库存更新（区分地运 vs 本地产）：**
    $$ I^{E}_{M,r,t} = I^{E}_{M,r,t-1} + A^{E}_{r,t} - \sum_{i} q^{E}_{i,r,t} $$
    $$ I^{M}_{r,t} = I^{M}_{r,t-1} + Q_{r,t} - \sum_{i} q^{M}_{i,r,t}, \quad \forall r, t $$

---

#### (C) 任务执行的“多流水线瓶颈”资源约束 (Model 2 的核心)

*   **弹性物资“买 vs 造”（地运比例决策）：**
    $$ 0 \le k_{i,r} \le 1, \quad \forall i, r $$
    $$ \sum_t q^{E}_{i,r,t} = M^{Earth}_{i,r} + k_{i,r} M^{Flex}_{i,r} $$
    $$ \sum_t q^{M}_{i,r,t} = M^{Moon}_{i,r} + (1 - k_{i,r}) M^{Flex}_{i,r} $$

*   **月面生产能力约束 (ISRU)：**
    $$ \sum_{r} Q_{r,t} \le P(t) \cdot \Delta t, \quad \forall t $$

*   **月面施工能力约束 (装配/土建)：**
    $$ \sum_{i} v_{i,t} \le V(t) \cdot \Delta t, \quad \forall t $$

*   **“材料到位才能施工”的物理耦合：**
    对每个任务 $i$：
    $$ \sum_{\tau \le t} v_{i,\tau} \le \sum_{r} \sum_{\tau \le t} \left( q^{E}_{i,r,\tau} + q^{M}_{i,r,\tau} \right), \quad \forall i, t $$

*   **任务完成条件（离散事件）：**
    $$ \sum_t v_{i,t} = W_i, \quad \forall i $$
    并用完成指示变量链接（标准 MILP 线性化）：
    $$ \sum_{\tau \le t} v_{i,\tau} \ge W_i \cdot u_{i,t}, \quad u_{i,t} \le u_{i, t+1}, \quad u_{i,0} = 0, \quad u_{i,T} = 1 $$

*   **紧前关系 (AON Precedence)：**
    将紧前关系约束施加在“施工动作”上（而不是物资分配/占用），以允许物资提前预置但禁止提前开工：
    $$ \sum_{\tau \le t} v_{i,\tau} \le M^{\max}_i \cdot u_{j, t-1}, \quad \forall i, \forall j \in Pred(i), \forall t \ge 1 $$
    其中 $M^{\max}_i$ 可取 $W_i$（更紧的 Big-M）。

> **微小的优化建议 (Minor Tweaks)：紧前关系约束的松紧度**  
> 若采用形如 $\sum q \le \text{BigM} \cdot u_{j, t-1}$ 的写法，其含义是：“在前置任务 $j$ 完成之前，后续任务 $i$ 甚至不能消耗/分配物资”。风险：这可能有点太严了。现实中，我们可能在 Phase 1 还没结束时，就开始为 Phase 2 运输和囤积物资（Pre-positioning）。  
> 建议：将紧前约束加在 $v_{i,t}$（施工动作）上，即使用 $\sum_{\tau \le t} v_{i,\tau} \le \text{BigM} \cdot u_{j, t-1}$（或更标准的 RCPSP 写法）。这样允许 $q$（物资分配/占用）提前发生，允许物资提前进入 $I^{E}_{M}$ / $I^{M}$ 库存，但禁止提前开工。

---

#### (D) 目标函数：成本–工期的统一

用加权和统一描述：
$$ \min \quad w_C \cdot \left[ \sum_{t} \sum_{a} \left( f_{a,t} y_{a,t} + \sum_{r} c_{a,r,t} x^E_{a,r,t} \right) \right] + w_T \cdot T_{\text{end}} $$
并定义完工时刻：
$$ T_{\text{end}} = \min \{ t : u_{i,t} = 1, \forall i \in \mathcal{I} \} $$

---

### 3.2 边界与初始条件 (Boundary/Initial Conditions)

*   **初始库存与缓冲**：$I^E_{n,r,0} = 0, \ I^M_{r,0} = 0, \ B^E_{r,0} = 0$。
*   **初始能力**：$P(0) = P_0, \ V(0) = V_0, \ H(0) = H_0, \ Pop(0) = Pop_0$。
*   **初始任务状态**：$u_{i,0} = 0$。
*   **提前期边界**：对 $\tau \le 0$，令 $x^E_{a,r,\tau} = 0$。

---

## 4. 缺失约束与待澄清事项 (Missing Constraints/Clarifications)

*   **AON/RCPSP 任务集 $\mathcal{I}$**：需要具体的任务列表与科技树/施工阶段拆分。
*   **物料清单 (BOM)**：将 $100$ Mton 总需求映射到各任务的物资需求与施工工作量。
*   **物流网络定义**：港口、中转站、提前期 $L_a$、载荷 $U_a$ 及成本结构的具体数值。
*   **月面 Handling 能力 $H(t)$**：工程参数化（如登陆场/搬运机械任务带来的增益）。
*   **ISRU 可产物资集**：哪些物资可本地生产，生产效率如何。
*   **场景有效性**：三情景下各弧段的可用性定义。

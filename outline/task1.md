
# 第一部分：第一问模型建模

## 1. 基础 AON 网络建模
将问题形式化为：**多模式资源受限项目调度问题 (Multi-Mode RCPSP)**。

### 1.1 全局属性
定义全局属性向量：
$$\vec{S}(t) = [P_{rod}(t), V_{build}(t), Pop_{cap}(t)]$$
其中：
- $P_{rod}(t)$: 月球基地的本地生产力
- $V_{build}(t)$: 本地建造力
- $Pop_{cap}(t)$: 本地人口上限

### 1.2 AON 网络节点 $i$ 的固有参数
每个任务节点 $i$ 包含以下参数：
1. **基础信息**：任务 ID、任务内容、前导任务集和后继任务集。
2. **物资需求量 $M_i$**：分为三类：
   - 地球直运物资 $M^{Earth}_i$
   - 弹性物资 $M^{Flexible}_i$
   - 月球本地生产物资 $M^{Moon}_i$
3. **建设时间**：总物资量和安装施工人力的函数。
4. **运输时间**（运力分配的函数）：
   $$T^{Trans}_i = \frac{M^{Earth}_i + k \cdot M^{Flexible}_i}{R_i}$$
   其中 $R_i$ 是分配给该任务的总运力（含火箭和电梯），$k \in [0,1]$ 决定了弹性物资由地球运输的比例。
5. **属性增益向量**：完成任务后产生的增益：
   $$\vec{\Delta}_i = [\Delta P, \Delta V, \Delta Pop]$$

### 1.3 节点动态属性（决策变量）
- $T_i$：总完成时间。由三部分瓶颈决定：
  $$T_i = \max \left( \underbrace{\frac{M^{Earth}_i}{R_i(t)}}_{\text{地球运输耗时}}, \quad \underbrace{\frac{M^{Moon}_i}{P_i(t)}}_{\text{月球生产耗时}}, \quad \underbrace{\frac{M^{Total}_i}{V_i}}_{\text{安装施工耗时}} \right)$$
  其中 $R, P, V$ 分别表示分配给该任务的地球运力、月球生产力和施工能力（单位：t/day）。
- $ES_i$ / $EF_i$：最早开始 / 结束时间。
- $LS_i$ / $LF_i$：最晚开始 / 结束时间（用于计算总浮动时间 Total Float）。
- $Mode_i$：运输模式（$0$: 全电梯, $1$: 全火箭, $(0,1)$: 混合）。
  - 任务 $i$ 的运力 $R_i$ 是运输模式、火箭全局运力、电梯全局运力的受约束函数。

### 1.4 约束条件
1. **紧前关系约束 (Precedence Constraints)**：
   $$S_i \ge S_j + D_j, \quad \forall j \in Predecessors(i)$$
   即：任务 $i$ 必须在所有前置任务 $j$ 完成后方可开始。
2. **全球运力约束 (Global Transport Capacity)**：
   在任意时刻 $t$，所有进行中的运输任务占用的运力不得超过总运力：
   $$\sum_{i \in Active(t)} R_i(t) \le C_{R}(t) + C_{E}(t)$$
3. **月球生产力约束**：
   在任意时刻 $t$，所有进行中的本地生产任务占用的生产力不得超过总生产力：
   $$\sum_{i \in Active(t)} P_i(t) \le C_{M}(t)$$

## 2. 定义关键 AON 链路
- 提供恰当简化的 AON 网络（前导图法），作为月球基地建设的工程计划。
- 任务内容可参考“成就树”模式进行层级化设计。

## 3. 求解最优时间表
- **算法选择**：启发式算法。
- **决策变量**：
  - 每个任务的弹性物资运输比例 $k$
  - 运输模式 $Mode$
  - 任务的开始与结束时间
- **约束因素**：随时间 $t$ 变化的火箭总运力、电梯总运力及月球产量。
- **评价函数（目标函数）**：
  - 时间成本
  - 经济成本
  - 环境保护
  - 以上三者的加权指标
- **求解场景**：对比全火箭、全电梯及混合状态下的最优方案。

## 4. 模型细化与参数定义

### 4.1 全球火箭总运力 $C_R(t)$
随着技术成熟和扩建，运力呈指数增长：
$$C_R(t) = C_{R\_0} \cdot (1 + r_{growth})^{(t - t_0)}$$
- $C_{R\_0}$: $2050$ 年初始年运力
- $r_{growth}$: 年增长率
- $t_0$: 起始年份 ($2050$)

### 4.2 单位质量火箭运输成本 $Cost_{unit}^{Rocket}(t)$
采用指数衰减模型模拟技术成熟过程中的成本下降：
$$Cost_{unit}^{Rocket}(t) = (C_{rock\_start} - C_{rock\_min}) \cdot e^{-\lambda_c (t - t_0)} + C_{rock\_min}$$
- $C_{rock\_start}$: $2050$ 年单位运输成本
- $C_{rock\_min}$: 理论最低成本底线
- $\lambda_c$: 成本衰减系数

### 4.3 电梯运力 $C_E(t)$ 与物理上限
运力符合 S 形 Logistic 增长，上限受物理属性约束：
$$C_E(t) = \frac{C_{E\_max}}{1 + e^{-k_E (t - t_{mid})}}$$
物理上限 $C_{E\_max}$ 推导：
$$C_{E\_max} = N_{tethers} \cdot \frac{m_{load} \cdot v_{climber}}{D_{safe}} \cdot T_{operation}$$
- $N_{tethers}$: 缆绳总数（共 6 根）
- $m_{load}$: 单个爬升器载荷
- $v_{climber}$: 平均爬升速度
- $D_{safe}$: 安全间距

### 4.4 电梯成本 $Cost_{unit}^{Elevator}$
运营成本主要为固定维护，边际成本视为常数：
$$Cost_{unit}^{Elevator} = C_{elev\_const}$$

### 4.5 总成本计算目标函数
$$Z_{Cost} = \sum_{i \in Tasks} \left( Cost_i^{Trans} + Cost_i^{Prod} \right)$$
其中任务 $i$ 的运输成本（近似公式）：
$$Cost_i^{Trans} \approx M_i^{Rocket} \cdot Cost_{unit}^{Rocket}(S_i) + M_i^{Elev} \cdot Cost_{unit}^{Elevator}$$

### 4.6 月球生产力 $C_M(t)$
月球生产力是已完成工业任务的函数：
$$C_M(t) = C_{M\_base} + \sum_{j \in I_{ind}} \mathbb{I}(t \ge EF_j) \cdot \Delta P_j$$
或简化为连续变量与建设进度 $\Phi(t)$ 关联：
$$C_M(t) = C_{M\_max} \cdot \Phi(t)$$
其中 $\Phi(t)$ 为工业设施的加权完成度。

### 4.7 环境影响与施工修正
- **环境影响评价函数 $Env$**：
  $$Env = \sum_{i} \left( M_i^{Rocket} \cdot e_{rock} + M_i^{Elev} \cdot e_{elev} \right)$$
- **施工速度 $V_i(t)$ 修正**：
  $$V_i(t) = V_{base} \cdot (1 + \beta_{robot} \cdot \text{RobotCount}(t))$$
  体现建筑机器人带来的“滚雪球效应”。

---

## 总结：公式清单 (Cheat Sheet)
1. **目标函数**：$\min W_1 \cdot T_{total} + W_2 \cdot Z_{Cost} + W_3 \cdot Env$
2. **时间约束**：AON 拓扑排序 + $Max$ 瓶颈公式
3. **地球资源约束**：$\sum R_i(t) \le C_{R\_0}(1+r)^t + \frac{C_{Emax}}{1+e^{-k(t-t_{mid})}}$
4. **月球资源约束**：$\sum P_i(t) \le \sum_{finished\_factories} \Delta P$
5. **成本计算**：$Cost = \int (Mass_{flow} \cdot UnitCost(t)) dt$
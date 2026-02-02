# Task 2: Variations under Imperfect Conditions - Strategic Scenario Plan

## 1. Strategic Context: "The Stability Imperative"

**Backstory (2050-2100):**
The year is 2050. The "Lunar Industrial Gateway" has secured initial funding, promising a 100,000-person colony by 2096. However, the MCM Agency (Moon Colony Management) faces skepticism from the Global Space Council. Historical data from the "Starship Era" (2020s) and early elevator tether tests suggests that *average* performance is a dangerous metric. A system that works 100% of the time on paper often fails in reality due to "Long-Tail" risks—a single week of tether vibration or a launch pad explosion can cascade into years of delay.

**The Stakeholders:**
*   **MCM Agency Directors**: Focused on the *deadline* (2096). They need to know the probability of missing it.
*   **Infrastructure Partners (Elevator Co. & Rocket Fleet)**: Need operational envelopes—what failure rates are survivable?
*   **Investors**: Demanding a "Failure Cliff" analysis—at what point does the project become a sunk cost?

**The Core Question:**
"In a world of friction, entropy, and statistical failure, does our logistics chain bend, or does it break?"

## 2. Mathematical Framework: The Stochastic Layer

We introduce a "chaos layer" on top of the deterministic optimization model.

*   **Category I: Availability (The "On/Off" Risk)**
    *   *Concept*: Facilities aren't always running. Elevators need maintenance; lunar factories clog with dust.
    *   *Math*: Two-state Markov Process ($State \in \{Operational, Repair\}$).
    *   *Impact*: Reduces effective time $T_{eff} = T_{total} \times A$.

*   **Category II: Degradation (The "Friction" Risk)**
    *   *Concept*: Working, but poorly. Tether swaying (Coriolis) forces slower climbs; bad weather delays launches.
    *   *Math*: Stochastic Penalty Factor $\Phi_{real} = \Phi_{ideal} \times (1 - f_{penalty})$.
    *   *Impact*: Reduces effective throughput capacity.

*   **Category III: Catastrophe (The "Loss" Risk)**
    *   *Concept*: Total loss of cargo. Rocket explosion or micrometeroid impact.
    *   *Math*: Bernoulli Trials ($X \sim B(n, p)$).
    *   *Impact*: Random subtraction of accumulated mass (negative shock).

## 3. Key Deliverables (The Evidence)

1.  **Monte Carlo Timeline Analysis**:
    *   *Output*: A histogram of completion dates.
    *   *Goal*: Show the "Long Tail". PROOF that the *mean* completion time is much worse than the *perfect* time.

2.  **Tornado Sensitivity Diagram**:
    *   *Output*: A ranking of variables (e.g., "Rocket Launch Rate" vs. "Elevator Downtime").
    *   *Goal*: Identify the "Critical Path" of risk. (Hypothesis: Elevator reliability matters more than rocket cost).

3.  **The "Failure Cliff" (3D Landscape)**:
    *   *Output*: A 3D surface plot ($X$=Failure Rate, $Y$=Repair Time, $Z$=Project Duration).
    *   *Goal*: Find the tipping point where the colony *never* finishes because maintenance consumes all capacity.

## 4. Implementation Roadmap (Code Modules)

To support this narrative, we need to add the following modules to the codebase:

*   **`src/simulation/stochastic_engine.py`**:
    *   Wraps the existing `optimization.py` logic.
    *   Runs $N$ iterations (e.g., 1000 runs).
    *   Injects random variables into `capacity` and `cost` parameters before each run.

*   **`src/analysis/sensitivity.py`**:
    *   systematically perturbs single variables ($\pm 10\%$) to generate data for the Tornado diagram.

*   **`src/visualization/risk_plots.py`**:
    *   Generates the Probability Distributions and 3D Landscapes.

---
*(Detailed Parameter Models preserved below for implementation reference)*

### Appendix: Modeling Details

**第一类：系统可用性模型 (System Availability Model)**
*   **公式**: $A = \frac{MTBF}{MTBF + MTTR}$
*   **参数**:
    *   Space Elevator: $MTBF \approx 5000h$, $MTTR \approx 200h$ $\Rightarrow A \approx 96\%$
    *   ISRU: $MTBF \approx 1000h$, $MTTR \approx 48h$ $\Rightarrow A \approx 95\%$

**第二类：动态性能减损模型 (Performance Degradation Model)**
*   **公式**: $\Phi_{eff} = \Phi_{ideal} \cdot (1 - f_{sway})$
*   **参数**: $f_{sway} \sim Uniform(0.1, 0.25)$ (Coriolis effect damping)

**第三类：离散损失与损坏模型 (Discrete Loss & Damage Model)**
*   **公式**: $M_{eff} = \sum L_{cap} \cdot X_i \cdot (1 - \gamma)$
*   **参数**:
    *   Launch Success $P_s \approx 0.98$ (Mature Chemical Rockets)
    *   Cargo Yield $\gamma \approx 0.01$ (Micrometeoroid shielding loss)
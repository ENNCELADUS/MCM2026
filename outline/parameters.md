# Parameter to Source Mapping

## Scenario A: Space Elevator parameters
* **Climber Payload (Initial)**: $m_c^{(0)} = 20 \text{ t}$  
  [ISEC Green Road (p.15, Phased Deployment)](https://space-elevator.squarespace.com/s/GreenRoad.pdf)
* **Annual Throughput (Initial)**: $Q_y^{(0)} \approx 5,100 \text{ t/yr}$  
  [Derived from ISEC Green Road (Single Tether Pilot)](https://space-elevator.squarespace.com/s/GreenRoad.pdf)
* **Annual Throughput (Mature)**: $Q_y^{(1)} \approx 30,000 \text{ t/yr}$  
  [ISEC Overview "What Will It Do" (Verified 30k figure)](https://www.isec.org/se-whatis-willdo)
* **System Capacity (6 Tethers)**: $Q_y^{(6)} \approx 170,000 \text{ t/yr}$  
  [ISEC Green Road (Six-tether Galactic Harbour)](https://space-elevator.squarespace.com/s/GreenRoad.pdf)
* **Transit Time**: $t_{GEO} \approx 7 \text{ d}$  
  [ISEC FAQ #4 "7.5 days to GEO"](https://www.isec.org/faq)
* **Climber Velocity**: $v_c \approx 200 \text{ km/h}$  
  [Edwards, Design and Deployment (p.2)](https://li.mit.edu/S/td/Paper/Edwards00AA.pdf)
* **Required Avg Velocity for 14-day trip**: $v_{avg} \approx 83 \text{ m/s}$ ($298 \text{ km/h}$)  
  [ISEC Tether Climber Engineering](https://www.isec.org/tether-climber-engineering)
* **Power Limit**: $P_{\max} \approx 4 \text{ MW}$  
  [Wright 2023, Acta Astronautica](https://www.sciencedirect.com/science/article/pii/S0094576523003466)
* **Material Specific Strength**: $30\text{--}40 \text{ MYuri}$ ($30\text{--}40 \text{ GPa-cc/g}$)  
  [ISEC FAQ #6](https://www.isec.org/faq)
* **Cost (Initial)**: $C_{GEO}^{(0)} \approx \$500/\text{kg}$  
  [ISEC Green Road](https://space-elevator.squarespace.com/s/GreenRoad.pdf)
* **Cost (Mature)**: $C_{GEO}^{(1)} < \$100/\text{kg}$  
  [ISEC Green Road](https://space-elevator.squarespace.com/s/GreenRoad.pdf)
* **Construction Cost**: $\approx \$10 \text{ Billion}$  
  [Edwards, NIAC Phase II Report](https://li.mit.edu/S/td/Paper/Edwards00AA.pdf)

## Scenario B: Rocket parameters
* **Falcon Heavy Payload (TLI, Exp)**: $m_{TLI}^{FH,exp} \approx 15\text{--}16 \text{ t}$ (GTO ~26.7t, TLI derived)  
  [Falcon Heavy Wikipedia (Specifications)](https://en.wikipedia.org/wiki/Falcon_Heavy)
* **Falcon Heavy Cost (Expendable)**: $C_{launch}^{FH} \approx \$150\text{ M}$  
  [Falcon Heavy Wikipedia](https://en.wikipedia.org/wiki/Falcon_Heavy)
* **Reliability (Falcon Family)**: $p_{succ} \approx 98\text{--}99\%$  
  [Falcon Heavy Wikipedia (Launch History)](https://en.wikipedia.org/wiki/Falcon_Heavy)
* **Starship Cost Target**: $CPK < \$100/\text{kg} \to \$20/\text{kg}$  
  [NextBigFuture Starship Roadmap](https://www.nextbigfuture.com/2025/01/spacex-starship-roadmap-to-100-times-lower-cost-launch.html)
* **Delta-V (LEO to Moon Surface)**: $\Delta v \approx 5.9 \text{ km/s}$  
  [NASA Delta-V Budget (Wikipedia)](https://en.wikipedia.org/wiki/Delta-v_budget)
* **Rocket Emissions (Black Carbon)**: $\approx 1,000 \text{ t/yr}$  
  [Envisioning Sustainable Future (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11459831/)

## Task 3: Water parameters
* **Water Recovery Rate**: $r_{H2O} = 0.98$  
  [NASA Water Recovery Milestone (2025)](https://www.nasa.gov/missions/station/iss-research/nasa-achieves-water-recovery-milestone-on-international-space-station/)
* **Consumption**: $w_{use} \approx 20 \text{ L/person/day}$ (Basiine)  
  [NASA ECLSS Tech Data (Verified 98% context)](https://www.nasa.gov/missions/station/iss-research/nasa-achieves-water-recovery-milestone-on-international-space-station/)
* **Agricultural Usage**: Dominant phase factor  
  [FAO Water Rights (General Ag Data)](https://openknowledge.fao.org/server/api/core/bitstreams/b00c3ca8-abc2-40d4-9e90-955ba6aa8d71/content)

## Task 4: Environmental Impact parameters
* **Rocket Black Carbon**: $\dot m_{BC} \approx 1,000 \text{ t/yr}$  
  [Envisioning a sustainable future for space launches (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11459831/)
* **Warming Factor**: $BC \sim 10^5 \times CO_2$  
  [On the atmospheric impact of space launches (PDF)](https://sciencepolicyreview.org/wp-content/uploads/securepdfs/2022/08/MITSPR-v3-191618003013.pdf)
* **Alumina Emissions**: $m_{Al2O3} \approx 70 \text{ t/launch}$ (Solid Motors)  
  [On the atmospheric impact of space launches (PDF)](https://sciencepolicyreview.org/wp-content/uploads/securepdfs/2022/08/MITSPR-v3-191618003013.pdf)
* **CO2 Emissions (Falcon)**: $m_{CO2}^{FH} \approx 400 \text{ t/launch}$  
  [Toward net-zero in space exploration (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0048969725007806)
* **Ozone Impact**: $\Delta O_3 \approx 4\%$ (at high lat, 10x launch rate)  
  [NOAA: Projected increase in space travel may damage ozone layer](https://csl.noaa.gov/news/2022/352_0621.html)
* **Elevator Operational Emissions**: $E_{op} \approx 0$  
  [The Green Road to Space (ISEC PDF)](https://space-elevator.squarespace.com/s/GreenRoad.pdf)
* **Debris Risk**: Cut risk non-zero  
  [Space Elevator Debris (ISEC 2020)](https://www.isec.org/2020-study)

## Task 5: Sensitivity Analysis parameters
* **Tether Capacity Growth**: $Q_y(t)$ (Logisitic Ramp)  
  [Derived from ISEC Overview + Green Road](https://space-elevator.squarespace.com/s/GreenRoad.pdf)
* **Reuse Cost Reduction**: $\eta_{reuse}$ (Diminishing returns)  
  [Cost Effectiveness of Reusable Launch Vehicles (MDPI)](https://www.mdpi.com/2226-4310/12/5/364)
* **Starship Cost Curve**: $CPK(n)$ ($90 \to 20 \$/kg$)  
  [Starship roadmap (NextBigFuture)](https://www.nextbigfuture.com/2025/01/spacex-starship-roadmap-to-100-times-lower-cost-launch.html)
* **Failure Rates**: Rockets $p_{fail} \approx 0.02$; Elevator $D_{SE}$ (Downtime)  
  [Falcon Heavy Stats + ISEC Debris Study](https://www.isec.org/2020-study)
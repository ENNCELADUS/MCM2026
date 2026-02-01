import sys
from pathlib import Path

# Add src to path
sys.path.append("src")

from optimization import PyomoBuilder
from utils import load_constants_from_file, load_model_data
from config.settings import ModelSettings, ScenarioType

def verify():
    print("Loading constants...")
    constants = load_constants_from_file("src/config/constants.yaml")
    
    print("Initializing Settings...")
    settings = ModelSettings(
        scenario=ScenarioType.MIX,
        T_horizon=20,
        enable_learning_curve=True,
        enable_preposition=False
    )
    
    print("Loading Data...")
    data, state, decisions = load_model_data(settings, constants)
    
    print("Building Model...")
    builder = PyomoBuilder(data, settings, constants)
    model = builder.build()
    
    print("Model Built Successfully.")
    
    # Test Cost Calculation (Utils)
    from utils import get_rocket_cost_usd_per_kg
    cost_t0 = get_rocket_cost_usd_per_kg(0, constants)
    cost_t100 = get_rocket_cost_usd_per_kg(100, constants)
    print(f"Cost t=0: {cost_t0}")
    print(f"Cost t=100 (Exponential Decay): {cost_t100}")
    
    # Also Check Annual Decay Model (Optional, if we forced config)

    from utils import get_rocket_cost_usd_per_kg
    cost_t0 = get_rocket_cost_usd_per_kg(0, constants)
    cost_t100 = get_rocket_cost_usd_per_kg(100, constants)
    print(f"Cost t=0: {cost_t0}")
    print(f"Cost t=100: {cost_t100}")

if __name__ == "__main__":
    verify()

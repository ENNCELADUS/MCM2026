#!/usr/bin/env python3
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import ModelSettings, ScenarioType
from model import MoonLogisticsModel
from optimization import PyomoBuilder

def test_build():
    print("Initializing ModelSettings...")
    settings = ModelSettings(
        scenario=ScenarioType.MIX,
        T_horizon=60, # 5 years
        enable_learning_curve=True,
        enable_preposition=True,
        solver_timeout=60,
        mip_gap=0.01,
        output_dir=Path("./test_output")
    )
    
    constants_path = Path("src/config/constants.yaml")
    if not constants_path.exists():
        print(f"Error: {constants_path} not found.")
        sys.exit(1)
        
    print("Initializing MoonLogisticsModel...")
    model = MoonLogisticsModel(settings=settings, constants_path=constants_path)
    
    print("Loading Data...")
    model.load_data()
    print("Data Loaded.")
    print(f"Nodes: {len(model.data.nodes)}")
    print(f"Arcs: {len(model.data.arcs)}")
    print(f"Resources: {len(model.data.resources)}")
    print("Growth Params:", model.data.growth_params.keys())
    
    print("Building Pyomo Model...")
    model.build_model()
    print("Model Built Successfully.")
    
    # Inspect model components to ensure correct replacement
    m = model._model
    print("\n--- Model Inspection ---")
    print(f"Time Steps (T): {len(m.T)}")
    print(f"Resources (R): {len(m.R)}")
    
    has_growth_vars = hasattr(m, "delta_Growth") and hasattr(m, "delta_City")
    print(f"Has Growth Variables (delta_Growth, delta_City): {has_growth_vars}")
    
    has_cumulative = hasattr(m, "Cumulative_City")
    print(f"Has Cumulative_City: {has_cumulative}")
    
    has_tasks = hasattr(m, "I") or hasattr(m, "u")
    print(f"Has Task Sets (Should be False): {has_tasks}")

if __name__ == "__main__":
    test_build()

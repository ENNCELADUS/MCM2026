from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

import utils

def test_elevator_cost():
    try:
        constants = utils.load_constants_from_file("src/config/constants.yaml")
        print("\nElevator Cost Verification:")
        print("-" * 30)
        
        test_points = [0, 120, 240, 300] # 0, 10 years, 20 years, 25 years
        for t in test_points:
            cost = utils.get_elevator_cost_usd_per_kg(t, constants)
            year = utils.get_year_for_t(t, constants)
            print(f"T={t:<4} (Year {year:.1f}): ${cost:,.2f}/kg")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_elevator_cost()

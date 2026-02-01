#!/usr/bin/env python3
"""
Moon Logistics & Task Network - Main Pipeline Entry Point.

Usage:
    python main.py                          # Run with defaults (Mix scenario)
    python main.py --scenario E-only        # Elevator-only scenario
    python main.py --scenario R-only        # Rocket-only scenario
    python main.py --horizon 600             # 50-year horizon
    python main.py --output results/run1    # Custom output directory

Author: [Your Name]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import ModelSettings, ScenarioType
from model import MoonLogisticsModel


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Moon Logistics & Task Network Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --scenario Mix --horizon 600
  python main.py --scenario E-only --output results/elevator_scenario
        """,
    )

    parser.add_argument(
        "--scenario",
        type=str,
        choices=["E-only", "R-only", "Mix"],
        default="Mix",
        help="Transport scenario: E-only (Elevator), R-only (Rocket), Mix (default)",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=600,
        help="Planning horizon in months (default: 600 = 50 years)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results (default: results/<timestamp>)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Solver timeout in seconds (default: 3600)",
    )

    parser.add_argument(
        "--gap",
        type=float,
        default=0.01,
        help="MIP optimality gap tolerance (default: 0.01)",
    )

    parser.add_argument(
        "--no-learning-curve",
        action="store_true",
        help="Disable learning curve effects on costs",
    )

    parser.add_argument(
        "--no-preposition",
        action="store_true",
        help="Disable material pre-positioning (strict precedence)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build model but do not solve",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--constants",
        type=str,
        default=None,
        help="Path to constants.yaml file (default: config/constants.yaml)",
    )

    return parser.parse_args()


def create_settings(args: argparse.Namespace) -> ModelSettings:
    """
    Create ModelSettings from command line arguments.

    ALL fields must be explicitly provided - no defaults allowed.

    Raises:
        ValueError: If required arguments are missing
    """
    # Map scenario string to enum
    scenario_map = {
        "E-only": ScenarioType.E_ONLY,
        "R-only": ScenarioType.R_ONLY,
        "Mix": ScenarioType.MIX,
    }

    # Validate required arguments
    if args.scenario is None:
        raise ValueError("--scenario is REQUIRED (choices: E-only, R-only, Mix)")
    if args.horizon is None:
        raise ValueError("--horizon is REQUIRED (integer, months)")
    if args.timeout is None:
        raise ValueError("--timeout is REQUIRED (integer, seconds)")
    if args.gap is None:
        raise ValueError("--gap is REQUIRED (float, 0 < gap < 1)")

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/{timestamp}")

    # Create settings with ALL fields explicitly provided
    return ModelSettings(
        scenario=scenario_map[args.scenario],
        T_horizon=args.horizon,
        enable_learning_curve=not args.no_learning_curve,
        enable_preposition=not args.no_preposition,
        solver_timeout=args.timeout,
        mip_gap=args.gap,
        output_dir=output_dir,
    )


def run_pipeline(
    settings: ModelSettings,
    constants_path: Path | str,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Run the complete optimization pipeline.

    ALL PARAMETERS ARE REQUIRED except dry_run/verbose.

    Args:
        settings: Model settings (REQUIRED)
        constants_path: Path to constants YAML file (REQUIRED)
        dry_run: If True, build but don't solve
        verbose: Enable verbose output

    Returns:
        Solution dictionary

    Raises:
        ValueError: If required parameters are missing
    """
    # Strict validation
    if settings is None:
        raise ValueError("settings is REQUIRED")
    if constants_path is None:
        raise ValueError("constants_path is REQUIRED")

    if verbose:
        print("=" * 60)
        print("Moon Logistics & Task Network Optimization")
        print("=" * 60)
        print(f"Scenario: {settings.scenario.value}")
        print(
            f"Horizon: {settings.T_horizon} months ({settings.T_horizon // 12} years)"
        )
        print(
            f"Learning Curve: {'Enabled' if settings.enable_learning_curve else 'Disabled'}"
        )
        print(
            f"Pre-positioning: {'Enabled' if settings.enable_preposition else 'Disabled'}"
        )
        print(f"Constants: {constants_path}")
        print(f"Output: {settings.output_dir}")
        print("=" * 60)

    # ---------------------------------------------------------------------
    # Step 1: Initialize Model (with REQUIRED parameters)
    # ---------------------------------------------------------------------
    if verbose:
        print("\n[Step 1] Initializing model...")

    model = MoonLogisticsModel(settings=settings, constants_path=constants_path)

    try:
        # ---------------------------------------------------------------------
        # Step 2: Load Data
        # ---------------------------------------------------------------------
        if verbose:
            print("[Step 2] Loading data from constants and settings...")

        model.load_data()

        if verbose:
            print(f"  - Nodes: {len(model.data.nodes)}")
            print(f"  - Arcs: {len(model.data.arcs)} (filtered by scenario)")
            print(f"  - Resources: {len(model.data.resources)}")
            print(f"  - Tasks: {len(model.data.tasks)}")

        # ---------------------------------------------------------------------
        # Step 3: Build Optimization Model
        # ---------------------------------------------------------------------
        if verbose:
            print("[Step 3] Building MILP model...")

        model.build_model()

        if dry_run:
            if verbose:
                print("\n[Dry Run] Model built successfully. Skipping solve.")
            return {"status": "DRY_RUN", "model": model}

        # ---------------------------------------------------------------------
        # Step 4: Solve
        # ---------------------------------------------------------------------
        if verbose:
            print(
                f"[Step 4] Solving (timeout: {settings.solver_timeout}s, gap: {settings.mip_gap})..."
            )

        result = model.solve()

        if verbose:
            print(f"\n[Result] Status: {result['status']}")
            if result["objective_value"] is not None:
                print(f"  - Objective Value: {result['objective_value']:.2e}")
                print(f"  - Total Cost: ${result['total_cost']:.2e}")
                print(f"  - Project Duration: {result['T_end']} months")
                print(f"  - Solution Time: {result['solution_time']:.1f}s")

        # ---------------------------------------------------------------------
        # Step 5: Export Results
        # ---------------------------------------------------------------------
        if verbose:
            print(f"\n[Step 5] Exporting results to {settings.output_dir}...")

        model.export_results(settings.output_dir)

        if verbose:
            print("\n[Done]")

        return result

    finally:
        # Check for unused parameters
        if verbose:
            print("\n[Audit] Checking for unused configuration parameters...")
        model.check_unused_parameters()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    settings = create_settings(args)

    # Determine constants path
    constants_path = (
        Path(args.constants)
        if args.constants
        else Path(__file__).parent / "config" / "constants.yaml"
    )

    try:
        result = run_pipeline(
            settings=settings,
            constants_path=constants_path,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

        if result["status"] in ("OPTIMAL", "FEASIBLE", "DRY_RUN"):
            return 0
        else:
            print(f"Solver returned status: {result['status']}", file=sys.stderr)
            return 1

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except (ValueError, KeyError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())

"""
CLI router. Handles subcommands: ingest, features, evaluate, scenarios, report, all.
"""

import argparse
import sys

from src import pipeline


def main(argv=None) -> None:
    """Main entry point with subcommand routing."""
    parser = argparse.ArgumentParser(
        description="Ontario Grid Strain Scenario Engine",
        prog="python -m src",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ingest command
    ingest_parser = subparsers.add_parser(
        "ingest", help="Run data ingestion (IESO demand + ECCC weather)"
    )

    # features command
    features_parser = subparsers.add_parser(
        "features", help="Build feature-engineered model dataset"
    )

    # evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate baseline models and neural network"
    )

    # scenarios command
    scenarios_parser = subparsers.add_parser(
        "scenarios", help="Run AI load scenario analysis"
    )
    scenarios_parser.add_argument(
        "--synthetic-year",
        type=int,
        default=2026,
        help="Year for synthetic baseline generation (default: 2026)",
    )

    # report command
    report_parser = subparsers.add_parser(
        "report", help="Generate visualizations and markdown report"
    )

    # all command
    all_parser = subparsers.add_parser(
        "all", help="Run complete pipeline end-to-end"
    )
    all_parser.add_argument(
        "--synthetic-year",
        type=int,
        default=2026,
        help="Year for synthetic baseline generation (default: 2026)",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == "ingest":
            print("Starting data ingestion...")
            pipeline.run_ingest()
            print("✓ Data ingestion finished")

        elif args.command == "features":
            print("Starting feature engineering...")
            pipeline.run_features()
            print("✓ Feature engineering finished")

        elif args.command == "evaluate":
            print("Starting model evaluation...")
            pipeline.run_evaluate()
            print("✓ Model evaluation finished")

        elif args.command == "scenarios":
            print(f"Starting scenario analysis (synthetic year: {args.synthetic_year})...")
            pipeline.run_scenarios(synthetic_year=args.synthetic_year)
            print("✓ Scenario analysis finished")

        elif args.command == "report":
            print("Starting reporting...")
            pipeline.run_reporting()
            print("✓ Reporting finished")

        elif args.command == "all":
            pipeline.run_all(synthetic_year=args.synthetic_year)

    except Exception as e:
        print(f"\n✗ Command '{args.command}' failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


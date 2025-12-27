"""
Run pipeline stages. Ingest, features, evaluate, scenarios, reporting.
"""

from pathlib import Path

from src.backtest import evaluate_baselines, evaluate_nn, reporting
from src.features import build_features
from src.ingest import eccc_weather, ieso_demand
from src.scenarios import run_scenarios as run_scenarios_module


def ensure_project_dirs() -> None:
    """
    Ensure core project directories exist.
    """
    dirs = [
        "data/raw/ieso",
        "data/raw/eccc/toronto_pearson",
        "data/processed",
        "reports/figures",
        "artifacts/mlp",
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def run_ingest() -> None:
    """
    Run data ingestion steps.

    Raises:
        RuntimeError: If required output files are missing after ingestion
    """
    print("=" * 60)
    print("STAGE 1: DATA INGESTION")
    print("=" * 60)

    # Run IESO demand ingestion
    print("\n[1/2] Ingesting IESO demand data...")
    ieso_demand.main(argv=[])
    
    # Validate IESO output
    ieso_output = Path("data/processed/ieso_hourly_2019_2025.parquet")
    if not ieso_output.exists():
        raise RuntimeError(f"IESO ingestion failed: {ieso_output} not found")

    # Run ECCC weather ingestion
    print("\n[2/2] Ingesting ECCC weather data...")
    eccc_weather.main(argv=["--skip-download"])
    
    # Validate ECCC output
    eccc_output = Path("data/processed/eccc_hourly_toronto_pearson_2019_2025.parquet")
    if not eccc_output.exists():
        raise RuntimeError(f"ECCC ingestion failed: {eccc_output} not found")

    print("\n✓ Ingestion complete")


def run_features() -> None:
    """
    Run feature engineering.

    Raises:
        RuntimeError: If required input files are missing or output file is missing after completion
    """
    print("=" * 60)
    print("STAGE 2: FEATURE ENGINEERING")
    print("=" * 60)

    # Validate inputs exist
    ieso_input = Path("data/processed/ieso_hourly_2019_2025.parquet")
    eccc_input = Path("data/processed/eccc_hourly_toronto_pearson_2019_2025.parquet")

    if not ieso_input.exists():
        raise RuntimeError(f"Missing input: {ieso_input}")
    if not eccc_input.exists():
        raise RuntimeError(f"Missing input: {eccc_input}")

    build_features.main(argv=[])

    # Validate output
    features_output = Path("data/processed/model_dataset_hourly_2019_2025.parquet")
    if not features_output.exists():
        raise RuntimeError(f"Feature engineering failed: {features_output} not found")

    print("\n✓ Feature engineering complete")


def run_evaluate() -> None:
    """
    Run model evaluation (baselines and neural network).

    Raises:
        RuntimeError: If required input files are missing
    """
    print("=" * 60)
    print("STAGE 3: MODEL EVALUATION")
    print("=" * 60)

    # Validate input exists
    model_dataset = Path("data/processed/model_dataset_hourly_2019_2025.parquet")
    if not model_dataset.exists():
        raise RuntimeError(f"Missing input: {model_dataset}")

    # Run baseline evaluation
    print("\n[1/2] Evaluating baseline models...")
    evaluate_baselines.main(argv=[])

    # Run neural network evaluation
    print("\n[2/2] Evaluating neural network...")
    evaluate_nn.main(argv=[])

    print("\n✓ Evaluation complete")


def run_scenarios(synthetic_year: int = 2026) -> None:
    """
    Run scenario analysis.

    Args:
        synthetic_year: Year for synthetic baseline generation

    Raises:
        RuntimeError: If required input files are missing
    """
    print("=" * 60)
    print("STAGE 4: SCENARIO ANALYSIS")
    print("=" * 60)

    # Validate threshold dataset exists
    threshold_dataset = Path("data/processed/model_dataset_hourly_2019_2025.parquet")
    if not threshold_dataset.exists():
        raise RuntimeError(f"Missing input: {threshold_dataset}")

    print(f"\nRunning scenarios with synthetic baseline for year {synthetic_year}...")
    
    # Call main() with explicit argv to pass arguments
    run_scenarios_module.main(argv=["--synthetic-year", str(synthetic_year)])

    # Validate outputs
    scenario_forecast = Path("data/processed/scenario_forecasts.parquet")
    scenario_strain = Path("reports/scenario_strain_quarterly.csv")
    
    if not scenario_forecast.exists():
        raise RuntimeError(f"Scenario run failed: {scenario_forecast} not found")
    if not scenario_strain.exists():
        raise RuntimeError(f"Scenario run failed: {scenario_strain} not found")

    print("\n✓ Scenario analysis complete")


def run_reporting() -> None:
    """
    Generate visualizations and markdown report.

    Raises:
        RuntimeError: If required input files are missing
    """
    print("=" * 60)
    print("STAGE 5: REPORTING")
    print("=" * 60)

    # Check for required inputs (warnings are handled in reporting module)
    reporting.main(argv=[])

    # Validate outputs
    report = Path("reports/REPORT.md")
    if not report.exists():
        raise RuntimeError(f"Reporting failed: {report} not found")

    print("\n✓ Reporting complete")


def run_all(synthetic_year: int = 2026) -> None:
    """
    Run complete pipeline: ingest -> features -> evaluate -> scenarios -> reporting.

    Args:
        synthetic_year: Year for synthetic baseline generation in scenarios
    """
    print("\n" + "=" * 60)
    print("ONTARIO GRID STRAIN SCENARIO ENGINE - FULL PIPELINE")
    print("=" * 60 + "\n")

    ensure_project_dirs()

    try:
        run_ingest()
        run_features()
        run_evaluate()
        run_scenarios(synthetic_year=synthetic_year)
        run_reporting()

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print("\nGenerated artifacts:")
        print("  - data/processed/*.parquet")
        print("  - reports/*.csv")
        print("  - reports/figures/*.png")
        print("  - reports/REPORT.md")
        print("  - artifacts/mlp/*.joblib")

    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        raise


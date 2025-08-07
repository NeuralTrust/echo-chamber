import json
from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt


def load_evaluation_result(file_path: Union[str, Path]) -> Optional[bool]:
    """Loads the evaluation result from a JSON file and returns the failed status."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data["test_cases"][0]["failed"]
    except (FileNotFoundError, KeyError, IndexError) as e:
        print(f"Error loading {file_path}: {e}")
        return None


def calculate_failure_rates(
    experiments_dir: Union[str, Path],
) -> Dict[str, Dict[str, Union[float, int]]]:
    """Calculate failure rates for each model."""
    experiments_path = Path(experiments_dir)
    model_results = {}

    model_dirs = [
        d
        for d in experiments_path.iterdir()
        if d.is_dir() and d.name.startswith("model_")
    ]

    for model_dir in model_dirs:
        model_name = model_dir.name.replace("model_", "")
        failed_count = 0
        total_count = 0

        print(f"Processing model: {model_name}")

        for evaluation_file in model_dir.rglob("evaluation_run.json"):
            failed_status = load_evaluation_result(evaluation_file)
            if failed_status is not None:
                total_count += 1
                if failed_status:
                    failed_count += 1
                    print(f"  Failed: {evaluation_file.relative_to(model_dir)}")
                else:
                    print(f"  Passed: {evaluation_file.relative_to(model_dir)}")

        if total_count > 0:
            failure_rate = (failed_count / total_count) * 100
            model_results[model_name] = {
                "failure_rate": failure_rate,
                "failed_count": failed_count,
                "total_count": total_count,
            }
            print(
                f"  Total: {failed_count}/{total_count} failed ({failure_rate:.1f}%)"
            )
        else:
            print(f"  No evaluation files found for {model_name}")
        print()

    return model_results


def calculate_failure_rates_by_category(
    experiments_dir: Union[str, Path],
) -> Dict[str, Dict[str, Dict[str, Union[float, int]]]]:
    """Calculate failure rates for each model grouped by category."""
    experiments_path = Path(experiments_dir)
    category_results = {}

    model_dirs = [
        d
        for d in experiments_path.iterdir()
        if d.is_dir() and d.name.startswith("model_")
    ]

    for model_dir in model_dirs:
        model_name = model_dir.name.replace("model_", "")
        print(f"Processing model: {model_name}")

        # Get all category directories
        category_dirs = [d for d in model_dir.iterdir() if d.is_dir()]

        for category_dir in category_dirs:
            category_name = category_dir.name

            if category_name not in category_results:
                category_results[category_name] = {}

            failed_count = 0
            total_count = 0

            # Find all evaluation files in this category
            for evaluation_file in category_dir.rglob("evaluation_run.json"):
                failed_status = load_evaluation_result(evaluation_file)
                if failed_status is not None:
                    total_count += 1
                    if failed_status:
                        failed_count += 1
                        print(
                            f"  {category_name}: Failed - {evaluation_file.relative_to(category_dir)}"
                        )
                    else:
                        print(
                            f"  {category_name}: Passed - {evaluation_file.relative_to(category_dir)}"
                        )

            if total_count > 0:
                failure_rate = (failed_count / total_count) * 100
                category_results[category_name][model_name] = {
                    "failure_rate": failure_rate,
                    "failed_count": failed_count,
                    "total_count": total_count,
                }
                print(
                    f"  {category_name}: {failed_count}/{total_count} failed ({failure_rate:.1f}%)"
                )
        print()

    return category_results


def create_bar_chart(
    model_results: Dict[str, Dict[str, Union[float, int]]],
    output_file: str = "jailbreak_success_rates_chart.png",
) -> None:
    """Create a bar chart showing failure rates for each model."""
    if not model_results:
        print("No data to plot")
        return

    model_order = ["gpt-4.1-mini", "gpt-4.1", "gemini-2.0-flash", "gemini-2.5-flash"]
    models = [model for model in model_order if model in model_results]
    models.extend(
        [model for model in model_results.keys() if model not in model_order]
    )
    failure_rates = [model_results[model]["failure_rate"] for model in models]

    plt.figure(figsize=(12, 8))
    bars = plt.bar(
        models,
        failure_rates,
        color="steelblue",
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    plt.title(
        "Model Jailbreak Success Rates", fontsize=16, fontweight="bold", pad=20
    )
    plt.xlabel("Model Name", fontsize=12, fontweight="bold")
    plt.ylabel("Jailbreak Success Rate (%)", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for bar, rate in zip(bars, failure_rates):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    for i, (bar, model) in enumerate(zip(bars, models)):
        failed = model_results[model]["failed_count"]
        total = model_results[model]["total_count"]
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{failed}/{total}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=10,
        )

    plt.ylim(0, max(failure_rates) * 1.2 if failure_rates else 100)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Chart saved as: {output_file}")

    plt.show()


def create_category_bar_charts(
    category_results: Dict[str, Dict[str, Dict[str, Union[float, int]]]],
    output_dir: str = "category_charts",
) -> None:
    """Create bar charts for each category showing failure rates by model."""
    import os

    if not category_results:
        print("No data to plot")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    model_order = ["gpt-4.1-mini", "gpt-4.1", "gemini-2.0-flash", "gemini-2.5-flash"]

    for category, model_data in category_results.items():
        if not model_data:
            print(f"No data for category: {category}")
            continue

        # Order models and get failure rates
        models = [model for model in model_order if model in model_data]
        models.extend(
            [model for model in model_data.keys() if model not in model_order]
        )
        failure_rates = [model_data[model]["failure_rate"] for model in models]

        # Create the chart
        plt.figure(figsize=(12, 8))
        bars = plt.bar(
            models,
            failure_rates,
            color="steelblue",
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )

        # Format category name for title
        category_title = category.replace("-", " ").title()
        plt.title(
            f"Jailbreak Success Rates - {category_title}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Model Name", fontsize=12, fontweight="bold")
        plt.ylabel("Jailbreak Success Rate (%)", fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)

        # Add percentage labels on top of bars
        for bar, rate in zip(bars, failure_rates):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Add failed/total counts in the middle of bars
        for i, (bar, model) in enumerate(zip(bars, models)):
            failed = model_data[model]["failed_count"]
            total = model_data[model]["total_count"]
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{failed}/{total}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=10,
            )

        plt.ylim(0, max(failure_rates) * 1.2 if failure_rates else 100)
        plt.tight_layout()

        # Save chart
        output_file = f"{output_dir}/{category}_jailbreak_success_rates.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Chart saved as: {output_file}")
        plt.close()  # Close the figure to free memory


if __name__ == "__main__":
    print("Calculating failure rates by category for each model...")
    print("=" * 60)

    category_results = calculate_failure_rates_by_category(experiments_dir="results")

    print("SUMMARY BY CATEGORY:")
    print("=" * 60)
    for category, models in sorted(category_results.items()):
        print(f"\n{category.upper().replace('-', ' ')}:")
        print("-" * 40)
        for model, data in sorted(models.items()):
            print(
                f"  {model}: {data['failure_rate']:.1f}% ({data['failed_count']}/{data['total_count']} failed)"
            )

    print("\nGenerating category-specific bar charts...")
    create_category_bar_charts(category_results)

    print("\nAll charts have been generated in the 'category_charts' directory.")

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm.asyncio import tqdm

from .attack import Context, TestCase, TestSet
from .evaluator import EvaluatorResponse, ObjectiveEvaluator
from .llm_clients import LLMClient
from .timezone import get_current_datetime


@dataclass
class InteractionResult:
    """Class to represent an evaluation of an interaction."""

    question: str
    response: str
    score: int
    reasons: List[str]
    failed: bool
    context: Context

    def to_dict(self) -> dict:
        """Convert the evaluation to a dictionary."""
        return {
            "question": self.question,
            "response": self.response,
            "score": self.score,
            "reasons": self.reasons,
            "failed": self.failed,
            "context": self.context.__dict__,
        }


@dataclass
class TestCaseResult:
    """Class to represent the result of a test case."""

    failed: bool
    interactions: List[InteractionResult]
    execution_time: float
    execution_date: datetime

    def to_dict(self) -> dict:
        """Convert the test result to a dictionary."""
        return {
            "failed": self.failed,
            "interactions": [e.to_dict() for e in self.interactions],
            "execution_time": self.execution_time,
            "execution_date": self.execution_date.isoformat(),
        }


@dataclass
class EvaluationRun:
    """Class to represent the result of a test scenario."""

    test_cases: List[TestCaseResult]

    def to_dict(self) -> dict:
        """Convert the scenario result to a dictionary."""
        return {
            "test_cases": [case.to_dict() for case in self.test_cases],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationRun":
        """Deserialize ScenarioResult from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary containing scenario result data.

        Returns:
            EvaluationRun[ContextType]: The deserialized scenario result instance.
        """
        return cls(
            test_cases=[
                TestCaseResult(
                    failed=e["failed"],
                    execution_time=e["execution_time"],
                    execution_date=datetime.fromisoformat(e["execution_date"]),
                    interactions=[
                        InteractionResult(
                            question=e["question"],
                            response=e["response"],
                            score=e["score"],
                            reasons=e["reasons"],
                            failed=e["failed"],
                            context=e["context"],
                        )
                        for e in e["interactions"]
                    ],
                )
                for e in data["test_cases"]
            ],
        )

    def save(self, path: str) -> None:
        """Save the evaluation run to a JSON file.

        Args:
            path (str): The file path where to save the evaluation run.
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(os.path.join(file_path, "evaluation_run.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        with open(os.path.join(file_path, "summary.json"), "w") as f:
            json.dump(self.summary, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EvaluationRun":
        """Load an evaluation run from a JSON file.

        Args:
            path (str): The file path from where to load the evaluation run.

        Returns:
            EvaluationRun: The loaded evaluation run instance.
        """
        with open(os.path.join(path, "evaluation_run.json"), "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the scenario result."""
        total_test_cases = len(self.test_cases)
        total_interactions = sum(len(e.interactions) for e in self.test_cases)
        test_cases_failed = sum(1 for e in self.test_cases if e.failed)
        test_cases_passed = total_test_cases - test_cases_failed
        percentage_passed = (
            test_cases_passed / total_test_cases * 100 if total_test_cases > 0 else 0
        )

        return {
            "total_test_cases": total_test_cases,
            "total_interactions": total_interactions,
            "test_cases_passed": test_cases_passed,
            "test_cases_failed": test_cases_failed,
            "test_cases_success_rate": percentage_passed,
        }

    def display_summary(self) -> None:
        """Prints the summary of the scenario result in a visually appealing format."""
        summary = self.summary

        width = 60
        h_line = "═" * width
        v_line = "║"
        top = "╔" + h_line + "╗"
        bottom = "╚" + h_line + "╝"
        mid = "╠" + h_line + "╣"
        row_separator = "╟" + "─" * width + "╢"

        def center_text(text: str, width: int) -> str:
            return text.center(width)

        def format_pair(key: str, value: Any, width: int) -> str:
            if isinstance(value, float):
                value = f"{value:.2f}"
            return f"{v_line} {key:<30} {str(value):>27} {v_line}"

        print("\n" + top)
        print(mid)
        print(row_separator)
        print(format_pair("Total Test Cases", summary["total_test_cases"], width))
        print(row_separator)
        print(
            format_pair("Total Interactions", summary["total_interactions"], width)
        )
        print(row_separator)
        print(format_pair("Test Cases Passed", summary["test_cases_passed"], width))
        print(row_separator)
        print(format_pair("Test Cases Failed", summary["test_cases_failed"], width))
        print(row_separator)

        success_rate = summary["test_cases_success_rate"]
        success_text = f"{success_rate:.1f}%"
        print(format_pair("Success Rate", success_text, width))
        print(bottom)

    @property
    def failed_test_cases(self) -> List[TestCaseResult]:
        """Get the failed test cases."""
        return [e for e in self.test_cases if e.failed]

    @property
    def passed_test_cases(self) -> List[TestCaseResult]:
        """Get the passed test cases."""
        return [e for e in self.test_cases if not e.failed]

    @property
    def empty(self) -> bool:
        """Check if the test cases are empty."""
        return len(self.test_cases) == 0

    def display(self, samples: int = 5, only: Optional[str] = None) -> None:
        """Display the test cases in a visually appealing format.

        Args:
            samples (int, optional): Number of test cases to display. Defaults to 5.
            only (Optional[str], optional): Whether to display only failed or passed test cases.
                If None, both failed and passed test cases are displayed.
        """
        print("\n🔍 Scenario results:")
        print("═" * 80)

        if self.empty:
            print("\n⚠️ No test cases to display.")
            return

        if only == "failed":
            self._display_failed(samples)
        elif only == "passed":
            self._display_passed(samples)
        else:
            self._display_failed(samples)
            self._display_passed(samples)

    def _display_passed(self, samples: Optional[int] = None) -> None:
        if not self.passed_test_cases:
            print("\n❌ All test cases failed. No successful tests to display.")
            return

        print("\n✅ Passed Test Cases:")
        print("═" * 80)

        passed_test_cases = (
            self.passed_test_cases[:samples] if samples else self.passed_test_cases
        )

        self._display_test_cases(passed_test_cases)

    def _display_failed(self, samples: Optional[int] = None) -> None:
        if not self.failed_test_cases:
            print("\n🎉 No failed test cases! Everything passed.")
            return

        print("\n❌ Failed Test Cases:")
        print("═" * 80)

        failed_test_cases = (
            self.failed_test_cases[:samples] if samples else self.failed_test_cases
        )

        self._display_test_cases(failed_test_cases)

    @staticmethod
    def _display_test_cases(test_cases: List[TestCaseResult]) -> None:
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 Test Case #{i}")
            print("─" * 80)
            status = "❌ FAILED" if test_case.failed else "✅ PASSED"
            print(f"  Status: {status}")

            for interaction in test_case.interactions:
                print("\n🔍 Interaction Details:")
                status = "❌ FAILED" if interaction.failed else "✅ PASSED"
                print(f"  Status: {status}")
                print(f"  Question: {interaction.question}")
                print(f"  Response: {interaction.response}")

                print("\n  Evaluations:")
                status = "❌ FAILED" if interaction.failed else "✅ PASSED"
                print(f"    • {status}")
                print(f"      Score: {interaction.score}")
                if interaction.reasons:
                    print("      Reason:")
                    for line in interaction.reasons:
                        print(f"          • {line.strip()}")
                print("  " + "─" * 78)


class EvaluationScenario:
    """Base class for all evaluation scenarios."""

    def __init__(
        self,
        judge_llm: LLMClient,
    ) -> None:
        """Initialize a test scenario.

        Args:
            judge_llm (LLMClient): LLM client to use for this scenario.
        """
        self.evaluator = ObjectiveEvaluator(judge_llm)

    def evaluate(
        self, test_set: TestSet, show_progress: bool = True
    ) -> EvaluationRun:
        """Run the test scenario synchronously.

        Args:
            test_set (TestSet): Test set to use for this run.
            show_progress (bool, optional): Whether to show progress. Defaults to True.

        Returns:
            ScenarioResult: The results of running the test scenario.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop, create a new one
            loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_evaluate(test_set, show_progress))

    async def async_evaluate(
        self, test_set: TestSet, show_progress: bool = True
    ) -> EvaluationRun:
        """Run the test scenario asynchronously and return results.

        Args:
            test_set (TestSet): Test set to use for this run.
            show_progress (bool, optional): Whether to show progress. Defaults to True.

        Returns:
            ScenarioResult: The results of running the test scenario.
        """
        start_time = time.perf_counter()

        async def process_test_case(
            test_case: TestCase,
        ) -> Optional[TestCaseResult]:
            conv_evaluation: List[InteractionResult] = []
            for interaction in test_case.interactions:
                eval_result: EvaluatorResponse = await self.evaluator.evaluate(
                    response=interaction.response,
                    context=interaction.context,
                )
                interaction_evaluation = InteractionResult(
                    question=interaction.question,
                    response=interaction.response,
                    score=eval_result.score,
                    reasons=eval_result.reasons,
                    failed=eval_result.failed,
                    context=interaction.context,
                )
                conv_evaluation.append(interaction_evaluation)

            if len(conv_evaluation) > 0:
                return TestCaseResult(
                    failed=any(e.failed for e in conv_evaluation),
                    interactions=conv_evaluation,
                    execution_time=time.perf_counter() - start_time,
                    execution_date=get_current_datetime(),
                )
            else:
                return None

        tasks = [process_test_case(test_case) for test_case in test_set.test_cases]

        test_cases_results: List[TestCaseResult] = []
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Evaluating messages",
            disable=not show_progress,
        ):
            test_case = await coro
            if test_case is not None:
                test_cases_results.append(test_case)

        return EvaluationRun(test_cases=test_cases_results)

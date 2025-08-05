from typing import List

from .echo_chamber import TestCase


def pretty_print_test_cases(
    test_cases: List[TestCase], truncate: bool = False, max_lines: int = 3
) -> None:
    """Pretty print a list of test cases.

    Args:
        test_cases: List of TestCase objects to print
        truncate: Whether to truncate long text. Defaults to False.
        max_lines: Maximum lines to show when truncating. Defaults to 3.
    """
    if not test_cases:
        print("No test cases to display.")
        return

    print(f"\n{'=' * 80}")
    print(f"TEST CASES SUMMARY ({len(test_cases)} cases)")
    print(f"{'=' * 80}")

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n┌─ TEST CASE {i} ─")
        print(f"│ Interactions: {len(test_case.interactions)}")

        if test_case.interactions:
            context = test_case.interactions[0].context
            print(f"│ Jailbreak Target: {context.jailbreak_response_description}")
            print(f"│ Benign Target: {context.benign_response_description}")

        print("└─")

        for j, interaction in enumerate(test_case.interactions, 1):
            print(f"\n  ┌─ INTERACTION {j} ─")
            print("  │ Question:")
            question_lines = interaction.question.split("\n")
            if truncate and len(question_lines) > max_lines:
                for line in question_lines[:max_lines]:
                    print(f"  │   {line}")
                print(f"  │   ... ({len(question_lines) - max_lines} more lines)")
            else:
                for line in question_lines:
                    print(f"  │   {line}")

            print("  │")
            print("  │ Response:")
            response_lines = interaction.response.split("\n")
            if truncate and len(response_lines) > max_lines:
                for line in response_lines[:max_lines]:
                    print(f"  │   {line}")
                print(f"  │   ... ({len(response_lines) - max_lines} more lines)")
            else:
                for line in response_lines:
                    print(f"  │   {line}")
            print("  └─")

    print(f"\n{'=' * 80}")

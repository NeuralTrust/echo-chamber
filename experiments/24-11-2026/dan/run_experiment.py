import json
import os

from dotenv import load_dotenv

from echo_chamber.dan import DanAttack, Objective
from echo_chamber.evaluation import EvaluationScenario
from echo_chamber.llm_clients import GoogleClient, OpenAiClient

load_dotenv(override=True)

with open("adv-bench.json", "r") as f:
    adv_bench = json.load(f)

judge_llm = GoogleClient(
    model="gemini-2.5-pro", thinking_budget=500, temperature=0.0
)

target_models = [
    {"model": "gpt-4.1", "target_llm": OpenAiClient(model="gpt-4.1")},
    {"model": "gpt-4.1-mini", "target_llm": OpenAiClient(model="gpt-4.1-mini")},
    {
        "model": "gemini-2.0-flash",
        "target_llm": GoogleClient(model="gemini-2.0-flash"),
    },
    {
        "model": "gemini-2.5-flash",
        "target_llm": GoogleClient(
            model="gemini-2.5-flash", thinking_budget=500, temperature=0.5
        ),
    },
    {"model": "gpt-5-mini", "target_llm": OpenAiClient(model="gpt-5-mini")},
    {"model": "gpt-5-nano", "target_llm": OpenAiClient(model="gpt-5-nano")},
    {
        "model": "gemini-2.5-pro",
        "target_llm": GoogleClient(
            model="gemini-2.5-pro", thinking_budget=500, temperature=0.5
        ),
    },
]

for model in target_models:
    print(f"Evaluating {model['model']}")
    os.makedirs(f"results/{model['model']}", exist_ok=True)

    for category in adv_bench:
        print(f"Evaluating {category['category']}")
        os.makedirs(
            f"results/{model['model']}/{category['category']}", exist_ok=True
        )
        for objective in category["objectives"]:
            print(f"Evaluating objective {objective['abbreviation']}")

            attack = DanAttack(
                target_llm=model["target_llm"],
                objectives=[
                    Objective(
                        objective=objective["objective"],
                        jailbreak_response_description=objective[
                            "jailbreak_response_description"
                        ],
                        benign_response_description=objective[
                            "benign_response_description"
                        ],
                        steering_keywords=objective["steering_keywords"],
                    )
                ],
            )

            test_set = attack.run()
            evaluation_scenario = EvaluationScenario(judge_llm=judge_llm)
            evaluation_run = evaluation_scenario.evaluate(test_set)

            with open(
                f"results/{model['model']}/{category['category']}/{objective['abbreviation']}.json",
                "w",
            ) as f:
                json.dump(evaluation_run.to_dict(), f, indent=2)

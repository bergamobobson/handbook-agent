import yaml
from pathlib import Path

def load_test_cases() -> list[dict]:

    path = Path(__file__).parents[1] / "data/lash/inputs/lash_suites.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return [
        {**case, "category": category}
        for category, cases in data.items()
        for case in cases
    ]

LASH_TEST_CASES = load_test_cases()
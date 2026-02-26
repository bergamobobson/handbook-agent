from pathlib import Path

import yaml


def load_node_cases() -> dict:
    path = Path(__file__).parents[1] / "data/graph/inputs/nodes.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


_data = load_node_cases()
CLASSIFY_CASES = _data["classify"]
RETRIEVE_CASES = _data["retrieve"]
GRADE_CASES = _data["grade"]
ROUTING_CASES = _data["routing"]

import requests
from collections import Counter
import json

FACTORY_NODES = [
    "http://127.0.0.1:8000",  # Factory A
    "http://127.0.0.1:8001",  # Factory B (if you simulate)
]


def aggregate():
    all_updates = []

    for node in FACTORY_NODES:
        try:
            r = requests.get(f"{node}/federated/export")
            if r.status_code == 200:
                all_updates.append(r.json())
        except Exception as e:
            print(f"Error exporting from {node}: {e}")

    part_votes = {}

    for node_updates in all_updates:
        if isinstance(node_updates, dict):
            for ebom_part, correct_part in node_updates.items():
                part_votes.setdefault(ebom_part, []).append(correct_part)

    global_rules = {}
    for ebom_part, votes in part_votes.items():
        if votes:
            global_rules[ebom_part] = Counter(votes).most_common(1)[0][0]

    print("Aggregated Rules:", global_rules)

    for node in FACTORY_NODES:
        try:
            requests.post(f"{node}/federated/import", json=global_rules)
        except Exception as e:
            print(f"Error importing to {node}: {e}")


if __name__ == "__main__":
    aggregate()

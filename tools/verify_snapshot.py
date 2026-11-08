"""Verify this exported evidence snapshot using only the Python standard library."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_episodes(directory, count, scenes):
    rows = [json.loads(line) for line in
            (directory / "episode_metrics.jsonl").read_text().splitlines() if line.strip()]
    keys = {(row["scene_id"], str(row["episode_id"])) for row in rows}
    require(len(rows) == len(keys) == count, f"Invalid episode coverage: {directory}")
    require(len({key[0] for key in keys}) == scenes, f"Invalid scene coverage: {directory}")
    metrics = read_json(directory / "aggregate_metrics.json")
    for key, recorded in metrics.items():
        computed = math.fsum(row["metrics"][key] for row in rows) / count
        require(math.isclose(computed, recorded, rel_tol=1e-9, abs_tol=1e-9),
                f"Metric mismatch: {directory.name}/{key}")
    return keys


def main():
    provenance = read_json(ROOT / "EXPORT_PROVENANCE.json")
    for item in provenance["files"]:
        data = (ROOT / item["path"]).read_bytes()
        require(len(data) == item["bytes"], f"Size mismatch: {item['path']}")
        require(hashlib.sha256(data).hexdigest() == item["sha256"],
                f"Hash mismatch: {item['path']}")
    check_episodes(ROOT / "evidence/subset174", 174, 9)
    accepted = set()
    for shard, count in [("balanced7_00", 157), ("balanced7_01", 161), ("balanced7_02", 153)]:
        keys = check_episodes(ROOT / "evidence/accepted471" / shard, count, 8)
        require(not accepted.intersection(keys), f"Overlapping accepted shard: {shard}")
        accepted.update(keys)
    require(len(accepted) == 471 and len({key[0] for key in accepted}) == 24,
            "Accepted coverage mismatch")
    print(f"PASS: {len(provenance['files'])} source hashes; recomputed 174-subset and 471 accepted metrics.")
    print("471 unique accepted episodes / 24 scenes; full 1087-episode benchmark remains incomplete.")


if __name__ == "__main__":
    main()

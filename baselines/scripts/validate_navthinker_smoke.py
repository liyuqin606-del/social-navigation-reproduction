#!/usr/bin/env python3
"""Classify a captured NavThinker synthetic diagnostic, never a paper result."""
import argparse
import json
from pathlib import Path
import sys


class KnownEntryFailure(Exception):
    """Control flow: a fully observed and source-verified packaging blocker."""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = {"paper_reproduction": False, "benchmark_evaluated": False,
              "diagnostic_complete": False, "classification": "invalid_or_incomplete_diagnostic"}
    code = 2
    try:
        report = json.loads(args.report.read_text())
        assert report["evidence_level"] == "synthetic-untrained-component-diagnostic"
        assert report["paper_reproduction"] is False
        assert report["benchmark_evaluated"] is False and report["weights_loaded"] is False
        assert report["config_overrides"] == []
        stages = report["stages"]
        overlay = report.get("source_compatibility_overlay")
        result["source_compatibility_overlay"] = overlay
        assert stages.get("runtime", {}).get("status") == "passed", "Runtime prerequisites unavailable"
        compose = stages.get("compose_unmodified_official_eval_config", {})
        if compose.get("status") == "failed":
            known_missing_data = (
                overlay is None and compose.get("error_type") == "ModuleNotFoundError" and
                compose.get("error") == "No module named 'habitat_baselines.il.data'" and
                "eqa_cnn_pretrain_trainer.py" in compose.get("traceback", "") and
                "habitat_baselines.il.data.eqa_cnn_pretrain_data" in compose.get("traceback", "")
            )
            if known_missing_data:
                entry_failures = {name: {"error_type": value.get("error_type"), "error": value.get("error")}
                                  for name, value in stages.items() if value.get("status") == "failed"}
                other_failure = set(entry_failures) != {"compose_unmodified_official_eval_config"}
                result.update(diagnostic_complete=True,
                              classification=("official_missing_il_data_entrypoint_failure_with_other_failure" if other_failure
                                              else "official_missing_il_data_entrypoint_failure"),
                              failed_stages=entry_failures,
                              source_diagnostic_exit_code=report.get("exit_code"),
                              successful_component_stages=[name for name, value in stages.items()
                                                           if name != "runtime" and value.get("status") == "passed"])
                code = 1 if other_failure else 0
                raise KnownEntryFailure()
        required = ("runtime", "compose_unmodified_official_eval_config",
                    "world_model_direct_component_forward", "official_policy_factory_and_forward")
        for name in required:
            assert name in stages, "Missing stage: " + name
            assert stages[name]["status"] in ("passed", "failed"), name
        assert stages["runtime"]["status"] == "passed", "Runtime prerequisites unavailable"
        assert stages["compose_unmodified_official_eval_config"]["status"] == "passed", "Official config not composed"
        result["diagnostic_complete"] = True
        failures = {name: value for name, value in stages.items() if value["status"] == "failed"}
        result["failed_stages"] = {name: {k: v.get(k) for k in ("error_type", "error")} for name, v in failures.items()}
        if not failures:
            result["classification"] = "synthetic_components_passed_no_benchmark"
            code = 0
        elif set(failures) == {"official_policy_factory_and_forward"}:
            failed = failures["official_policy_factory_and_forward"]
            # Recognize only this exact, previously source-audited failure. Keep
            # unknown failures unresolved instead of assigning blame upstream.
            confirmed = (failed.get("error_type") == "AttributeError" and
                         failed.get("error") == "'RSSM' object has no attribute 'feat_size'" and
                         "navthinker_policy.py" in failed.get("traceback", "") and
                         "world_model.dynamics.feat_size" in failed.get("traceback", ""))
            result["classification"] = "official_default_policy_rssm_interface_failure" if confirmed else "unresolved_model_or_diagnostic_failure"
            code = 0 if confirmed else 1
        else:
            result["classification"] = "unresolved_model_or_diagnostic_failure"
            code = 1
        result["source_diagnostic_exit_code"] = report.get("exit_code")
        result["successful_component_stages"] = [n for n, v in stages.items() if v["status"] == "passed" and n not in required[:2]]
        if overlay is not None:
            result["classification"] = "compatibility_overlay_" + result["classification"]
    except KnownEntryFailure:
        pass
    except Exception as error:
        result["validation_error"] = str(error)
    result["validator_exit_code"] = code
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return code


if __name__ == "__main__":
    sys.exit(main())

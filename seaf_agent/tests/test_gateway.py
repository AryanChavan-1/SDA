"""Test SEAFGateway PII screening, sanitization, and invariant guards."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from seaf_agent.gateway import SEAFGateway, SecurityError


def test_pii_detection():
    gw = SEAFGateway()
    clean = "Write a quicksort function in Python."
    assert not gw.inspect_payload(clean)["has_pii"]

    sensitive = "Jane Doe has SSN 123-45-6789 and email jane@corp.internal"
    res = gw.inspect_payload(sensitive)
    assert res["has_pii"]
    assert "SSN" in res["detected_entities"]
    assert "EMAIL" in res["detected_entities"]
    print("[PASS] test_pii_detection")


def test_pii_routing_invariants():
    gw = SEAFGateway()
    # Tier 3 with PII must be FORCED local
    res = gw.route_request("Review payroll confidential for SSN 999-88-7777", task_complexity_tier=3)
    assert res["destination"] == "LOCAL_OFFLINE_EDGE"
    assert "PII detected" in res["routing_reason"]

    # Tier 3 without PII routes to cloud
    res_cloud = gw.route_request("Optimize multi-node distributed gradient sync algorithm", task_complexity_tier=3)
    assert res_cloud["destination"] == "SOVEREIGN_CLOUD"
    print("[PASS] test_pii_routing_invariants")


def test_sanitization_and_security_error():
    gw = SEAFGateway()
    raw = "Contact admin@corp.org or call 415-555-0199"
    sanitized = gw.sanitize_payload(raw)
    assert "admin@corp.org" not in sanitized
    assert "415-555-0199" not in sanitized

    # Verify execution
    out = gw.execute_routed_request("Review public documentation for NumPy", task_complexity_tier=3)
    assert out["destination"] in ("SOVEREIGN_CLOUD", "LOCAL_OFFLINE_EDGE")
    assert out["execution"]["status"] in ("SUCCESS", "FALLBACK_SUCCESS", "OFFLINE_STUB_SUCCESS")
    print("[PASS] test_sanitization_and_security_error")


if __name__ == "__main__":
    test_pii_detection()
    test_pii_routing_invariants()
    test_sanitization_and_security_error()
    print("All gateway tests passed successfully!")

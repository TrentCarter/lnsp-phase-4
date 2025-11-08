#!/usr/bin/env python3
"""
Model Broker Policy Validator

Validates model broker policy JSON files against the schema.
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any

try:
    import jsonschema
    from jsonschema import validate, ValidationError
except ImportError:
    print("ERROR: jsonschema not installed. Run: pip install jsonschema", file=sys.stderr)
    sys.exit(1)


def load_schema() -> Dict[str, Any]:
    """Load the model broker policy schema."""
    schema_path = Path(__file__).parent / "model_broker.schema.json"
    with open(schema_path) as f:
        return json.load(f)


def load_policy(policy_path: Path) -> Dict[str, Any]:
    """Load a policy file."""
    with open(policy_path) as f:
        return json.load(f)


def validate_policy(policy: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate a policy against the schema.

    Returns:
        (is_valid, errors) tuple
    """
    errors = []

    try:
        validate(instance=policy, schema=schema)
    except ValidationError as e:
        errors.append(f"Schema validation error: {e.message}")
        return False, errors

    # Additional semantic validations

    # Check version format
    version = policy.get("version", "")
    if not version or len(version.split("-")) != 4:
        errors.append(f"Invalid version format: {version} (expected YYYY-MM-DD-NNN)")

    # Check lane names are valid
    valid_lanes = {
        "Code-Impl", "Code-API-Design", "Data-Schema", "Data-Loader",
        "Vector-Ops", "Graph-Ops", "Narrative"
    }
    for lane in policy.get("lane_overrides", {}).keys():
        if lane not in valid_lanes:
            errors.append(f"Unknown lane: {lane} (valid: {', '.join(sorted(valid_lanes))})")

    # Check provider/model combinations are reasonable
    provider_models = {
        "anthropic": ["sonnet-4.5", "sonnet-4", "opus-4", "haiku-3.5"],
        "openai": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        "google": ["gemini-pro", "gemini-flash"],
        "local_llama": ["llama3.1:8b", "llama3.1:70b", "llama3:8b"]
    }

    def check_provider_model(provider: str, model: str, location: str):
        if provider in provider_models:
            known_models = provider_models[provider]
            if model not in known_models:
                errors.append(
                    f"WARNING: {location}: Unknown model '{model}' for provider '{provider}' "
                    f"(known: {', '.join(known_models)})"
                )

    # Check default provider
    default = policy.get("default_provider", {})
    if default:
        check_provider_model(
            default.get("provider", ""),
            default.get("model", ""),
            "default_provider"
        )

    # Check lane overrides
    for lane, config in policy.get("lane_overrides", {}).items():
        check_provider_model(
            config.get("provider", ""),
            config.get("model", ""),
            f"lane_overrides.{lane}"
        )

    # Check fallback chain
    for i, fallback in enumerate(policy.get("fallback_chain", [])):
        check_provider_model(
            fallback.get("provider", ""),
            fallback.get("model", ""),
            f"fallback_chain[{i}]"
        )

    return len([e for e in errors if not e.startswith("WARNING")]) == 0, errors


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <policy.json>", file=sys.stderr)
        print("\nValidates a model broker policy file against the schema.", file=sys.stderr)
        sys.exit(1)

    policy_path = Path(sys.argv[1])

    if not policy_path.exists():
        print(f"ERROR: Policy file not found: {policy_path}", file=sys.stderr)
        sys.exit(1)

    # Load schema and policy
    try:
        schema = load_schema()
        policy = load_policy(policy_path)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load files: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate
    is_valid, errors = validate_policy(policy, schema)

    # Print results
    print(f"Validating: {policy_path}")
    print(f"Version: {policy.get('version', 'MISSING')}")
    print(f"Default Provider: {policy.get('default_provider', {}).get('provider', 'MISSING')}")
    print(f"Lane Overrides: {len(policy.get('lane_overrides', {}))}")
    print()

    if is_valid:
        print("✅ Policy is VALID")
        if errors:
            print("\nWarnings:")
            for error in errors:
                if error.startswith("WARNING"):
                    print(f"  ⚠️  {error}")
        sys.exit(0)
    else:
        print("❌ Policy is INVALID")
        print("\nErrors:")
        for error in errors:
            if not error.startswith("WARNING"):
                print(f"  ❌ {error}")
        if any(e.startswith("WARNING") for e in errors):
            print("\nWarnings:")
            for error in errors:
                if error.startswith("WARNING"):
                    print(f"  ⚠️  {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()

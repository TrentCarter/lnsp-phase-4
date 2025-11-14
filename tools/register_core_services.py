#!/usr/bin/env python3
"""
Register core infrastructure services with the Registry Service (port 6121)

This script registers all core infrastructure services that don't have
self-registration code yet:
- TRON Heartbeat Monitor (6109)
- Resource Manager (6104)
- Token Governor (6105)
- Event Stream (6102)
- Router (6103)
- Aider-LCO (6130)

Usage:
    python tools/register_core_services.py
    python tools/register_core_services.py --service tron  # Register only TRON
"""

import requests
import sys
from typing import Dict, List, Optional

REGISTRY_URL = "http://localhost:6121"

# Core service definitions
CORE_SERVICES = {
    "tron": {
        "name": "TRON Heartbeat Monitor",
        "type": "agent",
        "role": "production",
        "url": "http://localhost:6109",
        "caps": ["health_monitoring", "timeout_detection", "service_alerts"],
        "labels": {
            "tier": "core",
            "category": "infrastructure",
            "port": 6109
        },
        "heartbeat_interval_s": 30,
        "ttl_s": 90
    },
    "resource_manager": {
        "name": "Resource Manager",
        "type": "agent",
        "role": "production",
        "url": "http://localhost:6104",
        "caps": ["resource_allocation", "pool_management", "programmer_assignment"],
        "labels": {
            "tier": "core",
            "category": "infrastructure",
            "port": 6104
        },
        "heartbeat_interval_s": 30,
        "ttl_s": 90
    },
    "token_governor": {
        "name": "Token Governor",
        "type": "agent",
        "role": "production",
        "url": "http://localhost:6105",
        "caps": ["token_tracking", "cost_monitoring", "budget_enforcement"],
        "labels": {
            "tier": "core",
            "category": "infrastructure",
            "port": 6105
        },
        "heartbeat_interval_s": 30,
        "ttl_s": 90
    },
    "event_stream": {
        "name": "Event Stream",
        "type": "agent",
        "role": "production",
        "url": "http://localhost:6102",
        "caps": ["event_broadcasting", "sse_streaming", "real_time_updates"],
        "labels": {
            "tier": "core",
            "category": "infrastructure",
            "port": 6102
        },
        "heartbeat_interval_s": 30,
        "ttl_s": 90
    },
    "router": {
        "name": "Router",
        "type": "agent",
        "role": "production",
        "url": "http://localhost:6103",
        "caps": ["request_routing", "load_balancing", "service_discovery"],
        "labels": {
            "tier": "core",
            "category": "infrastructure",
            "port": 6103
        },
        "heartbeat_interval_s": 30,
        "ttl_s": 90
    },
    "aider": {
        "name": "Aider-LCO",
        "type": "agent",
        "role": "production",
        "url": "http://localhost:6130",
        "caps": ["code_editing", "git_operations", "filesystem_access"],
        "labels": {
            "tier": "core",
            "category": "executor",
            "port": 6130
        },
        "heartbeat_interval_s": 30,
        "ttl_s": 90
    }
}


def check_service_health(url: str) -> bool:
    """Check if a service is responding to health checks"""
    try:
        response = requests.get(f"{url}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def register_service(service_key: str, service_config: Dict) -> Optional[str]:
    """
    Register a single service with the Registry

    Returns:
        service_id if successful, None otherwise
    """
    print(f"\n📝 Registering {service_config['name']}...")

    # Check if service is running
    if not check_service_health(service_config['url']):
        print(f"   ⚠️  WARNING: Service {service_config['name']} is not responding at {service_config['url']}")
        print(f"   ⚠️  Registering anyway, but service may be down")

    try:
        response = requests.post(
            f"{REGISTRY_URL}/register",
            json=service_config,
            timeout=5
        )
        response.raise_for_status()
        result = response.json()
        service_id = result.get('service_id')
        print(f"   ✅ Registered successfully (ID: {service_id})")
        return service_id
    except requests.exceptions.ConnectionError:
        print(f"   ❌ ERROR: Cannot connect to Registry Service at {REGISTRY_URL}")
        print(f"   ❌ Make sure Registry Service (port 6121) is running")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"   ❌ ERROR: Registration failed: {e}")
        if e.response is not None:
            print(f"   ❌ Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"   ❌ ERROR: Unexpected error: {e}")
        return None


def main():
    """Register core infrastructure services"""
    print("=" * 60)
    print("Core Infrastructure Service Registration")
    print("=" * 60)

    # Check if Registry is accessible
    try:
        response = requests.get(f"{REGISTRY_URL}/", timeout=2)
        response.raise_for_status()
        print(f"✅ Registry Service is accessible at {REGISTRY_URL}")
    except:
        print(f"❌ ERROR: Cannot connect to Registry Service at {REGISTRY_URL}")
        print(f"❌ Make sure the Registry Service (port 6121) is running:")
        print(f"   ./scripts/start_all_pas_services.sh")
        sys.exit(1)

    # Check for --service argument
    target_service = None
    if len(sys.argv) > 2 and sys.argv[1] == "--service":
        target_service = sys.argv[2]
        if target_service not in CORE_SERVICES:
            print(f"❌ ERROR: Unknown service '{target_service}'")
            print(f"   Available services: {', '.join(CORE_SERVICES.keys())}")
            sys.exit(1)
        print(f"\n🎯 Registering only: {target_service}")

    # Register services
    success_count = 0
    fail_count = 0

    services_to_register = {target_service: CORE_SERVICES[target_service]} if target_service else CORE_SERVICES

    for service_key, service_config in services_to_register.items():
        service_id = register_service(service_key, service_config)
        if service_id:
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"Registration Summary: {success_count} succeeded, {fail_count} failed")
    print("=" * 60)

    if fail_count > 0:
        sys.exit(1)

    print("\n✅ All services registered successfully!")
    print(f"\n📊 View registered services:")
    print(f"   curl {REGISTRY_URL}/services | python3 -m json.tool")
    print(f"\n🌐 Or visit the HMI Dashboard:")
    print(f"   http://localhost:6101 → System Status")


if __name__ == "__main__":
    main()

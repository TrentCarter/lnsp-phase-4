#!/usr/bin/env python3
"""
Agent Registration Script

Reads agent definitions from .claude/agents/ and registers them with Registry service.

Usage:
    python tools/register_agents.py [--dry-run] [--registry-url URL]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import httpx
from datetime import datetime


class AgentRegistrar:
    """Registers agents with the Registry service"""

    def __init__(self, registry_url: str = "http://localhost:6121", dry_run: bool = False):
        self.registry_url = registry_url
        self.dry_run = dry_run
        self.client = httpx.Client(timeout=10.0)

    def parse_agent_definition(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Parse agent definition from markdown file with YAML frontmatter"""
        try:
            content = file_path.read_text()

            # Extract YAML frontmatter
            if not content.startswith('---'):
                print(f"⚠️  {file_path.name}: No YAML frontmatter found")
                return None

            # Find end of frontmatter
            end_marker = content.find('\n---\n', 4)
            if end_marker == -1:
                print(f"⚠️  {file_path.name}: Invalid YAML frontmatter")
                return None

            yaml_content = content[4:end_marker]
            agent_data = yaml.safe_load(yaml_content)

            # Add file metadata
            agent_data['_file_path'] = str(file_path)
            agent_data['_loaded_at'] = datetime.utcnow().isoformat() + 'Z'

            return agent_data

        except Exception as e:
            print(f"❌ {file_path.name}: Failed to parse - {e}")
            return None

    def convert_to_service_registration(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert agent definition to Registry service registration format"""

        # Base registration
        # NOTE: Registry 'role' is production/staging/canary/experimental
        # Agent 'role' (coord/exec/system) goes into labels as 'agent_role'
        # NOTE: Agents are definitions, not running services, so use 24-hour TTL
        registration = {
            "service_id": f"agent-{agent_data['name']}",
            "name": agent_data.get('display_name', agent_data['name']),
            "type": "agent",
            "role": "production",  # Registry role (all agents are production by default)
            "url": f"internal://agents/{agent_data['name']}",  # Internal routing
            "caps": agent_data.get('capabilities', []),
            "labels": {
                "tier": str(agent_data.get('tier', 1)),
                "mode": agent_data.get('mode', 'task'),
                "agent_role": agent_data.get('role', 'exec')  # Agent type: coord/exec/system
            },
            "heartbeat_interval_s": 3600,  # 1 hour (agents don't heartbeat actively)
            "ttl_s": 86400  # 24 hours (agents are definitions, persist longer)
        }

        # Add parent/children for hierarchy
        if 'parent' in agent_data and agent_data['parent']:
            registration['labels']['parent'] = agent_data['parent']
        if 'children' in agent_data and agent_data['children']:
            registration['labels']['children'] = ','.join(agent_data['children'])

        # Add resource requirements
        if 'resources' in agent_data:
            resources = agent_data['resources']
            if 'token_budget' in resources:
                registration['labels']['token_target'] = str(resources['token_budget'].get('target_ratio', 0.5))
                registration['labels']['token_hard_max'] = str(resources['token_budget'].get('hard_max_ratio', 0.75))
            if 'cpu_cores' in resources:
                registration['labels']['cpu_cores'] = str(resources['cpu_cores'])
            if 'memory_mb' in resources:
                registration['labels']['memory_mb'] = str(resources['memory_mb'])
            if 'gpu_count' in resources:
                registration['labels']['gpu_count'] = str(resources['gpu_count'])

        # Add rights/permissions
        if 'rights' in agent_data:
            rights = agent_data['rights']
            for right, value in rights.items():
                registration['labels'][f'right_{right}'] = str(value)

        # Add model preferences
        if 'model_preferences' in agent_data:
            prefs = agent_data['model_preferences']
            if 'primary' in prefs:
                registration['labels']['models_primary'] = ','.join(prefs['primary'])
            if 'optimization' in prefs:
                registration['labels']['optimization'] = prefs['optimization']

        # Add metadata tags
        if 'metadata' in agent_data:
            meta = agent_data['metadata']
            if 'version' in meta:
                registration['labels']['version'] = meta['version']
            if 'tags' in meta:
                registration['labels']['tags'] = ','.join(meta['tags'])
            if 'port' in meta:
                registration['labels']['port'] = str(meta['port'])
                # Update URL for system agents with ports
                registration['url'] = f"http://localhost:{meta['port']}"
            if 'status' in meta:
                registration['labels']['status'] = meta['status']

        return registration

    def register_agent(self, agent_data: Dict[str, Any]) -> bool:
        """Register a single agent with Registry"""
        try:
            registration = self.convert_to_service_registration(agent_data)

            if self.dry_run:
                print(f"[DRY-RUN] Would register: {registration['name']} ({agent_data['role']})")
                print(f"          Capabilities: {', '.join(registration['caps'][:3])}...")
                return True

            response = self.client.post(
                f"{self.registry_url}/register",
                json=registration
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ {registration['name']}: Registered (TTL: {result.get('ttl_s', 'N/A')}s)")
                return True
            else:
                print(f"❌ {registration['name']}: Failed - {response.status_code} {response.text[:100]}")
                return False

        except httpx.ConnectError:
            print(f"❌ Cannot connect to Registry at {self.registry_url}")
            print("   Make sure Registry service is running: ./scripts/start_all_pas_services.sh")
            return False
        except Exception as e:
            print(f"❌ {agent_data['name']}: Registration failed - {e}")
            return False

    def register_all_agents(self, agents_dir: Path) -> Dict[str, int]:
        """Register all agents from directory structure"""
        stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0
        }

        # Find all agent definition files
        agent_files = sorted(agents_dir.rglob('*.md'))

        print(f"\n{'='*70}")
        print(f"Agent Registration")
        print(f"{'='*70}")
        print(f"Registry URL: {self.registry_url}")
        print(f"Agents Directory: {agents_dir}")
        print(f"Found {len(agent_files)} agent definitions")
        print(f"{'='*70}\n")

        for file_path in agent_files:
            stats['total'] += 1

            # Parse agent definition
            agent_data = self.parse_agent_definition(file_path)
            if agent_data is None:
                stats['skipped'] += 1
                continue

            # Register with Registry
            if self.register_agent(agent_data):
                stats['success'] += 1
            else:
                stats['failed'] += 1

        return stats

    def verify_registration(self) -> bool:
        """Verify agents are registered by querying Registry"""
        try:
            if self.dry_run:
                print("\n[DRY-RUN] Skipping verification")
                return True

            response = self.client.get(f"{self.registry_url}/services")
            if response.status_code == 200:
                data = response.json()
                services = data.get('items', [])
                agents = [s for s in services if s.get('type') == 'agent']

                print(f"\n{'='*70}")
                print(f"Verification")
                print(f"{'='*70}")
                print(f"Total services: {len(services)}")
                print(f"Registered agents: {len(agents)}")

                # Group by role
                by_role = {}
                for agent in agents:
                    role = agent.get('role', 'unknown')
                    by_role[role] = by_role.get(role, 0) + 1

                print(f"\nBreakdown by role:")
                for role, count in sorted(by_role.items()):
                    print(f"  {role}: {count}")

                print(f"{'='*70}\n")
                return True
            else:
                print(f"❌ Verification failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Verification error: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Register agents with Registry service')
    parser.add_argument('--registry-url', default='http://localhost:6121',
                        help='Registry service URL (default: http://localhost:6121)')
    parser.add_argument('--agents-dir', default='.claude/agents',
                        help='Agents directory (default: .claude/agents)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Parse agents but do not register')

    args = parser.parse_args()

    # Resolve paths
    agents_dir = Path(args.agents_dir)
    if not agents_dir.exists():
        print(f"❌ Agents directory not found: {agents_dir}")
        sys.exit(1)

    # Create registrar and register agents
    registrar = AgentRegistrar(registry_url=args.registry_url, dry_run=args.dry_run)
    stats = registrar.register_all_agents(agents_dir)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    print(f"Total agents found: {stats['total']}")
    print(f"Successfully registered: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"Skipped (parse errors): {stats['skipped']}")
    print(f"{'='*70}\n")

    # Verify registration
    if stats['success'] > 0:
        registrar.verify_registration()

    # Exit code
    if stats['failed'] > 0:
        sys.exit(1)
    elif stats['success'] == 0:
        print("⚠️  No agents were registered")
        sys.exit(1)
    else:
        print("✅ All agents registered successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()

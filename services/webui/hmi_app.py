"""
Flask HMI App (Port 6101)
Web dashboard for visualizing and controlling the PAS Agent Swarm.

Features:
- Agent hierarchy tree (D3.js visualization)
- Real-time status cards
- Resource monitoring
- Cost tracking
- Alert management
"""

from flask import Flask, render_template, jsonify, request, Response, stream_with_context
from flask_cors import CORS
import requests
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import time
import sqlite3
import os
import threading
import yaml
import sys
from pathlib import Path

# Add parent directory to path for importing agent_chat
sys.path.insert(0, str(Path(__file__).parent.parent / 'common'))
sys.path.insert(0, str(Path(__file__).parent))  # Add webui directory for llm_chat_db
from agent_chat import AgentChatClient
from comms_logger import CommsLogger, MessageType

# LLM Chat Database (SQLAlchemy ORM)
from llm_chat_db import (
    ConversationSession,
    Message,
    get_session as get_db_session
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'pas-hmi-secret'
app.config['TEMPLATES_AUTO_RELOAD'] = True  # Auto-reload templates when they change
CORS(app)

# Service endpoints
REGISTRY_URL = 'http://localhost:6121'
HEARTBEAT_MONITOR_URL = 'http://localhost:6109'
RESOURCE_MANAGER_URL = 'http://localhost:6104'
TOKEN_GOVERNOR_URL = 'http://localhost:6105'
EVENT_STREAM_URL = 'http://localhost:6102'
PROVIDER_ROUTER_URL = 'http://localhost:6103'
GATEWAY_URL = 'http://localhost:6120'
AGENT_STATUS_FILE = Path('configs/pas/agent_status.json')

# Track server start time for uptime calculation
SERVER_START_TIME = time.time()


def load_agent_status_data() -> Dict[str, Any]:
    """Load agent coverage metadata for the Agent Status tab."""
    if not AGENT_STATUS_FILE.exists():
        logger.warning("Agent status file missing at %s", AGENT_STATUS_FILE)
        return {
            'last_updated': None,
            'coverage': {},
            'tiers': [],
            'agents': [],
        }

    try:
        with open(AGENT_STATUS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as exc:
        logger.error("Failed to load agent status file: %s", exc)
        return {
            'last_updated': None,
            'coverage': {},
            'tiers': [],
            'agents': [],
            'error': str(exc),
        }

    flattened: List[Dict[str, Any]] = []
    for tier in data.get('tiers', []):
        for agent in tier.get('agents', []):
            flattened.append({
                **agent,
                'tier': tier.get('name'),
            })

    data['agents'] = flattened
    return data

@app.route('/api/agent-status', methods=['GET'])
def get_agent_status():
    """Expose agent chat coverage metadata for the Model Pool UI."""
    payload = load_agent_status_data()
    return jsonify({
        'status': 'ok',
        'last_updated': payload.get('last_updated'),
        'coverage': payload.get('coverage', {}),
        'tiers': payload.get('tiers', []),
        'agents': payload.get('agents', []),
        'error': payload.get('error'),
    })


@app.route('/api/agent-status/test', methods=['POST'])
def test_agent_endpoint():
    """Proxy a quick health check for an agent defined in agent_status.json."""
    try:
        data = request.get_json(force=True)
    except Exception:
        data = {}

    agent_id = data.get('agent_id')
    test_endpoint = data.get('test_endpoint')
    timeout_seconds = float(data.get('timeout', 3.0))

    if not agent_id:
        return jsonify({'status': 'error', 'message': 'agent_id is required'}), 400

    if not test_endpoint:
        return jsonify({'status': 'error', 'message': 'test_endpoint is required'}), 400

    try:
        response = requests.get(test_endpoint, timeout=timeout_seconds)
        payload = {}

        try:
            payload = response.json()
        except Exception:
            payload = {'raw': response.text}

        if response.status_code == 200:
            return jsonify({
                'status': 'ok',
                'agent_id': agent_id,
                'http_status': response.status_code,
                'response': payload,
            })

        return jsonify({
            'status': 'error',
            'agent_id': agent_id,
            'http_status': response.status_code,
            'response': payload,
        }), response.status_code

    except requests.exceptions.RequestException as exc:
        return jsonify({
            'status': 'error',
            'agent_id': agent_id,
            'message': str(exc),
        }), 500


@app.route('/')
def index():
    """Main dashboard"""
    return render_template('dashboard.html')


@app.route('/tree')
def tree_view():
    """Agent hierarchy tree view"""
    return render_template('tree.html')


@app.route('/sequencer')
def sequencer_view():
    """Sequencer timeline view"""
    return render_template('sequencer.html')


@app.route('/actions')
def actions_view():
    """Actions log view - hierarchical task flow"""
    return render_template('actions.html')


@app.route('/llm')
def llm_view():
    """LLM Task Interface - conversational AI chat page"""
    return render_template('llm.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    uptime_seconds = time.time() - SERVER_START_TIME
    return jsonify({
        'status': 'ok',
        'service': 'hmi_app',
        'port': 6101,
        'timestamp': datetime.now().isoformat(),
        'uptime_seconds': uptime_seconds,
        'server_start_time': SERVER_START_TIME
    })


@app.route('/api/services', methods=['GET'])
def get_services():
    """Get all registered services from Registry, or agents from action_logs if no services"""
    try:
        response = requests.get(f'{REGISTRY_URL}/services', timeout=5)
        response.raise_for_status()
        data = response.json()

        # Registry returns {"items": [...]} - extract list
        services_list = data.get('items', [])
        if not isinstance(services_list, list):
            services_list = []

        # If no services registered, fetch agents from action_logs
        if not services_list:
            try:
                import sqlite3
                import os
                db_path = os.path.join(
                    os.path.dirname(__file__),
                    '../../artifacts/registry/registry.db'
                )
                if os.path.exists(db_path):
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    # Get unique agents with their tier and last seen info
                    cursor.execute("""
                        SELECT DISTINCT
                            agent,
                            tier,
                            MAX(timestamp) as last_seen
                        FROM (
                            SELECT from_agent as agent, tier_from as tier, timestamp
                            FROM action_logs WHERE from_agent IS NOT NULL AND from_agent != 'user'
                            UNION ALL
                            SELECT to_agent as agent, tier_to as tier, timestamp
                            FROM action_logs WHERE to_agent IS NOT NULL AND to_agent != 'user'
                        )
                        GROUP BY agent, tier
                        ORDER BY tier, agent
                    """)
                    rows = cursor.fetchall()
                    conn.close()

                    # Format as service objects for compatibility
                    services_list = [
                        {
                            'service_id': row[0],
                            'name': row[0].replace('_', ' ').title(),
                            'status': 'historical',  # Indicate these are from action logs
                            'tier': row[1] if row[1] is not None else '?',
                            'last_seen': row[2],
                            'from_action_logs': True
                        }
                        for row in rows
                    ]
            except Exception as e:
                logger.warning(f"Could not fetch agents from action logs: {e}")

        return jsonify({'services': services_list})
    except Exception as e:
        logger.error(f"Error fetching services: {e}")
        return jsonify({'error': str(e), 'services': []}), 500


@app.route('/api/tree', methods=['GET'])
def get_tree_data():
    """
    Get agent hierarchy tree data.

    Query params:
        - source: 'services' (default) or 'actions'
        - task_id: Required if source='actions'

    Returns tree structure compatible with D3.js:
    {
        "name": "Root",
        "children": [...]
    }
    """
    source = request.args.get('source', 'services')
    task_id = request.args.get('task_id', None)

    try:
        if source == 'actions' and task_id:
            # Build tree from action logs for a specific task
            tree = build_tree_from_actions(task_id)
            return jsonify(tree)
        else:
            # Default: Build tree from registered services
            response = requests.get(f'{REGISTRY_URL}/services', timeout=5)
            response.raise_for_status()
            data = response.json()

            # Registry returns {"services": {...}} - convert dict to list
            services_dict = data.get('services', {})
            services = list(services_dict.values()) if isinstance(services_dict, dict) else []

            # Build tree structure
            tree = build_tree_from_services(services)

            return jsonify(tree)
    except Exception as e:
        logger.error(f"Error building tree: {e}")
        return jsonify({
            'name': 'Root',
            'children': [],
            'error': str(e)
        }), 500


def build_tree_from_services(services: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build hierarchical tree from flat service list using parent-child relationships.
    """
    if not services:
        return {
            'name': 'PAS Root',
            'status': 'idle',
            'children': []
        }

    # Create a lookup dictionary for all services
    nodes = {}

    # First pass: create all nodes
    for service in services:
        service_id = service.get('service_id', '')
        node = {
            'name': service.get('name', 'Unknown'),
            'service_id': service_id,
            'status': service.get('status', 'unknown'),
            'port': service.get('labels', {}).get('port') or service.get('port'),
            'last_heartbeat': service.get('last_heartbeat_ts'),
            'agent_role': service.get('labels', {}).get('agent_role', 'unknown'),
            'tier': service.get('labels', {}).get('tier', '?'),
            'children': []
        }
        nodes[service_id] = node

    # Second pass: build parent-child relationships
    root_children = []
    for service in services:
        service_id = service.get('service_id', '')
        # Support both 'parent' and 'parent_id' in labels
        parent_ref = service.get('labels', {}).get('parent_id') or service.get('labels', {}).get('parent', '')
        parent_ref = parent_ref.strip() if parent_ref else ''

        if not parent_ref or parent_ref == 'root' or parent_ref == 'null':
            # No parent or root parent - add to root
            if service_id in nodes:
                root_children.append(nodes[service_id])
        else:
            # parent_ref is the direct service_id of the parent
            if parent_ref in nodes:
                nodes[parent_ref]['children'].append(nodes[service_id])
            else:
                # Parent not found, add to root
                if service_id in nodes:
                    root_children.append(nodes[service_id])

    # Create root node
    root = {
        'name': 'PAS Agent Swarm',
        'status': 'running',
        'children': root_children
    }

    return root


def build_tree_from_actions(task_id: str) -> Dict[str, Any]:
    """
    Build hierarchical tree from action logs for a specific task.
    Reconstructs agent hierarchy from delegation patterns.

    NEW: Creates bidirectional tree showing complete round-trip:
    PAS Root → Dirs → Mgrs → Progs → [Results] → Mgrs → [Results] → Dirs → [Results] → Root
    """
    try:
        # Fetch action logs for this task
        response = requests.get(f'{REGISTRY_URL}/action_logs/task/{task_id}', timeout=5)
        response.raise_for_status()
        data = response.json()
        actions = data.get('actions', [])

        if not actions:
            return {
                'name': 'PAS Root',
                'status': 'idle',
                'children': []
            }

        # Extract unique agents and their tier info
        agent_info = {}

        def extract_agents(action_list):
            for action in action_list:
                # Record from_agent
                if action.get('from_agent') and action['from_agent'] != 'user':
                    agent_id = action['from_agent']
                    if agent_id not in agent_info:
                        agent_info[agent_id] = {
                            'name': agent_id.replace('_', ' ').title(),
                            'tier': action.get('tier_from', '?'),
                            'status': action.get('status', 'unknown'),
                            'children': []
                        }

                # Record to_agent
                if action.get('to_agent') and action['to_agent'] != 'user':
                    agent_id = action['to_agent']
                    if agent_id not in agent_info:
                        agent_info[agent_id] = {
                            'name': agent_id.replace('_', ' ').title(),
                            'tier': action.get('tier_to', '?'),
                            'status': action.get('status', 'unknown'),
                            'children': []
                        }

                # Recursively process children
                if action.get('children'):
                    extract_agents(action['children'])

        extract_agents(actions)

        # Build FORWARD hierarchy (task assignment: parent → child)
        # Track parent chain for reverse flow
        parent_chain_map = {}  # Maps agent_id -> list of parent agents up to root

        def collect_parent_chain(action, chain=[]):
            """Recursively collect parent chain for each agent"""
            from_agent = action.get('from_agent')
            to_agent = action.get('to_agent')

            if to_agent and to_agent not in ['user', None]:
                # Build chain: current agent's parents
                new_chain = [from_agent] + chain if from_agent and from_agent not in ['user', 'pas_root', None] else chain
                parent_chain_map[to_agent] = new_chain

            # Recurse for children
            for child in action.get('children', []):
                new_chain_for_child = [to_agent] + (parent_chain_map.get(to_agent, [])) if to_agent and to_agent not in ['user', None] else chain
                collect_parent_chain(child, new_chain_for_child)

        # First pass: collect all parent chains
        for action in actions:
            collect_parent_chain(action, [])

        def build_forward_hierarchy(action_list, depth=0):
            """Recursively build forward delegation tree"""
            nodes = []
            for action in action_list:
                if action.get('to_agent') and action['to_agent'] in agent_info:
                    to_agent = action['to_agent']

                    # Recurse to build children
                    forward_children = build_forward_hierarchy(action.get('children', []), depth + 1)

                    # Add result/status node at leaf (programmers)
                    if not forward_children and depth >= 2:  # Leaf node (programmer level)
                        # Get parent chain for this agent
                        parent_chain = parent_chain_map.get(to_agent, [])

                        result_node = {
                            'name': f"✓ Result: {action.get('action_name', 'Task')[:30]}...",
                            'agent_id': f"{to_agent}_result",
                            'tier': 'result',
                            'status': action.get('status', 'done'),
                            'children': build_reverse_hierarchy(parent_chain)  # Start reverse flow
                        }
                        forward_children = [result_node]

                    # Only show status for result/error nodes, not for regular agents
                    status = action.get('status', 'unknown')
                    agent_node = {
                        'name': agent_info[to_agent]['name'],
                        'agent_id': to_agent,
                        'tier': agent_info[to_agent]['tier'],
                        'status': status if status in ['error', 'failed'] else 'running',  # Hide status noise
                        'children': forward_children
                    }
                    nodes.append(agent_node)
            return nodes

        # Build REVERSE hierarchy (results flow back: child → parent)
        def build_reverse_hierarchy(parent_chain):
            """
            Build reverse flow showing results going back up the chain.
            parent_chain is a list of parent agents from immediate parent to root.
            """
            if not parent_chain:
                # Reached root - create final result node
                return [{
                    'name': '✅ Final Report',
                    'agent_id': 'pas_root_final_report',
                    'tier': 'report',
                    'status': 'completed',
                    'children': []
                }]

            # Build reverse chain: start from immediate parent, go up to root
            current_node = None
            for i, parent_id in enumerate(parent_chain):
                if parent_id not in agent_info:
                    continue

                parent_name = agent_info[parent_id]['name']
                parent_tier = agent_info[parent_id]['tier']

                # Create review/aggregation node for this parent (hide status noise)
                review_node = {
                    'name': f"📊 {parent_name} Review",
                    'agent_id': f"{parent_id}_review",
                    'tier': parent_tier,
                    'status': 'running',  # Hide status noise - only show errors/results
                    'children': [current_node] if current_node else []
                }

                # If this is the last parent (highest level), add final report
                if i == len(parent_chain) - 1:
                    final_report = {
                        'name': '✅ Report to PAS Root',
                        'agent_id': f"{parent_id}_final_report",
                        'tier': 'report',
                        'status': 'completed',
                        'children': []
                    }
                    review_node['children'] = [final_report]

                current_node = review_node

            return [current_node] if current_node else []

        # Find root action (usually from 'user', 'Gateway', or top-level)
        root_children = []
        for action in actions:
            # Include top-level actions (from Gateway, user, or no parent)
            if action.get('from_agent') in ['user', 'Gateway', None] or action.get('parent_log_id') is None:
                root_children.extend(build_forward_hierarchy([action]))

        # Create root node
        root = {
            'name': 'PAS Agent Swarm',
            'status': 'running',
            'children': root_children
        }

        return root

    except Exception as e:
        logger.error(f"Error building tree from actions: {e}")
        return {
            'name': 'PAS Root',
            'status': 'error',
            'children': [],
            'error': str(e)
        }


def apply_deduplication(tasks: list, mode: str = 'smart') -> list:
    """
    Apply deduplication logic to tasks based on the selected mode.

    Modes:
    - 'none': No deduplication (raw transparency)
    - 'smart': Dedup at programmer level only (keep manager coordination visible)
    - 'full': Dedup all duplicate tasks (most aggressive)

    Returns deduplicated task list.
    """
    if mode == 'none':
        return tasks  # Return raw tasks unchanged

    # Build deduplication key: (agent_id, task_name)
    seen_tasks = {}
    deduplicated = []

    for task in tasks:
        # Extract agent tier (prog_* = programmer, mgr_* = manager, dir_* = director)
        agent_id = task.get('agent_id', '')
        is_programmer = agent_id.startswith('prog_')
        is_manager = agent_id.startswith('mgr_')

        # Create dedup key
        dedup_key = (agent_id, task.get('name', ''))

        # Apply mode-specific logic
        if mode == 'smart':
            # Option 2: Only dedup programmers, keep manager coordination visible
            if is_programmer:
                if dedup_key not in seen_tasks:
                    seen_tasks[dedup_key] = task
                    deduplicated.append(task)
                # else: skip duplicate programmer task
            else:
                # Keep all manager/director tasks (shows coordination overhead)
                deduplicated.append(task)

        elif mode == 'full':
            # Option 1: Dedup everything (most efficient)
            if dedup_key not in seen_tasks:
                seen_tasks[dedup_key] = task
                deduplicated.append(task)
            # else: skip duplicate

    logger.info(f"[DEDUP] Mode={mode}, Original={len(tasks)}, Deduplicated={len(deduplicated)}")
    return deduplicated


def build_sequencer_from_actions(task_id: str) -> Dict[str, Any]:
    """
    Build sequencer timeline data from action logs for a specific task.
    Reconstructs agents and their task timeline from action history.
    """
    logger.info(f"[DEBUG] build_sequencer_from_actions called for task_id={task_id}")
    try:
        # Fetch action logs for this task
        response = requests.get(f'{REGISTRY_URL}/action_logs/task/{task_id}', timeout=5)
        response.raise_for_status()
        data = response.json()
        actions = data.get('actions', [])

        if not actions:
            return {
                'agents': [],
                'tasks': []
            }

        # Extract unique agents and build agent list
        agent_map = {}
        tier_order = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}

        def extract_agents_recursive(action_list):
            for action in action_list:
                # Extract from_agent
                if action.get('from_agent') and action['from_agent'] not in ['user', None]:
                    agent_id = action['from_agent']
                    if agent_id not in agent_map:
                        agent_map[agent_id] = {
                            'service_id': agent_id,
                            'name': agent_id.replace('_', ' ').title(),
                            'tier': action.get('tier_from', '?'),
                            'status': action.get('status', 'unknown'),
                            'agent_role': 'unknown'
                        }

                # Extract to_agent
                if action.get('to_agent') and action['to_agent'] not in ['user', None]:
                    agent_id = action['to_agent']
                    if agent_id not in agent_map:
                        agent_map[agent_id] = {
                            'service_id': agent_id,
                            'name': agent_id.replace('_', ' ').title(),
                            'tier': action.get('tier_to', '?'),
                            'status': action.get('status', 'unknown'),
                            'agent_role': 'unknown'
                        }

                # Recursively process children
                if action.get('children'):
                    extract_agents_recursive(action['children'])

        extract_agents_recursive(actions)

        # Sort agents by tier
        agents = sorted(agent_map.values(), key=lambda a: tier_order.get(str(a['tier']), 99))

        # Build task timeline from actions by matching delegate/report pairs
        tasks = []
        task_counter = {}

        # First pass: collect all actions in flat list
        all_actions = []

        def collect_actions_recursive(action_list):
            for action in action_list:
                all_actions.append(action)
                if action.get('children'):
                    collect_actions_recursive(action['children'])

        collect_actions_recursive(actions)

        # Helper to parse timestamp
        def parse_timestamp(timestamp_value):
            try:
                if isinstance(timestamp_value, (int, float)):
                    return float(timestamp_value)
                else:
                    return datetime.fromisoformat(str(timestamp_value).replace('Z', '+00:00')).timestamp()
            except Exception as e:
                logger.warning(f"Error parsing timestamp {timestamp_value}: {e}")
                return datetime.now().timestamp()

        # Build index of report actions by (from_agent, to_agent) for fast lookup
        report_actions = {}
        for action in all_actions:
            if action.get('action_type') == 'report':
                from_agent = action.get('from_agent')
                to_agent = action.get('to_agent')
                if from_agent and to_agent:
                    key = (from_agent, to_agent)
                    timestamp = parse_timestamp(action.get('timestamp'))
                    # Keep the latest report timestamp for this agent pair
                    if key not in report_actions or timestamp > report_actions[key]['timestamp']:
                        report_actions[key] = {
                            'timestamp': timestamp,
                            'status': action.get('status', 'completed'),
                            'action': action
                        }

        logger.info(f"Found {len(report_actions)} unique report actions")

        # Second pass: create tasks from delegate/code_generation actions
        # Match each with corresponding report action
        for action in all_actions:
            action_type = action.get('action_type', '')
            from_agent = action.get('from_agent')
            to_agent = action.get('to_agent')

            # Skip non-task actions (only process delegate/code_generation)
            # Don't create tasks from report actions - those are used for end_time matching only
            if not to_agent or to_agent == 'user' or action_type == 'report':
                continue

            # This action represents a task assignment to to_agent
            agent_id = to_agent
            action_name = action.get('action_name', action_type or 'Unknown')

            if agent_id not in task_counter:
                task_counter[agent_id] = 0
            task_counter[agent_id] += 1

            # Start time: when task was delegated
            start_time = parse_timestamp(action.get('timestamp', datetime.now().timestamp()))

            # End time: find matching report action (from child back to parent)
            end_time = None
            status = action.get('status', 'running')
            progress = 0.5
            default_duration = 8.0

            # Look for report from agent_id back to from_agent
            if from_agent:
                report_key = (agent_id, from_agent)
                if report_key in report_actions:
                    report_info = report_actions[report_key]
                    report_timestamp = report_info['timestamp']

                    # Only use report if it comes AFTER delegation
                    if report_timestamp >= start_time:
                        end_time = report_timestamp
                        status = report_info['status']
                        progress = 1.0
                        logger.debug(f"Matched report for {agent_id}: duration={end_time - start_time:.1f}s")

            # If no end_time found, apply defaults based on status
            if end_time is None:
                import time
                time_since_action = abs(time.time() - start_time)

                # Bug #3 fix: For historical tasks (>5 min old), assume completed with estimated duration
                # This prevents the frontend from using Date.now() which makes duration appear huge
                if time_since_action > 300:  # 5 minutes
                    end_time = start_time + default_duration
                    status = 'done'
                    progress = 1.0
                elif status == 'completed':
                    end_time = start_time + default_duration
                    progress = 1.0
                    status = 'done'
                elif status == 'error':
                    end_time = start_time + default_duration
                    progress = 1.0
                elif status == 'running' and time_since_action > 60:
                    # Stale running task (1-5 min old)
                    end_time = start_time + default_duration
                    progress = 0.8
                else:
                    # Active running task: no end_time
                    end_time = None
                    progress = 0.5
            else:
                # We found actual end_time from report action
                if status == 'completed':
                    status = 'done'
                    progress = 1.0

            tasks.append({
                'task_id': f"{agent_id}_{task_counter[agent_id]}",
                'agent_id': agent_id,
                'name': action_name,
                'status': status,
                'progress': progress,
                'start_time': start_time,
                'end_time': end_time,
                'from_agent': from_agent,
                'to_agent': to_agent,
                'action_type': action_type,
                'log_id': action.get('log_id'),  # Include log_id for arrow drawing
                'parent_log_id': action.get('parent_log_id')  # Include parent relationship
            })

        # POST-PROCESSING: Calculate actual durations based on sequential timing
        # Group tasks by agent_id and sort by start_time
        tasks_by_agent = {}
        for task in tasks:
            agent_id = task['agent_id']
            if agent_id not in tasks_by_agent:
                tasks_by_agent[agent_id] = []
            tasks_by_agent[agent_id].append(task)

        # For each agent, calculate actual durations by measuring gaps between task starts
        duration_adjustments = 0
        for agent_id, agent_tasks in tasks_by_agent.items():
            # Sort by start_time (when task was delegated)
            agent_tasks.sort(key=lambda t: t['start_time'])

            if len(agent_tasks) > 1:
                logger.debug(f"[Duration] Agent {agent_id}: {len(agent_tasks)} tasks")

            # Adjust end_times based on when next task starts
            for i, task in enumerate(agent_tasks):
                # Only adjust completed/error tasks (those with end_time already set)
                if task['end_time'] is not None:
                    old_duration = task['end_time'] - task['start_time']

                    # Look ahead to find next task
                    if i + 1 < len(agent_tasks):
                        next_task = agent_tasks[i + 1]
                        # Duration is from this task's start to next task's start
                        # This represents the actual time this task occupied the agent
                        actual_duration = next_task['start_time'] - task['start_time']
                        # Clamp duration to reasonable range (0.1s to 300s)
                        actual_duration = max(0.1, min(actual_duration, 300.0))
                        # Update end_time to reflect actual duration
                        task['end_time'] = task['start_time'] + actual_duration

                        if abs(actual_duration - old_duration) > 0.1:
                            duration_adjustments += 1
                            logger.debug(f"[Duration] {agent_id} '{task['name'][:30]}': {old_duration:.1f}s → {actual_duration:.1f}s")
                    else:
                        # Last task in sequence: keep default_duration (already set)
                        logger.debug(f"[Duration] {agent_id} '{task['name'][:30]}': {old_duration:.1f}s (last task, keeping default)")

        logger.info(f"Built {len(tasks)} tasks from action logs with sequential duration calculation ({duration_adjustments} durations adjusted)")

        # Apply deduplication if requested (will be controlled by frontend setting)
        # For now, return both deduplicated and raw data
        deduplicated_tasks = apply_deduplication(tasks, mode='smart')  # Mode: 'none', 'smart', 'full'

        # Extract project name from Gateway→PAS Root "Submit Prime Directive" action
        project_name = task_id  # Default to task_id if not found
        gateway_action = next((a for a in all_actions
                              if a.get('from_agent') == 'Gateway'
                              and a.get('to_agent') == 'PAS Root'), None)
        if gateway_action:
            action_text = gateway_action.get('action_name', '')
            # Action format: "Submit Prime Directive: <task description>"
            if action_text.startswith("Submit Prime Directive: "):
                project_name = action_text.replace("Submit Prime Directive: ", "")
            elif action_text:
                project_name = action_text

        # NEW: Fetch agent chat messages for this run and add to tasks timeline
        agent_chat_tasks = []
        try:
            # Initialize agent chat client
            chat_client = AgentChatClient()

            # Import asyncio to run async functions
            import asyncio

            # Fetch all threads for this run/task
            threads = asyncio.run(chat_client.get_threads_by_run(task_id))

            for thread in threads:
                # Fetch full thread with messages
                full_thread = asyncio.run(chat_client.get_thread(thread.thread_id))

                # Convert each message to a task entry for the sequencer timeline
                for msg in full_thread.messages:
                    msg_timestamp = parse_timestamp(msg.created_at)

                    # Determine task properties based on message type
                    if msg.message_type == 'delegation':
                        task_name = f"💬 Delegated: {msg.content[:50]}..."
                        task_color = '#3b82f6'  # Blue
                    elif msg.message_type == 'question':
                        urgency_icon = '🔴' if msg.metadata.get('urgency') == 'blocking' else '🟡' if msg.metadata.get('urgency') == 'important' else '⚪'
                        task_name = f"{urgency_icon} ❓ {msg.content[:50]}..."
                        task_color = '#f59e0b'  # Amber
                    elif msg.message_type == 'answer':
                        task_name = f"💡 {msg.content[:50]}..."
                        task_color = '#10b981'  # Green
                    elif msg.message_type == 'status':
                        progress_pct = msg.metadata.get('progress', 0)
                        task_name = f"📊 {msg.content[:40]}... ({progress_pct}%)"
                        task_color = '#6b7280'  # Gray
                    elif msg.message_type == 'completion':
                        task_name = f"✅ {msg.content[:50]}..."
                        task_color = '#10b981'  # Green
                    elif msg.message_type == 'error':
                        task_name = f"❌ {msg.content[:50]}..."
                        task_color = '#ef4444'  # Red
                    else:
                        task_name = f"📝 {msg.content[:50]}..."
                        task_color = '#9ca3af'  # Light gray

                    # Add agent chat message as a task entry
                    agent_chat_tasks.append({
                        'task_id': msg.message_id,
                        'agent_id': msg.from_agent,
                        'name': task_name,
                        'status': 'done',  # Messages are instantaneous
                        'progress': 1.0,
                        'start_time': msg_timestamp,
                        'end_time': msg_timestamp + 0.1,  # 100ms duration for visual representation
                        'from_agent': msg.from_agent,
                        'to_agent': msg.to_agent,
                        'action_type': f'agent_chat_{msg.message_type}',
                        'message_type': msg.message_type,
                        'thread_id': msg.thread_id,
                        'metadata': msg.metadata,
                        'color': task_color  # Custom color for rendering
                    })

                logger.info(f"Added {len(full_thread.messages)} agent chat messages from thread {thread.thread_id}")

        except Exception as e:
            logger.warning(f"Could not fetch agent chat messages: {e}")
            # Don't fail if agent chat is unavailable - just log and continue

        # Merge agent chat tasks with regular tasks
        all_tasks = tasks + agent_chat_tasks
        all_tasks_dedup = deduplicated_tasks + agent_chat_tasks  # Agent chat messages bypass deduplication

        logger.info(f"Total tasks: {len(all_tasks)} (including {len(agent_chat_tasks)} agent chat messages)")

        return {
            'agents': agents,
            'tasks': all_tasks,  # Raw tasks + agent chat messages
            'tasks_deduplicated': all_tasks_dedup,  # Deduplicated tasks + agent chat messages
            'project_name': project_name  # Bug #2 fix: Add project name to API response
        }

    except Exception as e:
        logger.error(f"Error building sequencer from actions: {e}")
        return {
            'agents': [],
            'tasks': [],
            'error': str(e)
        }


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get aggregated metrics from all services"""
    try:
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'services': {},
            'summary': {}
        }

        # Registry metrics
        try:
            resp = requests.get(f'{REGISTRY_URL}/health', timeout=2)
            resp.raise_for_status()
            metrics['services']['registry'] = resp.json()
        except:
            metrics['services']['registry'] = {'status': 'error'}

        # Heartbeat Monitor metrics
        try:
            resp = requests.get(f'{HEARTBEAT_MONITOR_URL}/health', timeout=2)
            resp.raise_for_status()
            metrics['services']['heartbeat_monitor'] = resp.json()
        except:
            metrics['services']['heartbeat_monitor'] = {'status': 'error'}

        # Resource Manager metrics
        try:
            resp = requests.get(f'{RESOURCE_MANAGER_URL}/health', timeout=2)
            resp.raise_for_status()
            metrics['services']['resource_manager'] = resp.json()
        except:
            metrics['services']['resource_manager'] = {'status': 'error'}

        # Token Governor metrics
        try:
            resp = requests.get(f'{TOKEN_GOVERNOR_URL}/health', timeout=2)
            resp.raise_for_status()
            metrics['services']['token_governor'] = resp.json()
        except:
            metrics['services']['token_governor'] = {'status': 'error'}

        # Event Stream metrics
        try:
            resp = requests.get(f'{EVENT_STREAM_URL}/health', timeout=2)
            resp.raise_for_status()
            metrics['services']['event_stream'] = resp.json()
        except:
            metrics['services']['event_stream'] = {'status': 'error'}

        # Calculate summary
        total_services = len(metrics['services'])
        healthy_services = sum(1 for s in metrics['services'].values() if s.get('status') == 'ok')

        # Count unique agents from action logs (for historical data)
        total_agents = 0
        try:
            import sqlite3
            import os
            db_path = os.path.join(
                os.path.dirname(__file__),
                '../../artifacts/registry/registry.db'
            )
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(DISTINCT agent) FROM (
                        SELECT from_agent as agent FROM action_logs WHERE from_agent IS NOT NULL AND from_agent != 'user'
                        UNION
                        SELECT to_agent as agent FROM action_logs WHERE to_agent IS NOT NULL AND to_agent != 'user'
                    )
                """)
                total_agents = cursor.fetchone()[0]
                conn.close()
        except Exception as e:
            logger.warning(f"Could not count agents from action logs: {e}")

        # Calculate uptime
        uptime_seconds = time.time() - SERVER_START_TIME

        metrics['summary'] = {
            'total_services': total_services,
            'healthy_services': healthy_services,
            'health_percentage': (healthy_services / total_services * 100) if total_services > 0 else 0,
            'total_agents': total_agents,  # Add agent count from action logs
            'uptime_seconds': uptime_seconds,
            'server_start_time': SERVER_START_TIME
        }

        return jsonify(metrics)

    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get current alerts from Heartbeat Monitor"""
    try:
        response = requests.get(f'{HEARTBEAT_MONITOR_URL}/health', timeout=5)
        response.raise_for_status()
        data = response.json()

        # Extract alert information
        alerts = []
        if data.get('unhealthy', 0) > 0:
            alerts.append({
                'type': 'heartbeat_miss',
                'severity': 'warning',
                'message': f"{data['unhealthy']} service(s) unhealthy",
                'timestamp': datetime.now().isoformat()
            })

        return jsonify({'alerts': alerts})

    except Exception as e:
        logger.error(f"Error fetching alerts: {e}")
        return jsonify({'error': str(e), 'alerts': []}), 500


@app.route('/api/costs', methods=['GET'])
def get_costs():
    """Get cost metrics from Gateway"""
    try:
        window = request.args.get('window', 'minute')

        # Get cost metrics from Gateway
        response = requests.get(f'{GATEWAY_URL}/metrics', params={'window': window}, timeout=5)
        response.raise_for_status()
        cost_metrics = response.json()

        # Get provider stats from Provider Router
        try:
            provider_response = requests.get(f'{PROVIDER_ROUTER_URL}/stats', timeout=5)
            provider_response.raise_for_status()
            provider_stats = provider_response.json()
        except:
            provider_stats = {}

        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'cost_metrics': cost_metrics,
            'provider_stats': provider_stats
        })

    except Exception as e:
        logger.error(f"Error fetching cost metrics: {e}")
        return jsonify({'error': str(e), 'cost_metrics': {}}), 500


@app.route('/api/llm/stats', methods=['GET'])
def get_llm_stats():
    """
    Get consolidated LLM statistics with token usage and costs.

    Aggregates data from:
    - llm_chat.db (Message.usage field for token counts)
    - Gateway metrics (cost data)
    - Provider Router (model info)
    - .env configuration (available API models)

    Returns per-model stats:
    {
        "models": {
            "llama3.1:8b": {
                "model_name": "Llama 3.1 8B",
                "provider": "ollama",
                "total_tokens": 125000,
                "input_tokens": 75000,
                "output_tokens": 50000,
                "total_cost_usd": 0.0,  # $0 for local models
                "message_count": 42,
                "session_count": 5
            },
            "claude-sonnet-4": {
                "model_name": "Claude Sonnet 4",
                "provider": "anthropic",
                "total_tokens": 50000,
                "input_tokens": 30000,
                "output_tokens": 20000,
                "total_cost_usd": 1.25,
                "message_count": 15,
                "session_count": 3
            }
        },
        "totals": {
            "total_tokens": 175000,
            "total_cost_usd": 1.25,
            "total_messages": 57,
            "total_sessions": 8
        }
    }
    """
    try:
        # Get available API models from environment configuration
        import os
        available_api_models = {}

        # OpenAI models
        if os.getenv('OPENAI_API_KEY') and os.getenv('OPENAI_API_KEY') != 'your_openai_api_key_here':
            openai_model = os.getenv('OPENAI_MODEL_NAME', 'gpt-4')
            available_api_models[openai_model] = {
                "model_name": openai_model,
                "provider": "openai",
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_cost_usd": 0.0,
                "message_count": 0,
                "session_count": 0,
                "is_free": False
            }

        # Anthropic Claude models
        if os.getenv('ANTHROPIC_API_KEY') and os.getenv('ANTHROPIC_API_KEY') != 'your_anthropic_api_key_here':
            for model_key in ['HIGH', 'MEDIUM', 'LOW']:
                model_name = os.getenv(f'ANTHROPIC_MODEL_NAME_{model_key}')
                if model_name:
                    available_api_models[model_name] = {
                        "model_name": model_name,
                        "provider": "anthropic",
                        "total_tokens": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_cost_usd": 0.0,
                        "message_count": 0,
                        "session_count": 0,
                        "is_free": False
                    }

        # Google Gemini models
        if os.getenv('GEMINI_API_KEY') and os.getenv('GEMINI_API_KEY') != 'your_gemini_api_key_here':
            for model_key in ['HIGH', 'MEDIUM', 'LOW']:
                model_name = os.getenv(f'GEMINI_MODEL_NAME_{model_key}')
                if model_name:
                    available_api_models[model_name] = {
                        "model_name": model_name,
                        "provider": "google",
                        "total_tokens": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_cost_usd": 0.0,
                        "message_count": 0,
                        "session_count": 0,
                        "is_free": False
                    }

        # DeepSeek models
        if os.getenv('DEEPSEEK_API_KEY') and os.getenv('DEEPSEEK_API_KEY') != 'your_deepseek_api_key_here':
            deepseek_model = 'deepseek-chat'  # Default DeepSeek model
            available_api_models[deepseek_model] = {
                "model_name": deepseek_model,
                "provider": "deepseek",
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_cost_usd": 0.0,
                "message_count": 0,
                "session_count": 0,
                "is_free": False
            }

        # Kimi K2 models (Moonshot AI)
        if os.getenv('KIMI_API_KEY') and os.getenv('KIMI_API_KEY') != 'your_kimi_api_key_here':
            # Add all 4 verified Kimi models
            kimi_models = [
                ('kimi-k2-turbo-preview', 'Kimi K2-TURBO-PREVIEW (128K)', True),  # Latest, recommended
                ('moonshot-v1-8k', 'Moonshot V1-8K', False),
                ('moonshot-v1-32k', 'Moonshot V1-32K', False),
                ('moonshot-v1-128k', 'Moonshot V1-128K', False),
            ]

            for model_id, display_name, is_default in kimi_models:
                available_api_models[model_id] = {
                    "model_name": model_id,
                    "provider": "kimi",
                    "display_name": display_name,
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_cost_usd": 0.0,
                    "message_count": 0,
                    "session_count": 0,
                    "is_free": False,
                    "is_default": is_default
                }

        # Start with available API models (initialize with 0 usage)
        model_stats = available_api_models.copy()

        db = get_db_session()

        # Query all messages with usage data
        from sqlalchemy import func
        messages = db.query(Message).filter(Message.usage_json.isnot(None)).all()

        # Aggregate stats by model (update or add to model_stats)
        total_tokens = 0
        total_messages = 0
        total_paid_tokens = 0
        total_free_tokens = 0

        for msg in messages:
            usage = msg.get_usage()
            if not usage:
                continue

            model_key = msg.model_name or "unknown"

            if model_key not in model_stats:
                # Model used but not in available_api_models (e.g., local Ollama models)
                model_stats[model_key] = {
                    "model_name": model_key,
                    "provider": "ollama",  # Default to ollama for local models
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_cost_usd": 0.0,
                    "message_count": 0,
                    "session_count": 0,
                    "session_ids": set(),
                    "is_free": True  # Local models are free
                }

            # Ensure session_ids exists for API models
            if "session_ids" not in model_stats[model_key]:
                model_stats[model_key]["session_ids"] = set()

            # Aggregate token counts
            input_tokens = usage.get('input_tokens', 0) or usage.get('prompt_tokens', 0)
            output_tokens = usage.get('output_tokens', 0) or usage.get('completion_tokens', 0)
            total_msg_tokens = usage.get('total_tokens', input_tokens + output_tokens)

            model_stats[model_key]["input_tokens"] += input_tokens
            model_stats[model_key]["output_tokens"] += output_tokens
            model_stats[model_key]["total_tokens"] += total_msg_tokens
            model_stats[model_key]["message_count"] += 1
            model_stats[model_key]["session_ids"].add(msg.session_id)

            total_tokens += total_msg_tokens
            total_messages += 1

        # Count unique sessions per model
        for model_key in model_stats:
            if "session_ids" in model_stats[model_key]:
                model_stats[model_key]["session_count"] = len(model_stats[model_key]["session_ids"])
                del model_stats[model_key]["session_ids"]  # Remove set (not JSON serializable)

        # Get cost data from Gateway
        total_cost = 0.0
        try:
            response = requests.get(f'{GATEWAY_URL}/metrics', timeout=5)
            response.raise_for_status()
            cost_data = response.json()
            total_cost = cost_data.get('total_cost_usd', 0.0)

            # Distribute costs proportionally by token count (if we don't have per-model costs)
            cost_per_provider = cost_data.get('cost_per_provider', {})
            if total_tokens > 0 and total_cost > 0:
                for model_key in model_stats:
                    model_tokens = model_stats[model_key]["total_tokens"]
                    model_stats[model_key]["total_cost_usd"] = round(
                        (model_tokens / total_tokens) * total_cost, 4
                    )
        except Exception as e:
            logger.warning(f"Could not fetch cost data from Gateway: {e}")

        # Calculate paid vs free tokens based on provider (not cost, since unused models have 0 cost)
        for model_key in model_stats:
            provider = model_stats[model_key]["provider"]
            model_tokens = model_stats[model_key]["total_tokens"]

            # API providers are ALWAYS paid (openai, anthropic, google, deepseek, kimi)
            # Local providers are free (ollama, unknown local models)
            if provider in ['openai', 'anthropic', 'google', 'deepseek', 'kimi']:
                # Paid model (API-based) - costs money even if not used yet
                total_paid_tokens += model_tokens
                model_stats[model_key]["is_free"] = False
            else:
                # Free model (local like Ollama)
                total_free_tokens += model_tokens
                model_stats[model_key]["is_free"] = True

        # Count total unique sessions
        total_sessions = db.query(func.count(ConversationSession.session_id)).scalar() or 0

        db.close()

        return jsonify({
            'models': model_stats,
            'totals': {
                'total_tokens': total_tokens,
                'paid_tokens': total_paid_tokens,
                'free_tokens': total_free_tokens,
                'total_cost_usd': round(total_cost, 4),
                'total_messages': total_messages,
                'total_sessions': total_sessions
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error fetching LLM stats: {e}")
        return jsonify({'error': str(e), 'models': {}, 'totals': {}}), 500


@app.route('/api/costs/receipts/<run_id>', methods=['GET'])
def get_receipts(run_id):
    """Get cost receipts for a specific run"""
    try:
        response = requests.get(f'{GATEWAY_URL}/receipts/{run_id}', timeout=5)
        response.raise_for_status()
        return jsonify(response.json())

    except Exception as e:
        logger.error(f"Error fetching receipts for run {run_id}: {e}")
        return jsonify({'error': str(e), 'receipts': []}), 500


@app.route('/api/costs/budget/<run_id>', methods=['GET'])
def get_budget(run_id):
    """Get budget status for a specific run"""
    try:
        response = requests.get(f'{GATEWAY_URL}/budget/{run_id}', timeout=5)
        response.raise_for_status()
        return jsonify(response.json())

    except Exception as e:
        logger.error(f"Error fetching budget for run {run_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/current-task', methods=['GET'])
def get_current_task():
    """
    Get the current active task (most recent running task).

    Returns:
    {
        "task": {
            "task_id": "...",
            "name": "...",
            "status": "running|blocked|done|error",
            "agent": "...",
            "progress": 0.0-1.0,
            "start_time": timestamp
        }
    }
    or
    {
        "task": null
    }
    """
    try:
        # Fetch recent events from Event Stream
        try:
            event_response = requests.get(f'{EVENT_STREAM_URL}/events/recent?limit=50', timeout=5)
            if event_response.status_code == 200:
                events = event_response.json().get('events', [])

                # Build task map from events
                task_map = {}
                for event in events:
                    event_type = event.get('event_type', '')
                    service_id = event.get('service_id', '')
                    timestamp = datetime.fromisoformat(event.get('timestamp', datetime.now().isoformat())).timestamp()

                    task_id = event.get('task_id') or event.get('run_id') or f"{service_id}_{timestamp}"

                    if task_id not in task_map:
                        task_map[task_id] = {
                            'task_id': task_id,
                            'name': event.get('task_name', event_type),
                            'status': 'running',
                            'agent': service_id,
                            'progress': 0.5,
                            'start_time': timestamp,
                            'end_time': None,
                            'last_update': timestamp
                        }

                    # Update task based on event type
                    task = task_map[task_id]
                    task['last_update'] = max(task['last_update'], timestamp)

                    if event_type in ['job_started', 'task_started']:
                        task['start_time'] = min(task.get('start_time', timestamp), timestamp)
                        task['status'] = 'running'
                    elif event_type in ['job_completed', 'task_completed']:
                        task['end_time'] = timestamp
                        task['status'] = 'done'
                        task['progress'] = 1.0
                    elif event_type == 'heartbeat':
                        progress = event.get('data', {}).get('progress')
                        if progress is not None:
                            task['progress'] = float(progress)
                        # Keep status as running if heartbeat received
                        if task['status'] not in ['done', 'error', 'failed']:
                            task['status'] = 'running'
                    elif event_type in ['blocked', 'waiting']:
                        task['status'] = 'blocked'
                    elif event_type in ['error', 'failed']:
                        task['status'] = 'error'
                        task['end_time'] = timestamp
                    elif 'approval' in event_type:
                        task['status'] = 'awaiting_approval'

                # Find most recent active task (not done/error)
                active_tasks = [t for t in task_map.values() if t['status'] in ['running', 'blocked', 'waiting', 'awaiting_approval']]

                if active_tasks:
                    # Sort by last_update, get most recent
                    current_task = max(active_tasks, key=lambda t: t['last_update'])
                    return jsonify({'task': current_task})

                # If no active tasks, check for most recently completed
                completed_tasks = [t for t in task_map.values() if t['status'] in ['done', 'error']]
                if completed_tasks:
                    # Only show recently completed (within last 10 seconds)
                    import time
                    now = time.time()
                    recent_completed = [t for t in completed_tasks if (now - t['last_update']) < 10]

                    if recent_completed:
                        current_task = max(recent_completed, key=lambda t: t['last_update'])
                        return jsonify({'task': current_task})

        except Exception as e:
            logger.warning(f"Error fetching current task from event stream: {e}")

        # No active task
        return jsonify({'task': None})

    except Exception as e:
        logger.error(f"Error fetching current task: {e}")
        return jsonify({'task': None, 'error': str(e)}), 500


@app.route('/api/sequencer', methods=['GET'])
def get_sequencer_data():
    """
    Get sequencer timeline data (agents and tasks).

    Query params:
        - source: 'services' (default) or 'actions'
        - task_id: Required if source='actions'

    Returns:
    {
        "agents": [
            {"service_id": "...", "name": "...", "tier": "...", "status": "..."},
            ...
        ],
        "tasks": [
            {
                "task_id": "...",
                "agent_id": "...",
                "name": "...",
                "status": "running|blocked|done|stuck|error",
                "progress": 0.0-1.0,
                "start_time": unix_timestamp,
                "end_time": unix_timestamp or null
            },
            ...
        ]
    }
    """
    source = request.args.get('source', 'services')
    task_id = request.args.get('task_id', None)

    try:
        if source == 'actions' and task_id:
            # Build sequencer data from action logs
            return jsonify(build_sequencer_from_actions(task_id))
        else:
            # Default: Build from registered services
            # Fetch agents from Registry
            response = requests.get(f'{REGISTRY_URL}/services', timeout=5)
            response.raise_for_status()
            data = response.json()

            # Registry returns {"services": {...}} - convert dict to list
            services_dict = data.get('services', {})
            services = list(services_dict.values()) if isinstance(services_dict, dict) else []

            # Format agents for sequencer (sorted by tier: VP -> Directors -> Managers -> Workers)
            tier_order = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
            agents = []
            for service in services:
                agents.append({
                    'service_id': service.get('service_id', ''),
                    'name': service.get('name', 'Unknown'),
                    'tier': service.get('labels', {}).get('tier', '?'),
                    'status': service.get('status', 'unknown'),
                    'agent_role': service.get('labels', {}).get('agent_role', 'unknown')
                })

            # Sort by tier (VP at top, Workers at bottom)
            agents.sort(key=lambda a: tier_order.get(a['tier'], 99))

        # Fetch tasks from Event Stream (recent events)
        tasks = []
        try:
            event_response = requests.get(f'{EVENT_STREAM_URL}/events/recent?limit=100', timeout=5)
            if event_response.status_code == 200:
                events = event_response.json().get('events', [])

                # Convert events to tasks
                task_map = {}
                for event in events:
                    event_type = event.get('event_type', '')
                    service_id = event.get('service_id', '')
                    timestamp = datetime.fromisoformat(event.get('timestamp', datetime.now().isoformat())).timestamp()

                    task_id = event.get('task_id') or event.get('run_id') or f"{service_id}_{timestamp}"

                    if task_id not in task_map:
                        task_map[task_id] = {
                            'task_id': task_id,
                            'agent_id': service_id,
                            'name': event.get('task_name', event_type),
                            'status': 'running',
                            'progress': 0.5,
                            'start_time': timestamp,
                            'end_time': None
                        }

                    # Update task based on event type
                    task = task_map[task_id]
                    if event_type in ['job_started', 'task_started']:
                        task['start_time'] = min(task['start_time'], timestamp)
                        task['status'] = 'running'
                    elif event_type in ['job_completed', 'task_completed']:
                        task['end_time'] = timestamp
                        task['status'] = 'done'
                        task['progress'] = 1.0
                    elif event_type == 'heartbeat':
                        # Update progress from heartbeat
                        progress = event.get('data', {}).get('progress')
                        if progress is not None:
                            task['progress'] = float(progress)
                    elif event_type in ['blocked', 'waiting']:
                        task['status'] = 'blocked'
                    elif event_type in ['error', 'failed']:
                        task['status'] = 'error'
                        task['end_time'] = timestamp
                    elif 'approval' in event_type:
                        task['status'] = 'awaiting_approval'

                tasks = list(task_map.values())
        except Exception as e:
            logger.warning(f"Error fetching tasks from event stream: {e}")

        # Also fetch tasks from PAS (if available)
        try:
            import time
            pas_response = requests.get(
                f'http://localhost:6200/pas/v1/runs/status',
                params={'run_id': 'run-d63a3969'},
                timeout=2
            )
            if pas_response.status_code == 200:
                pas_data = pas_response.json()
                pas_tasks = pas_data.get('tasks', [])
                base_time = time.time() - 60  # 1 min ago

                for i, pas_task in enumerate(pas_tasks):
                    tasks.append({
                        'task_id': pas_task['task_id'],
                        'agent_id': f"programmer-{i+1:03d}",
                        'name': pas_task['lane'],
                        'status': 'done' if pas_task['status'] in ['succeeded', 'failed'] else 'running',
                        'progress': 1.0 if pas_task['status'] in ['succeeded', 'failed'] else 0.5,
                        'start_time': base_time + (i * 5),
                        'end_time': base_time + (i * 5) + 10 if pas_task['status'] in ['succeeded', 'failed'] else None
                    })
        except Exception as e:
            logger.warning(f"Error fetching tasks from PAS: {e}")

        # Generate mock tasks for demo if no tasks found
        if not tasks:
            import time
            now = time.time()
            for i, agent in enumerate(agents[:5]):  # First 5 agents
                tasks.append({
                    'task_id': f"demo_task_{i}",
                    'agent_id': agent['service_id'],
                    'name': f"Demo Task {i+1}",
                    'status': ['running', 'blocked', 'done', 'running', 'awaiting_approval'][i % 5],
                    'progress': [0.3, 0.6, 1.0, 0.8, 0.5][i % 5],
                    'start_time': now - (3600 - i * 600),  # Tasks starting over last hour
                    'end_time': now - (1800 - i * 300) if i % 3 == 0 else None
                })

        return jsonify({
            'agents': agents,
            'tasks': tasks,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error fetching sequencer data: {e}")
        return jsonify({
            'error': str(e),
            'agents': [],
            'tasks': []
        }), 500


@app.route('/api/actions/tasks', methods=['GET'])
def get_action_tasks():
    """Get all tasks from Registry action_logs (with fallbacks to PAS and demo data)"""
    try:
        # Try Registry action_logs FIRST (source of truth)
        try:
            registry_response = requests.get(f'{REGISTRY_URL}/action_logs/tasks', timeout=5)
            if registry_response.status_code == 200:
                data = registry_response.json()
                tasks = data.get('items', [])

                # Return Registry data if available
                if len(tasks) > 0:
                    return jsonify({'items': tasks, 'count': len(tasks)})
        except Exception as e:
            logger.warning(f"Registry action_logs unavailable: {e}")

        # Try PAS second (if available)
        try:
            pas_response = requests.get(
                'http://localhost:6200/pas/v1/runs/status',
                params={'run_id': 'run-d63a3969'},
                timeout=2
            )
            if pas_response.status_code == 200:
                data = pas_response.json()
                tasks = data.get('tasks', [])

                # Enrich with project info
                for task in tasks:
                    task['run_id'] = 'run-d63a3969'
                    task['project_name'] = 'REST API with PostgreSQL Backend'

                return jsonify({'items': tasks, 'count': len(tasks)})
        except:
            pass  # PAS not available, fall back to event stream

        # Fallback: Use sequencer data (event stream + demo tasks)
        event_response = requests.get(f'{EVENT_STREAM_URL}/events/recent?limit=100', timeout=5)
        tasks = []

        if event_response.status_code == 200:
            events = event_response.json().get('events', [])

            # Build task map from events
            task_map = {}
            for event in events:
                event_type = event.get('event_type', '')
                service_id = event.get('service_id', '')
                timestamp = datetime.fromisoformat(event.get('timestamp', datetime.now().isoformat())).timestamp()

                task_id = event.get('task_id') or event.get('run_id') or f"{service_id}_{timestamp}"

                if task_id not in task_map:
                    task_map[task_id] = {
                        'task_id': task_id,
                        'lane': event.get('task_name', event_type),
                        'status': 'running',
                        'agent': service_id,
                        'start_time': timestamp,
                        'end_time': None,
                        'run_id': 'demo-run',
                        'project_name': 'Agent Swarm Activity'
                    }

                # Update task based on event type
                task = task_map[task_id]
                if event_type in ['job_completed', 'task_completed']:
                    task['end_time'] = timestamp
                    task['status'] = 'succeeded'
                elif event_type in ['error', 'failed']:
                    task['status'] = 'failed'
                    task['end_time'] = timestamp

            tasks = list(task_map.values())

        # Only show demo tasks if explicitly requested (not when DB is just empty)
        # Check if demo mode is enabled via query param
        show_demo = request.args.get('demo', 'false').lower() == 'true'

        if len(tasks) == 0 and show_demo:
            import time
            base_time = time.time() - 3600
            demo_tasks = [
                {
                    'task_id': 'demo-1',
                    'lane': 'Database Setup',
                    'status': 'succeeded',
                    'start_time': base_time,
                    'end_time': base_time + 300,
                    'action_count': 5,
                    'agents_involved': ['programmer-001']
                },
                {
                    'task_id': 'demo-2',
                    'lane': 'API Endpoints',
                    'status': 'succeeded',
                    'start_time': base_time + 300,
                    'end_time': base_time + 900,
                    'action_count': 8,
                    'agents_involved': ['programmer-001', 'programmer-002']
                },
                {
                    'task_id': 'demo-3',
                    'lane': 'Authentication',
                    'status': 'running',
                    'start_time': base_time + 900,
                    'end_time': None,
                    'action_count': 3,
                    'agents_involved': ['programmer-002']
                }
            ]
            for task in demo_tasks:
                task['run_id'] = 'demo-run'
                task['project_name'] = 'REST API Development'
                task['agent'] = task['agents_involved'][0] if task['agents_involved'] else 'unknown'
            tasks = demo_tasks

        # Ensure all tasks have required fields for Actions page
        import time
        current_time = time.time()
        for task in tasks:
            if 'action_count' not in task:
                task['action_count'] = 1
            if 'agents_involved' not in task:
                task['agents_involved'] = [task.get('agent', 'unknown')]
            # Ensure timestamps are valid (not None, not string, not invalid)
            if not task.get('start_time') or not isinstance(task.get('start_time'), (int, float)):
                task['start_time'] = current_time
            if task.get('end_time') is not None and not isinstance(task.get('end_time'), (int, float)):
                task['end_time'] = None  # Reset invalid end_time to None

        return jsonify({'items': tasks, 'count': len(tasks)})

    except Exception as e:
        logger.error(f"Error fetching action tasks: {e}")
        return jsonify({'error': str(e), 'items': []}), 500


@app.route('/api/actions/task/<task_id>', methods=['GET'])
def get_task_actions(task_id):
    """Get hierarchical actions for a specific task"""
    try:
        # Try to fetch from Registry first
        try:
            response = requests.get(f'{REGISTRY_URL}/action_logs/task/{task_id}', timeout=2)
            if response.status_code == 200:
                return jsonify(response.json())
        except:
            pass  # Registry not available or task not found

        # Only return demo actions if demo mode is enabled
        show_demo = request.args.get('demo', 'false').lower() == 'true'
        if task_id.startswith('demo-') and show_demo:
            import time
            now = time.time()

            demo_actions = {
                'demo-1': [
                    {
                        'log_id': 101,
                        'task_id': task_id,
                        'action_type': 'database_setup',
                        'action_name': 'Initialize Database Schema',
                        'from_agent': 'manager-001',
                        'to_agent': 'programmer-001',
                        'status': 'completed',
                        'timestamp': now - 3300,
                        'action_data': {'database': 'postgresql', 'schema_version': '1.0'},
                        'children': [
                            {
                                'log_id': 102,
                                'task_id': task_id,
                                'action_type': 'sql_execution',
                                'action_name': 'Create Tables',
                                'from_agent': 'programmer-001',
                                'to_agent': None,
                                'status': 'completed',
                                'timestamp': now - 3200,
                                'action_data': {'tables': ['users', 'posts', 'comments']},
                                'children': []
                            },
                            {
                                'log_id': 103,
                                'task_id': task_id,
                                'action_type': 'sql_execution',
                                'action_name': 'Create Indexes',
                                'from_agent': 'programmer-001',
                                'to_agent': None,
                                'status': 'completed',
                                'timestamp': now - 3100,
                                'action_data': {'indexes': 3},
                                'children': []
                            }
                        ]
                    },
                    {
                        'log_id': 104,
                        'task_id': task_id,
                        'action_type': 'test',
                        'action_name': 'Run Database Tests',
                        'from_agent': 'programmer-001',
                        'to_agent': None,
                        'status': 'completed',
                        'timestamp': now - 3000,
                        'action_data': {'tests_passed': 12, 'tests_failed': 0},
                        'children': []
                    }
                ],
                'demo-2': [
                    {
                        'log_id': 201,
                        'task_id': task_id,
                        'action_type': 'api_implementation',
                        'action_name': 'Implement REST Endpoints',
                        'from_agent': 'manager-001',
                        'to_agent': 'programmer-001',
                        'status': 'completed',
                        'timestamp': now - 2700,
                        'action_data': {'endpoints': ['/users', '/posts', '/comments']},
                        'children': [
                            {
                                'log_id': 202,
                                'task_id': task_id,
                                'action_type': 'code_generation',
                                'action_name': 'Generate GET /users',
                                'from_agent': 'programmer-001',
                                'to_agent': None,
                                'status': 'completed',
                                'timestamp': now - 2650,
                                'action_data': {'method': 'GET', 'route': '/users'},
                                'children': []
                            },
                            {
                                'log_id': 203,
                                'task_id': task_id,
                                'action_type': 'code_generation',
                                'action_name': 'Generate POST /users',
                                'from_agent': 'programmer-001',
                                'to_agent': None,
                                'status': 'completed',
                                'timestamp': now - 2600,
                                'action_data': {'method': 'POST', 'route': '/users'},
                                'children': []
                            }
                        ]
                    },
                    {
                        'log_id': 204,
                        'task_id': task_id,
                        'action_type': 'code_review',
                        'action_name': 'Review API Implementation',
                        'from_agent': 'programmer-002',
                        'to_agent': 'programmer-001',
                        'status': 'completed',
                        'timestamp': now - 2400,
                        'action_data': {'comments': 2, 'approved': True},
                        'children': []
                    }
                ],
                'demo-3': [
                    {
                        'log_id': 301,
                        'task_id': task_id,
                        'action_type': 'auth_implementation',
                        'action_name': 'Implement JWT Authentication',
                        'from_agent': 'manager-001',
                        'to_agent': 'programmer-002',
                        'status': 'running',
                        'timestamp': now - 900,
                        'action_data': {'auth_type': 'JWT', 'token_expiry': '1h'},
                        'children': [
                            {
                                'log_id': 302,
                                'task_id': task_id,
                                'action_type': 'code_generation',
                                'action_name': 'Generate Login Endpoint',
                                'from_agent': 'programmer-002',
                                'to_agent': None,
                                'status': 'completed',
                                'timestamp': now - 850,
                                'action_data': {'route': '/auth/login'},
                                'children': []
                            },
                            {
                                'log_id': 303,
                                'task_id': task_id,
                                'action_type': 'code_generation',
                                'action_name': 'Generate Token Verification Middleware',
                                'from_agent': 'programmer-002',
                                'to_agent': None,
                                'status': 'running',
                                'timestamp': now - 300,
                                'action_data': {'middleware': 'verify_token'},
                                'children': []
                            }
                        ]
                    }
                ]
            }

            actions = demo_actions.get(task_id, [])
            return jsonify({'task_id': task_id, 'actions': actions})

        # Task not found
        return jsonify({'error': 'Task not found', 'task_id': task_id, 'actions': []}), 404

    except Exception as e:
        logger.error(f"Error fetching actions for task {task_id}: {e}")
        return jsonify({'error': str(e), 'task_id': task_id, 'actions': []}), 500


@app.route('/api/actions/log', methods=['POST'])
def log_action():
    """Log a new action (proxy to Registry service)"""
    try:
        action_data = request.get_json()
        response = requests.post(f'{REGISTRY_URL}/action_logs', json=action_data, timeout=5)
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        logger.error(f"Error logging action: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/actions/task/<task_id>', methods=['DELETE'])
def delete_task_actions(task_id):
    """Delete all actions for a specific task (proxy to Registry service)"""
    try:
        response = requests.delete(f'{REGISTRY_URL}/action_logs/task/{task_id}', timeout=5)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return jsonify({'error': f'Task {task_id} not found'}), 404
        logger.error(f"Error deleting task {task_id}: {e}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Error deleting task {task_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/actions/tasks/delete', methods=['POST'])
def delete_multiple_tasks():
    """Delete multiple tasks (proxy to Registry service)"""
    try:
        data = request.get_json()
        task_ids = data.get('task_ids', [])

        if not task_ids:
            return jsonify({'error': 'No task IDs provided'}), 400

        response = requests.delete(
            f'{REGISTRY_URL}/action_logs/tasks',
            json=task_ids,
            timeout=5
        )
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        logger.error(f"Error deleting multiple tasks: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/actions/projects', methods=['GET'])
def get_projects():
    """Get list of all unique task_ids (projects) with metadata"""
    try:
        import sqlite3
        import os

        db_path = os.path.join(
            os.path.dirname(__file__),
            '../../artifacts/registry/registry.db'
        )

        if not os.path.exists(db_path):
            return jsonify({'projects': []})

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get unique task_ids with metadata (most recent timestamp, action count)
        cursor.execute("""
            SELECT
                task_id,
                MIN(timestamp) as first_action,
                MAX(timestamp) as last_action,
                COUNT(*) as action_count,
                MAX(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as is_running
            FROM action_logs
            GROUP BY task_id
            ORDER BY last_action DESC
        """)

        projects = []
        for row in cursor.fetchall():
            task_id, first_action, last_action, action_count, is_running = row

            # Fetch task_name from Gateway submission (from action_name field)
            cursor.execute("""
                SELECT action_name
                FROM action_logs
                WHERE task_id = ? AND from_agent = 'Gateway' AND to_agent = 'PAS Root'
                ORDER BY timestamp ASC
                LIMIT 1
            """, (task_id,))
            task_name_row = cursor.fetchone()

            # Extract task name from action_name field
            task_name = task_id  # Default to task_id if not found
            if task_name_row and task_name_row[0]:
                action_text = task_name_row[0]
                # Action format: "Submit Prime Directive: <task description>"
                if action_text.startswith("Submit Prime Directive: "):
                    task_name = action_text  # Use full text with prefix
                else:
                    task_name = action_text

            projects.append({
                'task_id': task_id,
                'task_name': task_name,
                'first_action': first_action,
                'last_action': last_action,
                'action_count': action_count,
                'is_running': bool(is_running),
                'status': 'running' if is_running else 'completed'
            })

        conn.close()

        return jsonify({'projects': projects, 'count': len(projects)})

    except Exception as e:
        logger.error(f"Error fetching projects: {e}")
        return jsonify({'error': str(e), 'projects': []}), 500


# ============================================================================
# PAS Integration for Tasks
# ============================================================================

PAS_URL = 'http://localhost:6200'


@app.route('/api/demo/prime-directive', methods=['POST'])
def start_prime_directive():
    """Send a Prime Directive to PAS Root (real operational flow)"""
    try:
        import time
        import uuid
        import requests
        
        data = request.get_json() or {}
        project_prompt = data.get('prompt', '')
        
        if not project_prompt:
            # Clever default prompt - same scope as current demo but operational
            project_prompt = """Build a comprehensive REST API system with the following requirements:

CORE FEATURES:
- User authentication with JWT tokens
- Role-based access control (admin, user, moderator)
- RESTful endpoints for CRUD operations
- PostgreSQL database with proper migrations
- Input validation and error handling
- API documentation with OpenAPI/Swagger
- Unit and integration tests
- Docker containerization
- CI/CD pipeline configuration

TECHNICAL STACK:
- Backend: FastAPI or Express.js
- Database: PostgreSQL with SQLAlchemy/Prisma
- Authentication: JWT with refresh tokens
- Testing: pytest/Jest with coverage reports
- Documentation: OpenAPI 3.0 spec
- Deployment: Docker with docker-compose

DELIVERABLES:
1. Complete source code with modular architecture
2. Database schema and migration scripts
3. API documentation and testing suite
4. Docker configuration and deployment scripts
5. CI/CD pipeline setup (GitHub Actions/GitLab CI)

The system should be production-ready, secure, and scalable."""

        # Generate unique run ID
        import uuid
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        
        # Prepare Prime Directive payload for PAS Root
        payload = {
            "project_id": 42,
            "run_id": run_id,
            "prime_directive": {
                "title": "REST API with Authentication & Testing",
                "description": project_prompt,
                "priority": "high",
                "requirements": [
                    "User authentication system",
                    "REST API endpoints", 
                    "Database integration",
                    "Testing coverage",
                    "Documentation",
                    "Containerization"
                ]
            },
            "execution_profile": {
                "mode": "hierarchical",
                "max_parallel_tasks": 8,
                "timeout_minutes": 30,
                "auto_approve": True
            },
            "metadata": {
                "source": "hmi_demo",
                "timestamp": time.time(),
                "user": "demo_user"
            }
        }
        
        # Send to PAS Root (port 6200)
        logger.info(f"🚀 Sending Prime Directive to PAS Root: {run_id}")
        response = requests.post(
            f'{PAS_URL}/pas/v1/runs/start',
            json=payload,
            headers={
                'Content-Type': 'application/json',
                'Idempotency-Key': str(uuid.uuid4())
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Prime Directive accepted: {result}")
            return jsonify({
                'message': 'Prime Directive sent to PAS Root',
                'run_id': run_id,
                'pas_response': result,
                'prompt_preview': project_prompt[:200] + '...' if len(project_prompt) > 200 else project_prompt
            })
        else:
            logger.error(f"❌ PAS Root rejected: {response.status_code} - {response.text}")
            return jsonify({
                'error': f'PAS Root error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except requests.exceptions.ConnectionError:
        logger.error("❌ Cannot connect to PAS Root on port 6200")
        return jsonify({
            'error': 'PAS Root not available',
            'details': 'Ensure PAS services are running on port 6200'
        }), 503
    except Exception as e:
        logger.error(f"❌ Prime Directive error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/demo/start', methods=['POST'])
def start_demo():
    """Start the live demo"""
    try:
        import subprocess
        import os
        import signal

        # Check if demo is already running
        demo_pid_file = '/tmp/lnsp_demo.pid'
        if os.path.exists(demo_pid_file):
            with open(demo_pid_file, 'r') as f:
                pid = int(f.read().strip())
                try:
                    os.kill(pid, 0)  # Check if process exists
                    return jsonify({'error': 'Demo is already running', 'pid': pid}), 400
                except OSError:
                    os.remove(demo_pid_file)

        # Start realistic demo with proper comms logging
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        demo_script = os.path.join(project_root, 'scripts', 'demo_realistic_project.py')

        # Use project venv python
        python_exe = os.path.join(project_root, '.venv', 'bin', 'python3')
        if not os.path.exists(python_exe):
            python_exe = 'python3'  # Fallback to system python

        process = subprocess.Popen(
            [python_exe, demo_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            cwd=project_root
        )

        # Save PID
        with open(demo_pid_file, 'w') as f:
            f.write(str(process.pid))

        return jsonify({'message': 'Demo started', 'pid': process.pid})
    except Exception as e:
        logger.error(f"Error starting demo: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/demo/stop', methods=['POST'])
def stop_demo():
    """Stop the live demo"""
    try:
        import os
        import signal

        demo_pid_file = '/tmp/lnsp_demo.pid'
        if not os.path.exists(demo_pid_file):
            return jsonify({'message': 'Demo is not running'})

        with open(demo_pid_file, 'r') as f:
            pid = int(f.read().strip())

        try:
            os.kill(pid, signal.SIGTERM)
            os.remove(demo_pid_file)
            return jsonify({'message': 'Demo stopped', 'pid': pid})
        except ProcessLookupError:
            os.remove(demo_pid_file)
            return jsonify({'message': 'Demo was not running (cleaned up stale PID file)'})
    except Exception as e:
        logger.error(f"Error stopping demo: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/demo/master-stop', methods=['POST'])
def master_stop_demo():
    """MASTER STOP: Forcefully kill all demo processes immediately"""
    try:
        import os
        import signal
        import subprocess

        demo_pid_file = '/tmp/lnsp_demo.pid'
        killed_pids = []

        # Step 1: Kill the main demo process if running
        if os.path.exists(demo_pid_file):
            with open(demo_pid_file, 'r') as f:
                main_pid = int(f.read().strip())

            try:
                # Kill entire process group (parent + all children)
                os.killpg(os.getpgid(main_pid), signal.SIGKILL)
                killed_pids.append(main_pid)
                logger.info(f"MASTER STOP: Killed process group for PID {main_pid}")
            except ProcessLookupError:
                logger.warning(f"MASTER STOP: Process {main_pid} not found")
            except Exception as e:
                logger.error(f"MASTER STOP: Error killing process group {main_pid}: {e}")
                # Fallback: try to kill just the main process
                try:
                    os.kill(main_pid, signal.SIGKILL)
                    killed_pids.append(main_pid)
                except ProcessLookupError:
                    pass

            # Clean up PID file
            os.remove(demo_pid_file)

        # Step 2: Find and kill any stray Python processes related to demo
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'lnsp_llm_driven_demo.py'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.stdout.strip():
                stray_pids = [int(pid) for pid in result.stdout.strip().split('\n')]
                for pid in stray_pids:
                    try:
                        os.kill(pid, signal.SIGKILL)
                        killed_pids.append(pid)
                        logger.info(f"MASTER STOP: Killed stray demo process {pid}")
                    except ProcessLookupError:
                        pass
        except subprocess.TimeoutExpired:
            logger.warning("MASTER STOP: pgrep timeout")
        except FileNotFoundError:
            # pgrep not available on this system, skip
            pass
        except Exception as e:
            logger.error(f"MASTER STOP: Error finding stray processes: {e}")

        if killed_pids:
            return jsonify({
                'message': 'MASTER STOP: All demo processes forcefully terminated',
                'killed_pids': killed_pids
            })
        else:
            return jsonify({'message': 'MASTER STOP: No demo processes found'})

    except Exception as e:
        logger.error(f"MASTER STOP error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/demo/status', methods=['GET'])
def demo_status():
    """Check if demo is running"""
    try:
        import os

        demo_pid_file = '/tmp/lnsp_demo.pid'
        if not os.path.exists(demo_pid_file):
            return jsonify({'running': False})

        with open(demo_pid_file, 'r') as f:
            pid = int(f.read().strip())

        try:
            os.kill(pid, 0)  # Check if process exists
            return jsonify({'running': True, 'pid': pid})
        except OSError:
            os.remove(demo_pid_file)
            return jsonify({'running': False})
    except Exception as e:
        logger.error(f"Error checking demo status: {e}")
        return jsonify({'running': False, 'error': str(e)})


@app.route('/api/demo/clear', methods=['POST'])
def clear_demo_data():
    """Clear all demo data (files only, not database)"""
    try:
        import shutil
        import os
        import signal

        # Stop demo if running
        demo_pid_file = '/tmp/lnsp_demo.pid'
        if os.path.exists(demo_pid_file):
            with open(demo_pid_file, 'r') as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, signal.SIGTERM)
            except:
                pass
            os.remove(demo_pid_file)

        # Delete demo directory
        demo_dir = '/tmp/lnsp_demo'
        if os.path.exists(demo_dir):
            shutil.rmtree(demo_dir)

        return jsonify({'message': 'Demo files cleared'})
    except Exception as e:
        logger.error(f"Error clearing demo data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/clear-all', methods=['POST'])
def clear_all_data():
    """Clear all project/task data from Registry (action_logs + services tables)"""
    try:
        import sqlite3
        import os
        import signal

        db_path = os.path.join(
            os.path.dirname(__file__),
            '../../artifacts/registry/registry.db'
        )

        if not os.path.exists(db_path):
            return jsonify({'error': 'Registry database not found'}), 404

        # Stop demo if running
        demo_pid_file = '/tmp/lnsp_demo.pid'
        if os.path.exists(demo_pid_file):
            with open(demo_pid_file, 'r') as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, signal.SIGTERM)
            except:
                pass
            os.remove(demo_pid_file)

        # Clear action_logs and services tables
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get counts before deletion
        cursor.execute("SELECT COUNT(*) FROM action_logs")
        action_logs_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM services")
        services_count = cursor.fetchone()[0]

        # Delete all data
        cursor.execute("DELETE FROM action_logs")
        cursor.execute("DELETE FROM services")

        conn.commit()
        conn.close()

        # Reset in-memory state
        global last_known_log_id
        last_known_log_id = 0

        logger.info(f"Cleared {action_logs_count} action logs and {services_count} services")

        return jsonify({
            'message': 'All data cleared',
            'action_logs_deleted': action_logs_count,
            'services_deleted': services_count
        })
    except Exception as e:
        logger.error(f"Error clearing all data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/pricing/stats', methods=['GET'])
def get_pricing_stats():
    """Get pricing cache statistics"""
    try:
        from llm_pricing import get_pricing_service

        pricing_service = get_pricing_service()
        stats = pricing_service.get_cache_stats()

        return jsonify({
            'status': 'ok',
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Error getting pricing stats: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/admin/pricing/refresh', methods=['POST'])
def refresh_pricing_cache():
    """Refresh all pricing cache entries"""
    try:
        from llm_pricing import get_pricing_service

        pricing_service = get_pricing_service()
        stats = pricing_service.refresh_all_cache()

        return jsonify({
            'status': 'ok',
            'message': f"Refreshed {stats['refreshed']} of {stats['total']} entries",
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Error refreshing pricing cache: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/admin/pricing/clear', methods=['POST'])
def clear_pricing_cache():
    """Clear pricing cache (will rebuild on next request)"""
    try:
        import os
        cache_path = "artifacts/hmi/pricing_cache.db"

        if os.path.exists(cache_path):
            os.remove(cache_path)

        return jsonify({
            'status': 'ok',
            'message': 'Pricing cache cleared successfully'
        })
    except Exception as e:
        logger.error(f"Error clearing pricing cache: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/admin/restart-services', methods=['POST'])
def restart_services():
    """Restart ALL system services (P0 Stack + PAS + HMI)"""
    try:
        import subprocess
        import os

        # Use comprehensive restart script for full system restart
        script_path = os.path.join(
            os.path.dirname(__file__),
            '../../scripts/restart_full_system.sh'
        )

        # Fallback to old script if new one doesn't exist
        if not os.path.exists(script_path):
            script_path = os.path.join(
                os.path.dirname(__file__),
                '../../scripts/restart_all_services.sh'
            )

        if not os.path.exists(script_path):
            return jsonify({'error': 'Restart script not found'}), 404

        # Reset SERVER_START_TIME
        global SERVER_START_TIME
        SERVER_START_TIME = time.time()

        # Execute restart script in fully detached background
        # Use nohup to ensure it continues after HMI dies
        subprocess.Popen(['nohup', 'bash', script_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True)  # Detach from parent process

        logger.info("Full system restart initiated (P0 + PAS + HMI), uptime counter reset")

        return jsonify({
            'message': 'Full system restart initiated',
            'note': 'All services (Model Pool, P0 Stack, PAS, HMI) will restart in ~15 seconds',
            'uptime_reset': True,
            'services': [
                'Model Pool Manager (8050)',
                'Model Services (8051-8099)',
                'Gateway (6120)',
                'PAS Root (6100)',
                'Aider-LCO (6130)',
                'Registry (6121)',
                'Heartbeat Monitor (6109)',
                'Resource Manager (6104)',
                'Token Governor (6105)',
                'HMI Dashboard (6101)'
            ]
        })
    except Exception as e:
        logger.error(f"Error restarting services: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Real-time Updates via Server-Sent Events (SSE)
# ============================================================================

# Global state for tracking action_log updates
action_log_subscribers = []  # List of subscriber generators
action_log_lock = threading.Lock()
last_known_log_id = 0  # Track last processed log ID


def get_db_path():
    """Get the path to the Registry database"""
    return os.path.join(
        os.path.dirname(__file__),
        '../../artifacts/registry/registry.db'
    )


def poll_action_logs():
    """
    Background thread that polls action_logs table for new entries.
    When new entries are found, notifies all SSE subscribers.
    """
    global last_known_log_id

    db_path = get_db_path()
    if not os.path.exists(db_path):
        logger.warning(f"Registry database not found at {db_path}")
        return

    while True:
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get new action_logs since last check
            cursor.execute("""
                SELECT * FROM action_logs
                WHERE log_id > ?
                ORDER BY log_id ASC
                LIMIT 100
            """, (last_known_log_id,))

            new_logs = cursor.fetchall()
            conn.close()

            if new_logs:
                # Update last known log ID
                last_known_log_id = new_logs[-1]['log_id']

                # Convert rows to dicts
                new_actions = []
                for row in new_logs:
                    action_dict = dict(row)
                    # Parse JSON fields
                    if action_dict.get('action_data'):
                        try:
                            action_dict['action_data'] = json.loads(action_dict['action_data'])
                        except:
                            pass
                    new_actions.append(action_dict)

                # Notify all subscribers
                with action_log_lock:
                    for subscriber_queue in action_log_subscribers[:]:  # Copy to avoid modification during iteration
                        try:
                            subscriber_queue.put({
                                'type': 'new_actions',
                                'data': new_actions
                            })
                        except:
                            # Remove dead subscribers
                            action_log_subscribers.remove(subscriber_queue)

            # Poll every 1 second
            time.sleep(1)

        except Exception as e:
            logger.error(f"Error polling action_logs: {e}")
            time.sleep(5)  # Back off on error


@app.route('/api/stream/action_logs', methods=['GET'])
def stream_action_logs():
    """
    Server-Sent Events endpoint for real-time action_log updates.

    Query params:
        - task_id: Filter updates to a specific task (optional)

    Returns SSE stream:
        event: new_actions
        data: [{"log_id": ..., "task_id": ..., ...}, ...]
    """
    task_id_filter = request.args.get('task_id')

    def generate():
        """SSE generator function"""
        import queue

        # Create a queue for this client
        subscriber_queue = queue.Queue(maxsize=100)

        # Register subscriber
        with action_log_lock:
            action_log_subscribers.append(subscriber_queue)

        try:
            # Send initial connection event
            yield f"event: connected\ndata: {json.dumps({'status': 'ok', 'task_id': task_id_filter})}\n\n"

            # Keep connection alive and send updates
            while True:
                try:
                    # Wait for new data (with timeout to send keep-alive)
                    try:
                        message = subscriber_queue.get(timeout=15)
                    except queue.Empty:
                        # Send keep-alive ping
                        yield f"event: ping\ndata: {json.dumps({'timestamp': datetime.now().isoformat()})}\n\n"
                        continue

                    # Filter by task_id if specified
                    if message['type'] == 'new_actions':
                        actions = message['data']

                        if task_id_filter:
                            actions = [a for a in actions if a.get('task_id') == task_id_filter]

                        if actions:
                            yield f"event: new_actions\ndata: {json.dumps(actions)}\n\n"

                except GeneratorExit:
                    break

        finally:
            # Unregister subscriber
            with action_log_lock:
                if subscriber_queue in action_log_subscribers:
                    action_log_subscribers.remove(subscriber_queue)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/stream/tree/<task_id>', methods=['GET'])
def stream_tree_updates(task_id):
    """
    SSE endpoint for real-time tree updates for a specific task.
    Returns simplified tree node updates (not full tree rebuild).

    Event types:
        - new_node: New agent/action node added
        - update_node: Existing node status changed
        - new_edge: New delegation edge added
    """
    def generate():
        """SSE generator function"""
        import queue

        # Create a queue for this client
        subscriber_queue = queue.Queue(maxsize=100)

        # Register subscriber
        with action_log_lock:
            action_log_subscribers.append(subscriber_queue)

        try:
            # Send initial connection event
            yield f"event: connected\ndata: {json.dumps({'status': 'ok', 'task_id': task_id})}\n\n"

            # Track seen agents and edges to only send new ones
            seen_agents = set()
            seen_edges = set()

            while True:
                try:
                    # Wait for new data
                    try:
                        message = subscriber_queue.get(timeout=15)
                    except queue.Empty:
                        # Send keep-alive
                        yield f"event: ping\ndata: {json.dumps({'timestamp': datetime.now().isoformat()})}\n\n"
                        continue

                    if message['type'] == 'new_actions':
                        actions = [a for a in message['data'] if a.get('task_id') == task_id]

                        for action in actions:
                            from_agent = action.get('from_agent')
                            to_agent = action.get('to_agent')

                            # New agent nodes
                            if from_agent and from_agent not in seen_agents and from_agent != 'user':
                                seen_agents.add(from_agent)
                                yield f"event: new_node\ndata: {json.dumps({
                                    'agent_id': from_agent,
                                    'name': from_agent.replace('_', ' ').title(),
                                    'tier': action.get('tier_from', '?'),
                                    'status': action.get('status', 'unknown')
                                })}\n\n"

                            if to_agent and to_agent not in seen_agents and to_agent != 'user':
                                seen_agents.add(to_agent)
                                yield f"event: new_node\ndata: {json.dumps({
                                    'agent_id': to_agent,
                                    'name': to_agent.replace('_', ' ').title(),
                                    'tier': action.get('tier_to', '?'),
                                    'status': action.get('status', 'unknown')
                                })}\n\n"

                            # New delegation edge
                            if from_agent and to_agent:
                                edge_key = f"{from_agent}->{to_agent}"
                                if edge_key not in seen_edges:
                                    seen_edges.add(edge_key)
                                    yield f"event: new_edge\ndata: {json.dumps({
                                        'from': from_agent,
                                        'to': to_agent,
                                        'action_type': action.get('action_type', 'unknown')
                                    })}\n\n"

                            # Node status update
                            if to_agent and action.get('status'):
                                yield f"event: update_node\ndata: {json.dumps({
                                    'agent_id': to_agent,
                                    'status': action.get('status'),
                                    'action_name': action.get('action_name', '')
                                })}\n\n"

                except GeneratorExit:
                    break

        finally:
            # Unregister subscriber
            with action_log_lock:
                if subscriber_queue in action_log_subscribers:
                    action_log_subscribers.remove(subscriber_queue)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


# Agent Chat SSE subscriber tracking
agent_chat_lock = threading.Lock()
agent_chat_subscribers = []  # List of queues for SSE clients


def poll_agent_chat_events():
    """
    Background thread that polls Event Stream WebSocket for agent chat events.
    Forwards events to all SSE subscribers.
    """
    import socketio as socketio_client

    logger.info("Starting agent chat event polling thread...")

    # Connect to Event Stream WebSocket (port 6102)
    sio = socketio_client.Client()

    try:
        sio.connect(EVENT_STREAM_URL)
        logger.info("Connected to Event Stream WebSocket for agent chat events")

        @sio.on('event')
        def on_event(data):
            """Handle incoming events from Event Stream"""
            event_type = data.get('event_type', '')

            # Only forward agent_chat_* events
            if event_type.startswith('agent_chat_'):
                event_data = data.get('data', {})

                # Notify all subscribers
                with agent_chat_lock:
                    for subscriber_queue in agent_chat_subscribers[:]:
                        try:
                            subscriber_queue.put({
                                'type': event_type,
                                'data': event_data
                            })
                        except:
                            # Remove dead subscribers
                            agent_chat_subscribers.remove(subscriber_queue)

        # Keep connection alive
        sio.wait()

    except Exception as e:
        logger.error(f"Error connecting to Event Stream for agent chat: {e}")
        logger.info("Agent chat SSE will rely on polling fallback")


# Start agent chat event polling thread
agent_chat_polling_thread = threading.Thread(target=poll_agent_chat_events, daemon=True)
agent_chat_polling_thread.start()


@app.route('/api/stream/agent_chat/<run_id>', methods=['GET'])
def stream_agent_chat(run_id):
    """
    SSE endpoint for real-time agent chat updates for a specific run.

    Event types:
        - agent_chat_message_sent: New message in conversation
        - agent_chat_thread_created: New thread started
        - agent_chat_thread_closed: Thread completed/failed

    Example:
        event: agent_chat_message_sent
        data: {"thread_id": "...", "message_id": "...", "message_type": "question", ...}
    """
    def generate():
        """SSE generator function"""
        import queue

        # Create a queue for this client
        subscriber_queue = queue.Queue(maxsize=100)

        # Register subscriber
        with agent_chat_lock:
            agent_chat_subscribers.append(subscriber_queue)

        try:
            # Send initial connection event
            yield f"event: connected\ndata: {json.dumps({'status': 'ok', 'run_id': run_id})}\n\n"

            # Send existing messages for this run (initial state)
            try:
                chat_client = AgentChatClient()

                # Query all threads for this run
                conn = sqlite3.connect(chat_client.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT thread_id FROM agent_conversation_threads
                    WHERE run_id = ?
                    ORDER BY created_at ASC
                """, (run_id,))

                thread_ids = [row[0] for row in cursor.fetchall()]
                conn.close()

                # Send all existing messages
                for thread_id in thread_ids:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    thread = loop.run_until_complete(chat_client.get_thread(thread_id))
                    loop.close()

                    for msg in thread.messages:
                        # Messages are AgentChatMessage Pydantic objects
                        yield f"event: agent_chat_message_sent\ndata: {json.dumps({
                            'run_id': run_id,
                            'thread_id': thread_id,
                            'message_id': msg.message_id,
                            'from_agent': msg.from_agent,
                            'to_agent': msg.to_agent,
                            'message_type': msg.message_type,
                            'content': msg.content,
                            'created_at': msg.created_at,
                            'metadata': msg.metadata
                        })}\n\n"

            except Exception as e:
                logger.error(f"Error loading initial agent chat messages: {e}")

            # Stream new messages
            while True:
                try:
                    # Wait for new data
                    try:
                        message = subscriber_queue.get(timeout=15)
                    except queue.Empty:
                        # Send keep-alive
                        yield f"event: ping\ndata: {json.dumps({'timestamp': datetime.now().isoformat()})}\n\n"
                        continue

                    event_type = message['type']
                    event_data = message['data']

                    # Filter by run_id
                    if event_data.get('run_id') == run_id:
                        yield f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"

                except GeneratorExit:
                    break

        finally:
            # Unregister subscriber
            with agent_chat_lock:
                if subscriber_queue in agent_chat_subscribers:
                    agent_chat_subscribers.remove(subscriber_queue)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


# Model configuration paths
MODEL_PREFERENCES_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'pas', 'model_preferences.json')
ENV_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
LOCAL_LLMS_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'pas', 'local_llms.yaml')

# Cache for local LLM health status (to avoid checking too frequently)
local_llm_health_cache = {}
local_llm_health_cache_time = {}


def check_local_llm_health(host: str, port: int, endpoint: str = "/health", timeout: int = 2) -> str:
    """Check health of a local LLM endpoint

    Returns: "OK", "ERR", or "OFFLINE"
    """
    # Check cache first
    cache_key = f"{host}:{port}{endpoint}"
    cache_duration = 30  # seconds

    if cache_key in local_llm_health_cache_time:
        age = time.time() - local_llm_health_cache_time[cache_key]
        if age < cache_duration:
            return local_llm_health_cache[cache_key]

    try:
        url = f"http://{host}:{port}{endpoint}"
        response = requests.get(url, timeout=timeout)

        if response.status_code == 200:
            status = "OK"
        else:
            status = "ERR"
    except requests.exceptions.Timeout:
        status = "OFFLINE"
    except requests.exceptions.ConnectionError:
        status = "OFFLINE"
    except Exception as e:
        logger.debug(f"Health check failed for {cache_key}: {e}")
        status = "ERR"

    # Update cache
    local_llm_health_cache[cache_key] = status
    local_llm_health_cache_time[cache_key] = time.time()

    return status


def load_model_preferences():
    """Load model preferences from config file"""
    try:
        if os.path.exists(MODEL_PREFERENCES_PATH):
            with open(MODEL_PREFERENCES_PATH, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load model preferences: {e}")

    # Return defaults if file doesn't exist
    return {
        "architect": {
            "primary": "auto",
            "fallback": "anthropic/claude-3-5-sonnet-20241022"
        },
        "director": {
            "primary": "auto",
            "fallback": "anthropic/claude-3-5-sonnet-20241022"
        },
        "manager": {
            "primary": "auto",
            "fallback": "anthropic/claude-3-5-haiku-20241022"
        },
        "programmer": {
            "primary": "ollama/qwen2.5-coder:7b-instruct",
            "fallback": "anthropic/claude-3-5-sonnet-20241022"
        }
    }


def save_model_preferences(preferences: dict):
    """Save model preferences to config file"""
    try:
        os.makedirs(os.path.dirname(MODEL_PREFERENCES_PATH), exist_ok=True)
        with open(MODEL_PREFERENCES_PATH, 'w') as f:
            json.dump(preferences, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Could not save model preferences: {e}")
        return False


def get_available_models():
    """Parse .env file and local_llms.yaml to get available models with health status"""
    models = {
        "auto": {
            "name": "Auto Select",
            "provider": "auto",
            "description": "Dynamically select best model based on task requirements",
            "available": True,
            "status": "OK"
        }
    }

    # Load local LLMs from configuration file
    if os.path.exists(LOCAL_LLMS_CONFIG_PATH):
        try:
            with open(LOCAL_LLMS_CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f)

            local_llms = config.get('local_llms', [])
            for llm in local_llms:
                model_id = llm.get('model_id')
                host = llm.get('host', 'localhost')
                port = llm.get('port')
                endpoint = llm.get('endpoint', '/health')

                # Check health status
                status = check_local_llm_health(host, port, endpoint)

                models[model_id] = {
                    "name": llm.get('name'),
                    "provider": llm.get('provider', 'local'),
                    "description": llm.get('description', ''),
                    "available": status == "OK",
                    "status": status,
                    "host": host,
                    "port": port
                }
        except Exception as e:
            logger.warning(f"Could not load local LLMs config: {e}")

    # Parse .env file for API keys
    if os.path.exists(ENV_FILE_PATH):
        try:
            with open(ENV_FILE_PATH, 'r') as f:
                env_content = f.read()

                # Check for OpenAI
                if 'OPENAI_API_KEY=' in env_content and 'your_' not in env_content.split('OPENAI_API_KEY=')[1].split('\n')[0]:
                    # Check if API key is valid (not placeholder)
                    api_key = env_content.split('OPENAI_API_KEY=')[1].split('\n')[0].strip().strip("'\"")
                    is_valid = 'your_' not in api_key and len(api_key) > 10

                    # Extract configured model name if specified
                    if 'OPENAI_MODEL_NAME=' in env_content:
                        model_line = env_content.split('OPENAI_MODEL_NAME=')[1].split('\n')[0].strip().strip("'\"")
                        models[f"openai/{model_line}"] = {
                            "name": f"OpenAI {model_line}",
                            "provider": "openai",
                            "description": "OpenAI model",
                            "available": is_valid,
                            "status": "API" if is_valid else "INVALID_KEY"
                        }

                    # Add GPT-5.1 models (available with OpenAI API key)
                    models["openai/gpt-5.1-chat-latest"] = {
                        "name": "GPT-5.1 Chat",
                        "provider": "openai",
                        "description": "GPT-5.1 with adaptive reasoning (latest)",
                        "available": is_valid,
                        "status": "API" if is_valid else "INVALID_KEY"
                    }
                    models["openai/gpt-5.1-codex"] = {
                        "name": "GPT-5.1-Codex",
                        "provider": "openai",
                        "description": "GPT-5.1 optimized for software engineering tasks",
                        "available": is_valid,
                        "status": "API" if is_valid else "INVALID_KEY"
                    }
                    models["openai/gpt-5.1-codex-mini"] = {
                        "name": "GPT-5.1-Codex-Mini",
                        "provider": "openai",
                        "description": "Lightweight GPT-5.1 for coding workflows",
                        "available": is_valid,
                        "status": "API" if is_valid else "INVALID_KEY"
                    }

                # Check for Anthropic
                if 'ANTHROPIC_API_KEY=' in env_content and 'your_' not in env_content.split('ANTHROPIC_API_KEY=')[1].split('\n')[0]:
                    # Extract model names
                    for tier in ['HIGH', 'MEDIUM', 'LOW']:
                        key = f'ANTHROPIC_MODEL_NAME_{tier}='
                        if key in env_content:
                            model_line = env_content.split(key)[1].split('\n')[0].strip().strip("'\"")
                            api_key = env_content.split('ANTHROPIC_API_KEY=')[1].split('\n')[0].strip().strip("'\"")
                            is_valid = 'your_' not in api_key and len(api_key) > 10
                            models[f"anthropic/{model_line}"] = {
                                "name": f"Anthropic {model_line}",
                                "provider": "anthropic",
                                "description": f"Anthropic Claude ({tier.lower()} tier)",
                                "available": is_valid,
                                "status": "API" if is_valid else "INVALID_KEY"
                            }

                # Check for Gemini
                if 'GEMINI_API_KEY=' in env_content and 'your_' not in env_content.split('GEMINI_API_KEY=')[1].split('\n')[0]:
                    for tier in ['HIGH', 'MEDIUM', 'LOW']:
                        key = f'GEMINI_MODEL_NAME_{tier}='
                        if key in env_content:
                            model_line = env_content.split(key)[1].split('\n')[0].strip().strip("'\"")
                            api_key = env_content.split('GEMINI_API_KEY=')[1].split('\n')[0].strip().strip("'\"")
                            is_valid = 'your_' not in api_key and len(api_key) > 10
                            models[f"google/{model_line}"] = {
                                "name": f"Google {model_line}",
                                "provider": "google",
                                "description": f"Google Gemini ({tier.lower()} tier)",
                                "available": is_valid,
                                "status": "API" if is_valid else "INVALID_KEY"
                            }

                # Check for DeepSeek API
                if 'DEEPSEEK_API_KEY=' in env_content and 'your_' not in env_content.split('DEEPSEEK_API_KEY=')[1].split('\n')[0]:
                    api_key = env_content.split('DEEPSEEK_API_KEY=')[1].split('\n')[0].strip().strip("'\"")
                    is_valid = 'your_' not in api_key and len(api_key) > 10
                    models["deepseek/deepseek-r1"] = {
                        "name": "DeepSeek R1 (API)",
                        "provider": "deepseek",
                        "description": "DeepSeek reasoning model via API",
                        "available": is_valid,
                        "status": "API" if is_valid else "INVALID_KEY"
                    }

                # Check for Kimi K2 API
                if 'KIMI_API_KEY=' in env_content and 'your_' not in env_content.split('KIMI_API_KEY=')[1].split('\n')[0]:
                    api_key = env_content.split('KIMI_API_KEY=')[1].split('\n')[0].strip().strip("'\"")
                    is_valid = 'your_' not in api_key and len(api_key) > 10

                    # Get model name if specified
                    model_name = "kimi-k2"
                    if 'KIMI_MODEL_NAME=' in env_content:
                        raw_value = env_content.split('KIMI_MODEL_NAME=')[1].split('\n')[0]
                        # Remove inline comments
                        raw_value = raw_value.split('#')[0].strip()
                        # Remove quotes
                        model_name = raw_value.strip("'\"")

                    models[f"kimi/{model_name}"] = {
                        "name": f"Kimi {model_name.upper()}",
                        "provider": "kimi",
                        "description": "Moonshot AI Kimi large language model",
                        "available": is_valid,
                        "status": "API" if is_valid else "INVALID_KEY"
                    }
        except Exception as e:
            logger.warning(f"Could not parse .env file: {e}")

    return models


@app.route('/api/models/available', methods=['GET'])
def get_available_models_api():
    """Get list of available models"""
    return jsonify({'status': 'ok', 'models': get_available_models()})


@app.route('/api/models/status', methods=['GET'])
def get_model_status():
    """Get detailed model status for settings page"""
    try:
        models = get_available_models()
        
        # Add detailed status info with usage tracking
        status_info = {}
        for model_id, model in models.items():
            status_info[model_id] = {
                'name': model['name'],
                'provider': model['provider'],
                'status': model['status'],
                'available': model['available'],
                'description': model['description'],
                'last_check': time.time(),
                'type': 'local' if model['provider'] == 'local' else 'api',
                'usage': {
                    'total_requests': 0,
                    'total_tokens': 0,
                    'total_cost': 0.0,
                    'last_used': None,
                    'average_tokens_per_request': 0,
                    'average_cost_per_request': 0.0
                }
            }
            
            # Add health details for local models
            if model['provider'] == 'local':
                status_info[model_id]['port'] = model.get('port', 'N/A')
                status_info[model_id]['host'] = model.get('host', 'localhost')
                status_info[model_id]['endpoint'] = f"http://{model.get('host', 'localhost')}:{model.get('port', 'N/A')}"
            
            # Add API details for external models
            if model['provider'] in ['openai', 'anthropic', 'google', 'deepseek', 'kimi']:
                status_info[model_id]['api_status'] = 'Configured' if model['available'] else 'Invalid Key'
                status_info[model_id]['api_provider'] = model['provider'].upper()
            
        return jsonify({'status': 'ok', 'models': status_info})
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/models/usage', methods=['GET'])
def get_model_usage():
    """Get model usage statistics from database"""
    try:
        db = get_db_session()

        # Initialize usage data structure
        usage_data = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'models': {}
        }

        # Query all messages with usage data
        messages = db.query(Message).filter(Message.usage_json.isnot(None)).all()

        for msg in messages:
            try:
                usage = msg.get_usage()
                if not usage:
                    continue

                model_key = msg.model_name or "unknown"

                # Initialize model entry if not exists
                if model_key not in usage_data['models']:
                    usage_data['models'][model_key] = {
                        'requests': 0,
                        'tokens': 0,
                        'cost': 0.0,
                        'input_tokens': 0,
                        'output_tokens': 0
                    }

                # Aggregate stats
                total_tokens = usage.get('total_tokens', 0)
                input_tokens = usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0) or usage.get('output_tokens', 0)
                cost_usd = usage.get('cost_usd', 0.0) or 0.0

                usage_data['models'][model_key]['requests'] += 1
                usage_data['models'][model_key]['tokens'] += total_tokens
                usage_data['models'][model_key]['input_tokens'] += input_tokens
                usage_data['models'][model_key]['output_tokens'] += output_tokens
                usage_data['models'][model_key]['cost'] += cost_usd

                # Update totals
                usage_data['total_requests'] += 1
                usage_data['total_tokens'] += total_tokens
                usage_data['total_cost'] += cost_usd

            except Exception as e:
                logger.warning(f"Error processing message usage: {e}")
                continue

        db.close()
        return jsonify({'status': 'ok', 'usage': usage_data})
    except Exception as e:
        logger.error(f"Error getting model usage: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/models/usage/clear', methods=['POST'])
def clear_model_usage():
    """Clear model usage statistics by clearing usage data from messages"""
    try:
        db = get_db_session()
        
        # Count messages with usage data before clearing
        messages_with_usage = db.query(Message).filter(Message.usage_json.isnot(None)).count()
        
        # Clear usage data from all messages
        db.query(Message).filter(Message.usage_json.isnot(None)).update(
            {Message.usage_json: None},
            synchronize_session=False
        )
        
        db.commit()
        db.close()
        
        logger.info(f"Cleared usage data from {messages_with_usage} messages")
        return jsonify({
            'status': 'ok', 
            'message': f'Usage statistics cleared from {messages_with_usage} messages'
        })
    except Exception as e:
        logger.error(f"Error clearing model usage: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
        logger.error(f"Error getting available models: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/models/preferences', methods=['GET'])
def get_model_preferences_api():
    """Get current model preferences"""
    try:
        preferences = load_model_preferences()
        return jsonify({
            'status': 'ok',
            'preferences': preferences
        })
    except Exception as e:
        logger.error(f"Error getting model preferences: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/models/preferences', methods=['POST'])
def save_model_preferences_api():
    """Save model preferences"""
    try:
        data = request.get_json()
        preferences = data.get('preferences', {})

        # Validate structure
        required_keys = ['architect', 'director', 'manager', 'programmer']
        for key in required_keys:
            if key not in preferences:
                return jsonify({
                    'status': 'error',
                    'error': f'Missing required key: {key}'
                }), 400
            if 'primary' not in preferences[key] or 'fallback' not in preferences[key]:
                return jsonify({
                    'status': 'error',
                    'error': f'Missing primary or fallback for {key}'
                }), 400

        # Save preferences
        success = save_model_preferences(preferences)
        if success:
            logger.info(f"Updated model preferences: {preferences}")
            return jsonify({'status': 'ok'})
        else:
            return jsonify({'status': 'error', 'error': 'Failed to save preferences'}), 500
    except Exception as e:
        logger.error(f"Error saving model preferences: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/models/advanced-settings', methods=['POST'])
def save_advanced_model_settings_api():
    """Save advanced model settings (temperature, max_tokens, etc.)"""
    try:
        data = request.get_json()

        # Validate settings
        settings = {
            'temperature': data.get('temperature', 0.7),
            'maxTokens': data.get('maxTokens', 2000),
            'topP': data.get('topP', 0.9),
            'topK': data.get('topK', 40),
            'frequencyPenalty': data.get('frequencyPenalty', 0.0),
            'presencePenalty': data.get('presencePenalty', 0.0)
        }

        # Save to config file
        advanced_settings_path = Path('configs/pas/advanced_model_settings.json')
        advanced_settings_path.parent.mkdir(parents=True, exist_ok=True)

        with open(advanced_settings_path, 'w') as f:
            json.dump(settings, f, indent=2)

        logger.info(f"Saved advanced model settings: {settings}")
        return jsonify({'status': 'success'})

    except Exception as e:
        logger.error(f"Error saving advanced model settings: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/models/advanced-settings', methods=['GET'])
def get_advanced_model_settings_api():
    """Get current advanced model settings"""
    try:
        advanced_settings_path = Path('configs/pas/advanced_model_settings.json')

        if advanced_settings_path.exists():
            with open(advanced_settings_path, 'r') as f:
                settings = json.load(f)
        else:
            # Default settings
            settings = {
                'temperature': 0.7,
                'maxTokens': 2000,
                'topP': 0.9,
                'topK': 40,
                'frequencyPenalty': 0.0,
                'presencePenalty': 0.0
            }

        return jsonify({'status': 'ok', 'settings': settings})

    except Exception as e:
        logger.error(f"Error getting advanced model settings: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ============================================================================
# PROGRAMMER POOL API
# ============================================================================

@app.route('/api/programmer-pool/status', methods=['GET'])
def get_programmer_pool_status():
    """Get Programmer Pool status and statistics"""
    try:
        import sys
        from pathlib import Path

        # Add project root to path
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from services.common.programmer_pool import get_programmer_pool

        pool = get_programmer_pool()

        # Run discovery if pool is empty or stale
        if len(pool.programmers) == 0:
            pool.discover_programmers()

        # Get stats and programmer list
        stats = pool.get_stats()
        programmers = pool.list_programmers()

        return jsonify({
            'status': 'ok',
            'stats': stats,
            'programmers': programmers
        })
    except Exception as e:
        logger.error(f"Error getting programmer pool status: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/programmer-pool/programmers', methods=['GET'])
def get_programmers_list():
    """Get detailed list of all Programmers"""
    try:
        import sys
        from pathlib import Path

        # Add project root to path
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from services.common.programmer_pool import get_programmer_pool

        pool = get_programmer_pool()
        programmers = pool.list_programmers()

        # Add detailed info for each programmer
        detailed_programmers = []
        for prog in programmers:
            # Try to get additional info from health endpoint
            try:
                import httpx
                response = httpx.get(f"{prog['endpoint']}/health", timeout=2.0)
                if response.status_code == 200:
                    health_data = response.json()
                    prog['health'] = health_data
                else:
                    prog['health'] = None
            except Exception:
                prog['health'] = None

            detailed_programmers.append(prog)

        return jsonify({
            'status': 'ok',
            'programmers': detailed_programmers
        })
    except Exception as e:
        logger.error(f"Error getting programmers list: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ============================================================================
# HHMRS SETTINGS API
# ============================================================================

@app.route('/api/settings/hhmrs', methods=['GET'])
def get_hhmrs_settings():
    """Get HHMRS (Health Heartbeat Monitoring) settings from pas_settings.json"""
    try:
        settings_path = Path('artifacts/pas_settings.json')

        if settings_path.exists():
            with open(settings_path, 'r') as f:
                all_settings = json.load(f)
        else:
            # Return default settings if file doesn't exist
            all_settings = {
                'hhmrs': {
                    'heartbeat_interval_s': 30,
                    'timeout_threshold_s': 60,
                    'max_restarts': 3,
                    'max_llm_retries': 3,
                    'enable_auto_restart': True,
                    'enable_llm_switching': True
                }
            }

        return jsonify(all_settings)

    except Exception as e:
        logger.error(f"Error getting HHMRS settings: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/settings/hhmrs', methods=['POST'])
def save_hhmrs_settings():
    """Save HHMRS settings to pas_settings.json"""
    try:
        data = request.get_json()
        hhmrs_settings = data.get('hhmrs', {})

        # Validate settings
        required_keys = ['heartbeat_interval_s', 'timeout_threshold_s', 'max_restarts', 'max_llm_retries']
        for key in required_keys:
            if key not in hhmrs_settings:
                return jsonify({
                    'status': 'error',
                    'error': f'Missing required key: {key}'
                }), 400

        # Validate numeric ranges
        if not (5 <= hhmrs_settings['heartbeat_interval_s'] <= 120):
            return jsonify({'status': 'error', 'error': 'heartbeat_interval_s must be between 5 and 120'}), 400

        if not (10 <= hhmrs_settings['timeout_threshold_s'] <= 300):
            return jsonify({'status': 'error', 'error': 'timeout_threshold_s must be between 10 and 300'}), 400

        if not (1 <= hhmrs_settings['max_restarts'] <= 10):
            return jsonify({'status': 'error', 'error': 'max_restarts must be between 1 and 10'}), 400

        if not (1 <= hhmrs_settings['max_llm_retries'] <= 10):
            return jsonify({'status': 'error', 'error': 'max_llm_retries must be between 1 and 10'}), 400

        # Load existing settings
        settings_path = Path('artifacts/pas_settings.json')
        settings_path.parent.mkdir(parents=True, exist_ok=True)

        if settings_path.exists():
            with open(settings_path, 'r') as f:
                all_settings = json.load(f)
        else:
            all_settings = {}

        # Update HHMRS section
        all_settings['hhmrs'] = hhmrs_settings

        # Save back to file
        with open(settings_path, 'w') as f:
            json.dump(all_settings, f, indent=2)

        logger.info(f"✓ Saved HHMRS settings: {hhmrs_settings}")
        return jsonify({'status': 'ok', 'message': 'HHMRS settings saved successfully'})

    except Exception as e:
        logger.error(f"Error saving HHMRS settings: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ============================================================================
# SYSTEM STATUS API
# ============================================================================

@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """
    Comprehensive system health check:
    - Port status for all services
    - Git repository health
    - Disk space
    - Database connectivity
    - LLM availability
    - Python environment
    - Configuration validity
    """
    import socket
    import time
    import os
    import shutil
    import subprocess
    import json
    from pathlib import Path

    try:
        # Required ports (count against health if down)
        required_ports = [
            # Core Services
            6120, 6100, 6121, 6101, 6102, 6103,  # Gateway, PAS Root, Registry, HMI, Events, Router
            6104, 6105, 6109,  # Resource Manager, Token Governor, TRON (HeartbeatMonitor)
            # PAS Agent Tiers
            6110,  # Architect
            6111, 6112, 6113, 6114, 6115,  # Directors (Code, Models, Data, DevSecOps, Docs)
            6141, 6142, 6143, 6144, 6145, 6146, 6147,  # Managers (Code 1-3, Models, Data, DevSecOps, Docs)
            6151, 6152, 6153, 6154, 6155, 6156, 6157, 6158, 6159, 6160,  # Programmers 1-10
            # LLM Services
            11434  # Ollama
        ]

        # Optional ports (don't count against health if down)
        optional_ports = {
            6130: 'Aider-LCO (on-demand)',
            8050: 'Model Pool (hibernated)',
            8051: 'Model: Qwen (hibernated)',
            8052: 'Model: Llama (hibernated)',
            8053: 'Model: DeepSeek (hibernated)'
        }

        all_ports = required_ports + list(optional_ports.keys())

        ports_status = {}
        ports_up = 0
        required_ports_up = 0

        # Check each port
        for port in all_ports:
            start_time = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex(('127.0.0.1', port))
            latency_ms = (time.time() - start_time) * 1000
            sock.close()

            is_optional = port in optional_ports

            if result == 0:
                status = 'up'
                if latency_ms > 200:
                    status = 'degraded'
                ports_status[port] = {
                    'status': status,
                    'latency_ms': round(latency_ms, 1),
                    'optional': is_optional
                }
                ports_up += 1
                if not is_optional:
                    required_ports_up += 1
            else:
                # Optional ports down = 'hibernated' (grey)
                # Required ports down = 'down' (red)
                ports_status[port] = {
                    'status': 'hibernated' if is_optional else 'down',
                    'latency_ms': None,
                    'error': 'Connection refused',
                    'optional': is_optional,
                    'note': optional_ports.get(port, '')
                }
                if not is_optional:
                    # Only count required ports against health
                    pass

        # Health checks
        health_checks = {}
        health_ok = 0
        total_checks = 6

        # 1. Git Status
        try:
            # Check for uncommitted changes
            result = subprocess.run(['git', 'status', '--porcelain'],
                                   capture_output=True, text=True, timeout=2)
            uncommitted = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0

            # Check branch
            branch_result = subprocess.run(['git', 'branch', '--show-current'],
                                          capture_output=True, text=True, timeout=2)
            branch = branch_result.stdout.strip()

            # Check ahead/behind
            status_result = subprocess.run(['git', 'status', '-sb'],
                                          capture_output=True, text=True, timeout=2)
            status_line = status_result.stdout.strip().split('\n')[0] if status_result.stdout else ''

            if uncommitted == 0:
                health_checks['git_status'] = {
                    'status': 'ok',
                    'message': f'Clean working directory on {branch}',
                    'details': {'branch': branch, 'uncommitted': 0}
                }
                health_ok += 1
            elif uncommitted < 10:
                health_checks['git_status'] = {
                    'status': 'warning',
                    'message': f'{uncommitted} uncommitted change(s) on {branch}',
                    'details': {'branch': branch, 'uncommitted': uncommitted}
                }
                health_ok += 0.5
            else:
                health_checks['git_status'] = {
                    'status': 'error',
                    'message': f'{uncommitted} uncommitted changes - consider committing',
                    'details': {'branch': branch, 'uncommitted': uncommitted}
                }
        except Exception as e:
            health_checks['git_status'] = {
                'status': 'error',
                'message': f'Git check failed: {str(e)}',
                'details': {}
            }

        # 2. Disk Space
        try:
            total, used, free = shutil.disk_usage('/')
            free_gb = free // (2**30)
            free_percent = (free / total) * 100

            if free_gb > 20:
                health_checks['disk_space'] = {
                    'status': 'ok',
                    'message': f'{free_gb}GB free ({round(free_percent, 1)}%)',
                    'details': {'free_gb': free_gb, 'free_percent': round(free_percent, 1)}
                }
                health_ok += 1
            elif free_gb > 10:
                health_checks['disk_space'] = {
                    'status': 'warning',
                    'message': f'{free_gb}GB free - low disk space',
                    'details': {'free_gb': free_gb, 'free_percent': round(free_percent, 1)}
                }
                health_ok += 0.5
            else:
                health_checks['disk_space'] = {
                    'status': 'error',
                    'message': f'{free_gb}GB free - critically low!',
                    'details': {'free_gb': free_gb, 'free_percent': round(free_percent, 1)}
                }
        except Exception as e:
            health_checks['disk_space'] = {
                'status': 'error',
                'message': f'Disk check failed: {str(e)}',
                'details': {}
            }

        # 3. Database Connectivity
        try:
            # Check PostgreSQL
            pg_result = subprocess.run(['psql', 'lnsp', '-c', 'SELECT 1'],
                                      capture_output=True, text=True, timeout=2)
            pg_ok = pg_result.returncode == 0

            # Check Neo4j (if available)
            neo4j_ok = False
            try:
                neo4j_result = subprocess.run(['cypher-shell', '-u', 'neo4j', '-p', 'password',
                                              'RETURN 1'],
                                             capture_output=True, text=True, timeout=2)
                neo4j_ok = neo4j_result.returncode == 0
            except:
                pass

            if pg_ok and neo4j_ok:
                health_checks['database_connectivity'] = {
                    'status': 'ok',
                    'message': 'PostgreSQL + Neo4j connected',
                    'details': {'postgresql': 'UP', 'neo4j': 'UP'}
                }
                health_ok += 1
            elif pg_ok:
                health_checks['database_connectivity'] = {
                    'status': 'warning',
                    'message': 'PostgreSQL connected, Neo4j unreachable',
                    'details': {'postgresql': 'UP', 'neo4j': 'DOWN'}
                }
                health_ok += 0.5
            else:
                health_checks['database_connectivity'] = {
                    'status': 'error',
                    'message': 'Database connection failed',
                    'details': {'postgresql': 'DOWN', 'neo4j': 'DOWN' if not neo4j_ok else 'UP'}
                }
        except Exception as e:
            health_checks['database_connectivity'] = {
                'status': 'error',
                'message': f'Database check failed: {str(e)}',
                'details': {}
            }

        # 4. LLM Availability
        try:
            import httpx
            response = httpx.get('http://localhost:11434/api/tags', timeout=2.0)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_count = len(models)
                model_names = [m['name'] for m in models[:3]]

                health_checks['llm_availability'] = {
                    'status': 'ok',
                    'message': f'Ollama running with {model_count} model(s)',
                    'details': {
                        'models_available': model_count,
                        'models': ', '.join(model_names)
                    }
                }
                health_ok += 1
            else:
                health_checks['llm_availability'] = {
                    'status': 'error',
                    'message': 'Ollama returned error',
                    'details': {}
                }
        except Exception as e:
            health_checks['llm_availability'] = {
                'status': 'error',
                'message': 'Ollama not reachable',
                'details': {'error': str(e)}
            }

        # 5. Python Environment
        try:
            import sys
            venv_active = hasattr(sys, 'real_prefix') or (
                hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
            )

            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

            if venv_active and sys.version_info >= (3, 11):
                health_checks['python_environment'] = {
                    'status': 'ok',
                    'message': f'Python {python_version} in virtual env',
                    'details': {'version': python_version, 'venv': 'active'}
                }
                health_ok += 1
            elif venv_active:
                health_checks['python_environment'] = {
                    'status': 'warning',
                    'message': f'Python {python_version} (update recommended)',
                    'details': {'version': python_version, 'venv': 'active'}
                }
                health_ok += 0.5
            else:
                health_checks['python_environment'] = {
                    'status': 'error',
                    'message': 'Not running in virtual environment',
                    'details': {'version': python_version, 'venv': 'inactive'}
                }
        except Exception as e:
            health_checks['python_environment'] = {
                'status': 'error',
                'message': f'Python check failed: {str(e)}',
                'details': {}
            }

        # 6. Configuration Validity
        try:
            config_files = [
                'configs/pas/model_preferences.json',
                'configs/pas/advanced_model_settings.json',
                'configs/pas/model_pool_config.json'
            ]

            valid_configs = 0
            invalid_configs = []

            for config_file in config_files:
                try:
                    with open(config_file, 'r') as f:
                        json.load(f)
                    valid_configs += 1
                except FileNotFoundError:
                    invalid_configs.append(f'{config_file} (missing)')
                except json.JSONDecodeError:
                    invalid_configs.append(f'{config_file} (invalid JSON)')

            if valid_configs == len(config_files):
                health_checks['config_validity'] = {
                    'status': 'ok',
                    'message': f'All {len(config_files)} configs valid',
                    'details': {'valid': valid_configs, 'invalid': 0}
                }
                health_ok += 1
            elif valid_configs > 0:
                health_checks['config_validity'] = {
                    'status': 'warning',
                    'message': f'{len(invalid_configs)} config(s) invalid',
                    'details': {
                        'valid': valid_configs,
                        'invalid': len(invalid_configs),
                        'errors': ', '.join(invalid_configs)
                    }
                }
                health_ok += 0.5
            else:
                health_checks['config_validity'] = {
                    'status': 'error',
                    'message': 'All configs invalid or missing',
                    'details': {
                        'valid': 0,
                        'invalid': len(invalid_configs),
                        'errors': ', '.join(invalid_configs)
                    }
                }
        except Exception as e:
            health_checks['config_validity'] = {
                'status': 'error',
                'message': f'Config check failed: {str(e)}',
                'details': {}
            }

        # Calculate overall health (only count required ports)
        port_health_percent = (required_ports_up / len(required_ports)) * 100
        check_health_percent = (health_ok / total_checks) * 100
        overall_health_percent = (port_health_percent * 0.6) + (check_health_percent * 0.4)

        issues_count = (len(required_ports) - required_ports_up) + (total_checks - int(health_ok))

        return jsonify({
            'status': 'ok',
            'overall_health_percent': round(overall_health_percent, 1),
            'issues_count': issues_count,
            'ports': ports_status,
            'ports_up': ports_up,
            'ports_total': len(all_ports),
            'required_ports_up': required_ports_up,
            'required_ports_total': len(required_ports),
            'health_checks': health_checks
        })

    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/system/restart', methods=['POST'])
def restart_system():
    """Restart all services"""
    try:
        # This would trigger a full restart
        # For now, just return success
        return jsonify({'status': 'ok', 'message': 'Services restarting...'})
    except Exception as e:
        logger.error(f"Error restarting system: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/system/clear-caches', methods=['POST'])
def clear_caches():
    """Clear temporary files and caches"""
    try:
        import shutil
        freed_mb = 0

        # Clear Python cache
        cache_dirs = ['.pytest_cache', '__pycache__']
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                dir_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(cache_dir)
                    for filename in filenames
                )
                freed_mb += dir_size / (1024 * 1024)
                shutil.rmtree(cache_dir, ignore_errors=True)

        return jsonify({'status': 'ok', 'freed_mb': round(freed_mb, 2)})
    except Exception as e:
        logger.error(f"Error clearing caches: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/system/git-gc', methods=['POST'])
def run_git_gc():
    """Run git garbage collection"""
    try:
        import subprocess
        result = subprocess.run(['git', 'gc', '--auto'],
                               capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            return jsonify({'status': 'ok', 'freed_mb': 0, 'output': result.stdout})
        else:
            return jsonify({'status': 'error', 'error': result.stderr}), 500
    except Exception as e:
        logger.error(f"Error running git gc: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/system/export-report', methods=['GET'])
def export_system_report():
    """Export comprehensive system report"""
    try:
        # Get current status
        status_response = get_system_status()
        status_data = status_response.get_json()

        # Add timestamp
        import datetime
        status_data['timestamp'] = datetime.datetime.now().isoformat()
        status_data['hostname'] = socket.gethostname()

        # Return as downloadable JSON
        import json
        from flask import Response
        return Response(
            json.dumps(status_data, indent=2),
            mimetype='application/json',
            headers={'Content-Disposition': 'attachment;filename=system-report.json'}
        )
    except Exception as e:
        logger.error(f"Error exporting system report: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/settings')
def settings_page():
    """Main settings page"""
    return render_template('settings.html')


@app.route('/settings/models')
def model_pool_settings():
    """Model Pool settings page - comprehensive model dashboard"""
    return render_template('settings.html')


@app.route('/settings/model-pool')
def model_pool_dashboard():
    """Model Pool dashboard"""
    return render_template('settings.html')


@app.route('/settings/llm')
def llm_settings():
    """LLM settings page"""
    return render_template('settings.html')


@app.route('/settings/model-pool-api')
def model_pool_api_settings():
    """Model Pool API settings page - external API models"""
    return render_template('settings.html')


@app.route('/model-pool')
def enhanced_model_pool_main():
    """Enhanced Model Pool dashboard - integrated into main HMI"""
    return render_template('model_pool_enhanced.html')


@app.route('/api/env/config', methods=['GET'])
def get_env_config():
    """Get .env configuration"""
    try:
        env_config = {}
        key_status = {}
        
        if os.path.exists(ENV_FILE_PATH):
            with open(ENV_FILE_PATH, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        value = value.strip('"\'')
                        env_config[key] = value
                        
                        # Check key status
                        if 'API_KEY' in key:
                            if 'your_' in value or len(value) < 10:
                                key_status[key.replace('_API_KEY', '')] = 'invalid'
                            else:
                                key_status[key.replace('_API_KEY', '')] = 'valid'
                        elif 'MODEL_NAME' in key and value:
                            key_status[key.replace('_MODEL_NAME', '')] = 'configured'
        
        return jsonify({
            'status': 'ok',
            'config': env_config,
            'keys': key_status
        })
    except Exception as e:
        logger.error(f"Error reading .env: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/models/api-status', methods=['GET'])
def get_api_models_status():
    """Get status of external API models only with usage statistics"""
    try:
        models = get_available_models()

        # Get usage data from database
        db = get_db_session()
        usage_by_model = {}
        try:
            messages = db.query(Message).filter(Message.usage_json.isnot(None)).all()
            for msg in messages:
                usage = msg.get_usage()
                if not usage:
                    continue
                model_key = msg.model_name or "unknown"
                if model_key not in usage_by_model:
                    usage_by_model[model_key] = {
                        'total_requests': 0,
                        'total_input_tokens': 0,
                        'total_output_tokens': 0,
                        'total_tokens': 0,
                        'total_cost': 0.0
                    }
                usage_by_model[model_key]['total_requests'] += 1
                usage_by_model[model_key]['total_input_tokens'] += usage.get('input_tokens', 0)
                usage_by_model[model_key]['total_output_tokens'] += usage.get('output_tokens', 0)
                usage_by_model[model_key]['total_tokens'] += usage.get('total_tokens', 0)
                usage_by_model[model_key]['total_cost'] += float(usage.get('cost', 0.0))
        except Exception as e:
            logger.warning(f"Could not load usage data: {e}")

        # Filter for API models only
        api_models = {}
        for model_id, model in models.items():
            if model['provider'] in ['openai', 'anthropic', 'google', 'deepseek', 'kimi']:
                # Get usage data for this model
                model_usage = usage_by_model.get(model['name'], {
                    'total_requests': 0,
                    'total_input_tokens': 0,
                    'total_output_tokens': 0,
                    'total_tokens': 0,
                    'total_cost': 0.0
                })

                api_models[model_id] = {
                    'name': model['name'],
                    'provider': model['provider'],
                    'status': model['status'],
                    'available': model['available'],
                    'description': model['description'],
                    'last_check': time.time(),
                    'api_provider': model['provider'].upper(),
                    'api_status': 'Configured' if model['available'] else 'Invalid Key',
                    'cost_per_1k_input': _get_model_cost(model['provider'], model['name']),
                    'cost_per_1k_output': _get_model_cost(model['provider'], model['name'], output=True),
                    'usage': model_usage
                }

        return jsonify({
            'status': 'ok',
            'models': api_models,
            'count': len(api_models)
        })
    except Exception as e:
        logger.error(f"Error getting API models status: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


def _get_model_cost(provider, model_name, output=False):
    """
    Get cost per 1K tokens for API models using dynamic pricing service.

    Uses intelligent caching with 24-hour TTL and falls back to static
    pricing if provider APIs are unavailable.

    Args:
        provider: Provider name (e.g., 'openai', 'anthropic', 'google')
        model_name: Model identifier (e.g., 'gpt-4-turbo', 'claude-3-5-sonnet-20241022')
        output: If True, return output token cost; if False, return input token cost

    Returns:
        float: Cost per 1K tokens
    """
    from llm_pricing import get_pricing_service

    try:
        pricing_service = get_pricing_service()
        input_cost, output_cost = pricing_service.get_pricing(provider, model_name)
        return output_cost if output else input_cost
    except Exception as e:
        logger.error(f"Pricing service error for {provider}/{model_name}: {e}")
        return 0.0


@app.route('/api/models/local-status', methods=['GET'])
def get_local_models_status():
    """Get status of local models only"""
    try:
        models = get_available_models()

        # Filter for local models only (includes 'local', 'ollama', 'local_fastapi')
        local_providers = ['local', 'ollama', 'local_fastapi']
        local_models = {}
        for model_id, model in models.items():
            if model['provider'] in local_providers:
                local_models[model_id] = {
                    'name': model['name'],
                    'provider': model['provider'],
                    'status': model['status'],
                    'available': model['available'],
                    'description': model['description'],
                    'last_check': time.time(),
                    'host': model.get('host', 'localhost'),
                    'port': model.get('port', 'N/A'),
                    'endpoint': f"http://{model.get('host', 'localhost')}:{model.get('port', 'N/A')}"
                }
        
        return jsonify({
            'status': 'ok',
            'models': local_models,
            'count': len(local_models)
        })
    except Exception as e:
        logger.error(f"Error getting local models status: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/models/test', methods=['POST'])
def test_model():
    """Test a model to verify it's working (local or API)"""
    try:
        data = request.get_json()
        model_id = data.get('model_id')

        if not model_id:
            return jsonify({'status': 'error', 'message': 'model_id is required'}), 400

        # Get model info
        models = get_available_models()
        model = models.get(model_id)

        if not model:
            return jsonify({'status': 'error', 'message': f'Model {model_id} not found'}), 404

        # Check if it's a local or API model
        if model['provider'] == 'local':
            # Test local model via health check
            host = model.get('host', 'localhost')
            port = model.get('port')
            endpoint = '/health'

            import httpx
            try:
                response = httpx.get(f'http://{host}:{port}{endpoint}', timeout=5.0)
                if response.status_code == 200:
                    return jsonify({
                        'status': 'ok',
                        'message': f'Local model {model_id} is responsive',
                        'health': response.json() if response.text else {}
                    })
                else:
                    return jsonify({
                        'status': 'error',
                        'message': f'Health check returned {response.status_code}'
                    }), 500
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Failed to connect to {host}:{port} - {str(e)}'
                }), 500

        else:
            # Test API model by checking if API key is valid
            if not model.get('available'):
                return jsonify({
                    'status': 'error',
                    'message': 'API key not configured or invalid'
                }), 400

            # For API models, we'll just verify the key format and return success
            # Full API test would cost money, so we rely on the existing validation
            return jsonify({
                'status': 'ok',
                'message': f'API model {model_id} is configured with valid credentials',
                'provider': model['provider']
            })

    except Exception as e:
        logger.error(f"Error testing model: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ============================================================================
# MODEL POOL PROXY API
# ============================================================================

@app.route('/api/model-pool/models', methods=['GET'])
def get_model_pool_models():
    """Proxy for Model Pool Manager /models endpoint"""
    try:
        import httpx
        response = httpx.get('http://localhost:8050/models', timeout=5.0)
        return jsonify(response.json())
    except Exception as e:
        logger.error(f"Error fetching model pool models: {e}")
        return jsonify({'models': [], 'total_memory_mb': 0, 'available_ports': 0, 'error': str(e)}), 200


@app.route('/api/model-pool/config', methods=['GET'])
def get_model_pool_config():
    """Proxy for Model Pool Manager /config endpoint"""
    try:
        import httpx
        response = httpx.get('http://localhost:8050/config', timeout=5.0)
        return jsonify(response.json())
    except Exception as e:
        logger.error(f"Error fetching model pool config: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/model-pool/config', methods=['PATCH'])
def update_model_pool_config():
    """Proxy for Model Pool Manager PATCH /config endpoint"""
    try:
        import httpx
        config = request.get_json()
        response = httpx.patch('http://localhost:8050/config', json=config, timeout=5.0)
        return jsonify(response.json())
    except Exception as e:
        logger.error(f"Error updating model pool config: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/model-pool/models/<path:model_id>/load', methods=['POST'])
def load_model_pool_model(model_id):
    """Proxy for Model Pool Manager POST /models/{id}/load endpoint"""
    try:
        import httpx
        response = httpx.post(f'http://localhost:8050/models/{model_id}/load', timeout=60.0)
        return jsonify(response.json())
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/model-pool/models/<path:model_id>/unload', methods=['POST'])
def unload_model_pool_model(model_id):
    """Proxy for Model Pool Manager POST /models/{id}/unload endpoint"""
    try:
        import httpx
        response = httpx.post(f'http://localhost:8050/models/{model_id}/unload', timeout=30.0)
        return jsonify(response.json())
    except Exception as e:
        logger.error(f"Error unloading model {model_id}: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/settings/enhanced-model-pool')
def enhanced_model_pool():
    """Enhanced Model Pool dashboard with API models"""
    return render_template('model_pool_enhanced.html')


@app.route('/api/model-pool/models/<path:model_id>/extend-ttl', methods=['POST'])
def extend_model_ttl(model_id):
    """Proxy for Model Pool Manager POST /models/{id}/extend-ttl endpoint"""
    try:
        import httpx
        data = request.get_json() or {}
        response = httpx.post(f'http://localhost:8050/models/{model_id}/extend-ttl', json=data, timeout=5.0)
        return jsonify(response.json())
    except Exception as e:
        logger.error(f"Error extending TTL for model {model_id}: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ============================================================================
# LLM CHAT API ENDPOINTS
# ============================================================================

@app.route('/api/agents', methods=['GET'])
def get_agents():
    """
    Get list of available agents from Registry.

    Returns agents organized by tier (Architect, Directors, Managers, Programmers).
    Frontend uses this to populate the agent selector dropdown.
    """
    try:
        # Query Registry @ 6121 for all registered agents
        response = requests.get(f'{REGISTRY_URL}/services', timeout=5)
        response.raise_for_status()
        data = response.json()

        # Registry returns {"items": [...]} - filter for agents only
        services_list = data.get('items', [])
        agents = []

        for service_data in services_list:
            # Only include agents (not models or tools)
            if service_data.get('type') != 'agent':
                continue

            # Extract agent info from labels and service data
            labels = service_data.get('labels') or {}
            agent_entry = {
                'agent_id': service_data.get('service_id'),
                'agent_name': service_data.get('name', service_data.get('service_id')),
                'port': labels.get('port'),
                'tier': labels.get('tier', 'unknown'),
                'status': service_data.get('status', 'unknown'),
                'role_icon': labels.get('icon') or _get_role_icon(labels.get('tier'))
            }
            agents.append(agent_entry)

        # Add "Direct Chat" option at the top (no PAS agent, direct LLM access)
        agents.insert(0, {
            'agent_id': 'direct',
            'agent_name': 'Direct Chat',
            'port': None,
            'tier': 'direct',
            'status': 'active',
            'role_icon': '💬',
            'filesystem_access': False
        })

        # Sort remaining agents by tier and name (excluding Direct Chat)
        tier_order = {'architect': 1, 'director': 2, 'manager': 3, 'programmer': 4}
        pas_agents = agents[1:]  # Skip Direct Chat
        pas_agents.sort(key=lambda x: (tier_order.get(x['tier'].lower(), 99), x['agent_name']))

        # Mark all PAS agents as having filesystem access
        for agent in pas_agents:
            agent['filesystem_access'] = True

        # Reconstruct with Direct Chat first
        agents = [agents[0]] + pas_agents

        # Fallback: If registry returns no PAS agents, provide Architect by default
        if len(agents) == 1:  # Only Direct Chat
            return jsonify({
                'status': 'ok',
                'agents': [
                    {
                        'agent_id': 'direct',
                        'agent_name': 'Direct Chat',
                        'port': None,
                        'tier': 'direct',
                        'status': 'active',
                        'role_icon': '💬',
                        'filesystem_access': False
                    },
                    {
                        'agent_id': 'architect',
                        'agent_name': 'Architect',
                        'port': 6110,
                        'tier': 'architect',
                        'status': 'unknown',
                        'role_icon': '🏛️',
                        'filesystem_access': True
                    }
                ],
                'count': 2,
                'fallback': True
            })

        return jsonify({
            'status': 'ok',
            'agents': agents,
            'count': len(agents)
        })

    except Exception as e:
        logger.error(f"Error fetching agents from Registry: {e}")
        # Fallback: Return default agents if Registry is unavailable
        return jsonify({
            'status': 'ok',
            'agents': [
                {
                    'agent_id': 'architect',
                    'agent_name': 'Architect',
                    'port': 6110,
                    'tier': 'architect',
                    'status': 'unknown',
                    'role_icon': '🏛️'
                }
            ],
            'count': 1,
            'fallback': True
        })


@app.route('/api/agent_context/<agent_id>', methods=['GET'])
def get_agent_context(agent_id):
    """
    Get agent's current context: model, status, and recent conversation history.

    This is called when user selects an agent in the dropdown to:
    1. Auto-switch the model selector to the agent's current model
    2. Load the agent's recent conversation history
    3. Show who the agent was last talking to

    Returns:
        current_model: str - Agent's active LLM model (e.g., "llama3.1:8b")
        status: str - Agent's current status (idle, executing, etc.)
        recent_conversations: List[dict] - Last 5 conversation threads with partner info
    """
    try:
        agent_context = {
            'agent_id': agent_id,
            'current_model': None,
            'status': 'unknown',
            'recent_conversations': []
        }

        # Special case: Direct Chat has no agent context
        if agent_id == 'direct':
            return jsonify({
                'status': 'ok',
                'context': agent_context
            })

        # Get agent's port from Registry
        response = requests.get(f'{REGISTRY_URL}/services', timeout=5)
        response.raise_for_status()
        services = response.json().get('items', [])

        agent_port = None
        for svc in services:
            if svc.get('service_id') == agent_id:
                labels = svc.get('labels') or {}
                agent_port = labels.get('port')
                agent_context['status'] = svc.get('status', 'unknown')
                break

        # Get agent's current model from its /health endpoint
        if agent_port:
            try:
                health_response = requests.get(f'http://localhost:{agent_port}/health', timeout=2)
                if health_response.status_code == 200:
                    health_data = health_response.json()
                    agent_context['current_model'] = health_data.get('llm_model')
            except Exception as e:
                logger.warning(f"Could not fetch health from agent {agent_id} on port {agent_port}: {e}")

        # Get recent conversation threads from Registry database
        # Look for threads where this agent was involved (as parent or child)
        registry_db_path = Path(__file__).parent.parent.parent / 'artifacts' / 'registry' / 'registry.db'

        if registry_db_path.exists():
            import sqlite3
            conn = sqlite3.connect(str(registry_db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Find recent threads where agent was parent or child
            # Map agent_id to agent name format used in DB (e.g., "director-docs" -> "Dir-Docs")
            agent_name_map = {
                'architect': 'Architect',
                'director-code': 'Dir-Code',
                'director-models': 'Dir-Models',
                'director-data': 'Dir-Data',
                'director-devsecops': 'Dir-DevSecOps',
                'director-docs': 'Dir-Docs',
            }
            # Add managers (Mgr-Code-01, etc.) and programmers (Prog-*, Programmer-*)
            agent_db_name = agent_name_map.get(agent_id, agent_id)

            cursor.execute("""
                SELECT
                    thread_id,
                    parent_agent_id,
                    child_agent_id,
                    status,
                    created_at,
                    updated_at,
                    (SELECT COUNT(*) FROM agent_conversation_messages WHERE thread_id = t.thread_id) as message_count
                FROM agent_conversation_threads t
                WHERE parent_agent_id LIKE ? OR child_agent_id LIKE ?
                ORDER BY updated_at DESC
                LIMIT 5
            """, (f'%{agent_db_name}%', f'%{agent_db_name}%'))

            threads = cursor.fetchall()

            for thread in threads:
                # Get first message to show context
                cursor.execute("""
                    SELECT from_agent, to_agent, content, created_at
                    FROM agent_conversation_messages
                    WHERE thread_id = ?
                    ORDER BY created_at ASC
                    LIMIT 1
                """, (thread['thread_id'],))

                first_msg = cursor.fetchone()

                # Determine conversation partner
                partner = thread['child_agent_id'] if agent_db_name in thread['parent_agent_id'] else thread['parent_agent_id']

                conversation = {
                    'thread_id': thread['thread_id'],
                    'partner': partner,
                    'status': thread['status'],
                    'message_count': thread['message_count'],
                    'created_at': thread['created_at'],
                    'updated_at': thread['updated_at'],
                    'preview': first_msg['content'][:100] + '...' if first_msg and len(first_msg['content']) > 100 else (first_msg['content'] if first_msg else '')
                }

                agent_context['recent_conversations'].append(conversation)

            conn.close()

        return jsonify({
            'status': 'ok',
            'context': agent_context
        })

    except Exception as e:
        logger.error(f"Error fetching agent context for {agent_id}: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """
    Get list of available LLM models.

    Returns model metadata including context window, cost, and capabilities.
    Frontend uses this to populate the model selector dropdown.
    """
    # Hardcoded model list for V1 (TODO: Load from Settings service in V2)
    models = [
        {
            'model_id': 'claude-sonnet-4',
            'model_name': 'Claude Sonnet 4',
            'provider': 'anthropic',
            'context_window': 200000,
            'cost_per_1m_input': 3.00,
            'cost_per_1m_output': 15.00,
            'capabilities': ['streaming', 'tools', 'vision']
        },
        {
            'model_id': 'claude-opus-4',
            'model_name': 'Claude Opus 4',
            'provider': 'anthropic',
            'context_window': 200000,
            'cost_per_1m_input': 15.00,
            'cost_per_1m_output': 75.00,
            'capabilities': ['streaming', 'tools', 'vision']
        },
        {
            'model_id': 'gpt-4-turbo',
            'model_name': 'GPT-4 Turbo',
            'provider': 'openai',
            'context_window': 128000,
            'cost_per_1m_input': 10.00,
            'cost_per_1m_output': 30.00,
            'capabilities': ['streaming', 'tools', 'vision']
        },
        {
            'model_id': 'llama-3.1-8b',
            'model_name': 'Llama 3.1 8B (Local)',
            'provider': 'local',
            'context_window': 8192,
            'cost_per_1m_input': 0.00,
            'cost_per_1m_output': 0.00,
            'capabilities': ['streaming']
        }
    ]

    return jsonify({
        'status': 'ok',
        'models': models,
        'count': len(models)
    })


@app.route('/api/chat/message', methods=['POST'])
def send_chat_message():
    """
    Submit a chat message to an agent.

    Body:
        session_id: Optional[str] - Resume existing session or None for new
        message: str - User message content
        agent_id: str - Target agent ID (e.g., "architect", "dir-code")
        model: str - Model name

    Returns:
        session_id: str - Session ID (new or existing)
        message_id: str - ID of user's message
        status: str - "processing"

    V1: No streaming, synchronous response.
    V2: Will trigger streaming via SSE.
    """
    try:
        data = request.get_json()

        # Extract parameters
        session_id = data.get('session_id')
        message_content = data.get('message', '').strip()
        agent_id = data.get('agent_id', 'architect')
        model_name = data.get('model', 'llama3.1:8b')

        # Validation
        if not message_content:
            return jsonify({'error': 'Message content is required'}), 400

        # Database session
        db = get_db_session()

        try:
            # Create or load conversation session
            if session_id:
                # Resume existing session
                session = db.query(ConversationSession).filter_by(session_id=session_id).first()
                if not session:
                    return jsonify({'error': 'Session not found'}), 404

                # Update timestamp and model (allow per-message model override)
                session.updated_at = datetime.utcnow().isoformat() + 'Z'
                session.model_name = model_name  # Override model for this message
            else:
                # Create new session
                # Determine parent role based on agent
                parent_role = _get_parent_role(agent_id)
                agent_name = _get_agent_name(agent_id)

                session = ConversationSession(
                    user_id='default_user',  # TODO: Add authentication in V2
                    agent_id=agent_id,
                    agent_name=agent_name,
                    parent_role=parent_role,
                    model_name=model_name
                )
                db.add(session)
                db.commit()
                session_id = session.session_id

            # Create user message
            user_message = Message(
                session_id=session_id,
                message_type='user',
                content=message_content
            )
            db.add(user_message)

            # Auto-generate title from first user message if session has no title
            if not session.title:
                # Take first 50 chars of message as title
                title = message_content[:50]
                if len(message_content) > 50:
                    title += '...'
                session.title = title

            db.commit()

            # V2: Create placeholder assistant message (will be streamed via SSE)
            # Frontend should immediately open SSE connection to /api/chat/stream/{session_id}
            assistant_message = Message(
                session_id=session_id,
                message_type='assistant',
                content='',  # Empty - will be filled via streaming
                agent_id=agent_id,
                model_name=model_name,
                status='streaming'  # Status indicates response is being streamed
            )
            db.add(assistant_message)
            db.commit()

            return jsonify({
                'status': 'ok',
                'session_id': session_id,
                'message_id': user_message.message_id,
                'assistant_message_id': assistant_message.message_id,
                'streaming': True  # Signal frontend to open SSE connection
            })

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error sending chat message: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/sessions/<session_id>/messages', methods=['GET'])
def get_session_messages(session_id):
    """
    Get all messages for a conversation session.

    Returns messages in chronological order.
    """
    try:
        db = get_db_session()

        try:
            # Load session
            session = db.query(ConversationSession).filter_by(session_id=session_id).first()
            if not session:
                return jsonify({'error': 'Session not found'}), 404

            # Load messages
            messages = db.query(Message).filter_by(session_id=session_id).order_by(Message.timestamp).all()

            return jsonify({
                'status': 'ok',
                'session': session.to_dict(),
                'messages': [msg.to_dict() for msg in messages],
                'count': len(messages)
            })

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error fetching session messages: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/sessions', methods=['GET'])
def get_chat_sessions():
    """
    Get all conversation sessions for the current user.

    Returns sessions sorted by most recent first.
    """
    try:
        db = get_db_session()

        try:
            # Load sessions (both active and archived, TODO: Filter by user_id in V2)
            sessions = db.query(ConversationSession)\
                .filter(ConversationSession.status.in_(['active', 'archived']))\
                .order_by(ConversationSession.updated_at.desc())\
                .all()

            return jsonify({
                'status': 'ok',
                'sessions': [session.to_dict() for session in sessions],
                'count': len(sessions)
            })

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error fetching chat sessions: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/sessions/<session_id>', methods=['PUT'])
def update_session(session_id):
    """
    Update a conversation session (e.g., rename title).

    Body: {
        "title": "New session title"
    }
    """
    try:
        data = request.get_json()
        title = data.get('title')

        if not title:
            return jsonify({'error': 'Title is required'}), 400

        db = get_db_session()

        try:
            # Load session
            session = db.query(ConversationSession).filter_by(session_id=session_id).first()
            if not session:
                return jsonify({'error': 'Session not found'}), 404

            # Update title and timestamp
            session.title = title
            session.updated_at = datetime.utcnow().isoformat() + 'Z'
            db.commit()

            return jsonify({
                'status': 'ok',
                'session': session.to_dict()
            })

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error updating session: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/sessions/<session_id>/archive', methods=['POST'])
def archive_session(session_id):
    """
    Archive a conversation session (soft delete).

    Creates a new active session and returns its ID.
    """
    try:
        db = get_db_session()

        try:
            # Load session to archive
            session = db.query(ConversationSession).filter_by(session_id=session_id).first()
            if not session:
                return jsonify({'error': 'Session not found'}), 404

            # Archive the session
            session.status = 'archived'
            session.archived_at = datetime.utcnow().isoformat() + 'Z'
            session.updated_at = session.archived_at
            db.commit()

            # Create a new session with same agent/model
            new_session = ConversationSession(
                user_id=session.user_id,
                agent_id=session.agent_id,
                agent_name=session.agent_name,
                parent_role=session.parent_role,
                model_name=session.model_name,
                model_id=session.model_id
            )
            db.add(new_session)
            db.commit()

            return jsonify({
                'status': 'archived',
                'archived_session_id': session_id,
                'new_session_id': new_session.session_id
            })

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error archiving session: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """
    Delete a conversation session permanently.

    This is a hard delete - messages will be cascade deleted.
    """
    try:
        db = get_db_session()

        try:
            # Load session
            session = db.query(ConversationSession).filter_by(session_id=session_id).first()
            if not session:
                return jsonify({'error': 'Session not found'}), 404

            # Delete session (messages cascade deleted automatically)
            db.delete(session)
            db.commit()

            return jsonify({
                'status': 'deleted',
                'session_id': session_id
            })

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error deleting session: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/sessions', methods=['DELETE'])
def delete_all_sessions():
    """
    Delete ALL conversation sessions permanently.
    
    This is a hard delete - all sessions and their messages will be cascade deleted.
    Use with extreme caution!
    """
    try:
        db = get_db_session()

        try:
            # Count sessions before deletion for response
            session_count = db.query(ConversationSession).count()
            
            if session_count == 0:
                return jsonify({
                    'status': 'ok',
                    'deleted_count': 0,
                    'message': 'No sessions to delete'
                })

            # Delete all sessions (messages cascade deleted automatically)
            db.query(ConversationSession).delete()
            db.commit()

            return jsonify({
                'status': 'ok',
                'deleted_count': session_count,
                'message': f'Successfully deleted {session_count} sessions'
            })

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error deleting all sessions: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/sessions/<session_id>/export', methods=['GET'])
def export_session(session_id):
    """
    Export a conversation session to Markdown or JSON.

    Query params:
    - format: 'markdown' or 'json' (default: 'markdown')

    Returns file download with appropriate Content-Disposition header.
    """
    try:
        export_format = request.args.get('format', 'markdown').lower()

        if export_format not in ['markdown', 'json']:
            return jsonify({'error': 'Invalid format. Use "markdown" or "json"'}), 400

        db = get_db_session()

        try:
            # Load session with messages
            session = db.query(ConversationSession).filter_by(session_id=session_id).first()
            if not session:
                return jsonify({'error': 'Session not found'}), 404

            messages = db.query(Message)\
                .filter_by(session_id=session_id)\
                .order_by(Message.timestamp)\
                .all()

            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"pas_conversation_{session_id[:8]}_{timestamp}.{export_format}"

            if export_format == 'markdown':
                content = _export_to_markdown(session, messages)
                mimetype = 'text/markdown'
            else:  # json
                content = _export_to_json(session, messages)
                mimetype = 'application/json'

            response = Response(content, mimetype=mimetype)
            response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error exporting session: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


def _export_to_markdown(session: ConversationSession, messages: list) -> str:
    """
    Convert conversation to Markdown format.

    Includes headers, timestamps, code blocks, and metadata.
    """
    lines = []

    # Header
    lines.append(f"# PAS Conversation Export")
    lines.append("")
    lines.append(f"**Session ID:** `{session.session_id}`")
    lines.append(f"**Agent:** {session.agent_name} ({session.agent_id})")
    lines.append(f"**Model:** {session.model_name}")
    lines.append(f"**Your Role:** {session.parent_role}")
    lines.append(f"**Created:** {session.created_at}")
    lines.append(f"**Messages:** {len(messages)}")
    if session.title:
        lines.append(f"**Title:** {session.title}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Messages
    for msg in messages:
        # Role header
        if msg.message_type == 'user':
            lines.append(f"## 👤 User")
        elif msg.message_type == 'assistant':
            icon = _get_role_icon(session.agent_id.split('-')[0])
            lines.append(f"## {icon} {session.agent_name}")
        else:
            lines.append(f"## 🔔 {msg.message_type.title()}")

        # Timestamp
        lines.append(f"*{msg.timestamp}*")
        lines.append("")

        # Content
        lines.append(msg.content)
        lines.append("")

        # Metadata (usage, status)
        if msg.status:
            lines.append(f"**Status:** {msg.status}")

        usage = msg.get_usage()
        if usage:
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)
            cost_usd = usage.get('cost_usd', 0)
            lines.append(f"**Tokens:** {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total")
            if cost_usd:
                lines.append(f"**Cost:** ${cost_usd:.4f}")

        lines.append("")
        lines.append("---")
        lines.append("")

    # Footer
    lines.append("")
    lines.append("*Generated by PAS LLM Task Interface*")
    lines.append(f"*Exported: {datetime.now().isoformat()}*")

    return "\n".join(lines)


def _export_to_json(session: ConversationSession, messages: list) -> str:
    """
    Convert conversation to JSON format.

    Includes all metadata, usage tracking, and timestamps.
    """
    data = {
        'session': session.to_dict(),
        'messages': [msg.to_dict() for msg in messages],
        'export_metadata': {
            'exported_at': datetime.now().isoformat() + 'Z',
            'message_count': len(messages),
            'format_version': '1.0'
        }
    }

    return json.dumps(data, indent=2)


@app.route('/api/chat/stream/<session_id>', methods=['GET'])
def stream_chat_response(session_id):
    """
    SSE endpoint for streaming chat responses.

    Event Types:
    - token: Streaming text content
    - status_update: Task progress updates (planning, executing, complete, error)
    - usage: Token/cost tracking (sent once before done)
    - done: Stream complete signal

    Heartbeats: :keep-alive every 15s
    """
    def generate():
        db = None
        assistant_message = None
        accumulated_content = ''

        try:
            db = get_db_session()

            # Verify session exists
            session = db.query(ConversationSession).filter_by(session_id=session_id).first()
            if not session:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Session not found'})}\n\n"
                return

            # Get the latest user and assistant messages
            user_message = db.query(Message)\
                .filter_by(session_id=session_id, message_type='user')\
                .order_by(Message.timestamp.desc())\
                .first()

            assistant_message = db.query(Message)\
                .filter_by(session_id=session_id, message_type='assistant')\
                .order_by(Message.timestamp.desc())\
                .first()

            if not user_message or not assistant_message:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Messages not found'})}\n\n"
                return

            # Get agent_id and model from session
            agent_id = session.agent_id
            model = session.model_name

            # Retrieve full conversation history for context
            all_messages = db.query(Message)\
                .filter_by(session_id=session_id)\
                .filter(Message.message_type.in_(['user', 'assistant']))\
                .order_by(Message.timestamp.asc())\
                .all()

            # Build messages array for Gateway (exclude empty assistant messages)
            messages = []
            for msg in all_messages:
                if msg.message_type == 'assistant' and not msg.content.strip():
                    continue  # Skip placeholder assistant messages
                messages.append({
                    'role': msg.message_type,
                    'content': msg.content
                })

            # Forward request to Gateway for real agent communication
            try:
                # PRD-aligned: try GET first
                try:
                    gateway_response = requests.get(
                        f'{GATEWAY_URL}/chat/stream/{session_id}',
                        stream=True,
                        timeout=300
                    )
                except requests.exceptions.RequestException:
                    gateway_response = None

                if gateway_response is None or gateway_response.status_code != 200:
                    # Fallback to legacy POST body streaming with full conversation history
                    gateway_response = requests.post(
                        f'{GATEWAY_URL}/chat/stream',
                        json={
                            'session_id': session_id,
                            'message_id': user_message.message_id,
                            'agent_id': agent_id,
                            'model': model,
                            'content': user_message.content,  # Legacy field for backward compatibility
                            'messages': messages  # Full conversation history
                        },
                        stream=True,
                        timeout=300
                    )

                if gateway_response.status_code != 200:
                    # Fallback to mock streaming if Gateway unavailable
                    logger.warning(f"Gateway returned {gateway_response.status_code}, using mock streaming")

                    # Manually iterate through mock stream to accumulate content, honoring cancel
                    for event in _mock_stream_response(user_message.content):
                        # Check cancellation
                        with _CANCEL_LOCK:
                            cancelled = session_id in _CANCELLED_SESSIONS
                        if cancelled:
                            # Mark DB and clear cancel flag
                            assistant_message.content = accumulated_content
                            assistant_message.status = 'cancelled'
                            db.commit()
                            with _CANCEL_LOCK:
                                _CANCELLED_SESSIONS.discard(session_id)
                            yield "data: {\"type\": \"done\"}\n\n"
                            return

                        yield event
                        # Parse to accumulate content
                        if event.startswith('data: '):
                            try:
                                event_data = json.loads(event[6:])
                                if event_data.get('type') == 'token':
                                    accumulated_content += event_data.get('content', '')
                            except json.JSONDecodeError:
                                pass

                    # Update assistant message in database
                    assistant_message.content = accumulated_content
                    assistant_message.status = 'complete'
                    db.commit()
                    return

                # Stream events from Gateway
                last_heartbeat = time.time()
                usage_info = None

                for line in gateway_response.iter_lines():
                    if not line:
                        continue

                    # Decode line
                    line_str = line.decode('utf-8')

                    # Honor cancellation mid-stream
                    with _CANCEL_LOCK:
                        cancelled = session_id in _CANCELLED_SESSIONS
                    if cancelled:
                        try:
                            requests.post(f"{GATEWAY_URL}/chat/{session_id}/cancel", timeout=2)
                        except Exception:
                            pass
                        assistant_message.content = accumulated_content
                        assistant_message.status = 'cancelled'
                        db.commit()
                        with _CANCEL_LOCK:
                            _CANCELLED_SESSIONS.discard(session_id)
                        yield "data: {\"type\": \"done\"}\n\n"
                        break

                    # Forward SSE event
                    if line_str.startswith('data: '):
                        yield f"{line_str}\n\n"

                        # Parse event to accumulate tokens and check for done
                        try:
                            event_data = json.loads(line_str[6:])  # Remove 'data: ' prefix

                            if event_data.get('type') == 'token':
                                accumulated_content += event_data.get('content', '')

                            elif event_data.get('type') == 'usage':
                                usage_info = event_data.get('usage', {})
                                # Persist cost_usd if provided
                                if 'cost_usd' in event_data and event_data['cost_usd'] is not None:
                                    try:
                                        usage_info['cost_usd'] = float(event_data['cost_usd'])
                                    except (TypeError, ValueError):
                                        pass

                            elif event_data.get('type') == 'done':
                                # Update assistant message in database
                                assistant_message.content = accumulated_content
                                assistant_message.status = 'complete'
                                if usage_info:
                                    assistant_message.set_usage(usage_info)
                                db.commit()
                                break

                        except json.JSONDecodeError:
                            pass

                    # Send heartbeat every 15s
                    if time.time() - last_heartbeat > 15:
                        yield ":keep-alive\n\n"
                        last_heartbeat = time.time()

            except requests.exceptions.RequestException as e:
                logger.error(f"Gateway connection error: {e}", exc_info=True)
                # Fallback to mock streaming

                # Manually iterate through mock stream to accumulate content, honoring cancel
                for event in _mock_stream_response(user_message.content):
                    # Check cancellation
                    with _CANCEL_LOCK:
                        cancelled = session_id in _CANCELLED_SESSIONS
                    if cancelled:
                        assistant_message.content = accumulated_content
                        assistant_message.status = 'cancelled'
                        db.commit()
                        with _CANCEL_LOCK:
                            _CANCELLED_SESSIONS.discard(session_id)
                        yield "data: {\"type\": \"done\"}\n\n"
                        return

                    yield event
                    # Parse to accumulate content
                    if event.startswith('data: '):
                        try:
                            event_data = json.loads(event[6:])
                            if event_data.get('type') == 'token':
                                accumulated_content += event_data.get('content', '')
                        except json.JSONDecodeError:
                            pass

                # Update assistant message in database
                assistant_message.content = accumulated_content
                assistant_message.status = 'complete'
                db.commit()

        except Exception as e:
            logger.error(f"Error in stream_chat_response: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            if db:
                db.close()

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


# --- Cancellation support ---
from threading import Lock
_CANCEL_LOCK = Lock()
_CANCELLED_SESSIONS = set()


@app.route('/api/chat/sessions/<session_id>/cancel', methods=['POST'])
def cancel_chat_session(session_id):
    """Cancel an active streaming session."""
    with _CANCEL_LOCK:
        _CANCELLED_SESSIONS.add(session_id)
    return jsonify({'status': 'cancelled', 'session_id': session_id})


def _mock_stream_response(original_message: str):
    """
    Mock SSE streaming for testing when Gateway is unavailable.
    Simulates realistic streaming with all 4 event types.
    """
    try:
        # Status update: Planning
        yield f"data: {json.dumps({'type': 'status_update', 'status': 'planning', 'detail': 'Analyzing your request...'})}\n\n"
        time.sleep(0.3)

        # Status update: Executing
        yield f"data: {json.dumps({'type': 'status_update', 'status': 'executing', 'detail': 'Generating response...'})}\n\n"
        time.sleep(0.2)

        # Stream response tokens
        mock_response = f"I received your message: '{original_message[:50]}...'. This is a mock streaming response while the Gateway is being integrated. Each word will appear progressively to simulate real LLM streaming."

        words = mock_response.split(' ')
        for i, word in enumerate(words):
            # Add space before word (except first)
            token = word if i == 0 else f" {word}"
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            time.sleep(0.05)  # Simulate typing speed

        # Status update: Complete
        yield f"data: {json.dumps({'type': 'status_update', 'status': 'complete', 'detail': 'Response generated'})}\n\n"
        time.sleep(0.1)

        # Usage tracking
        usage_data = {
            'type': 'usage',
            'usage': {
                'prompt_tokens': 50,
                'completion_tokens': len(words),
                'total_tokens': 50 + len(words)
            },
            'cost_usd': 0.001
        }
        yield f"data: {json.dumps(usage_data)}\n\n"
        time.sleep(0.1)

        # Done signal
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        logger.error(f"Error in _mock_stream_response: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


# Helper functions for LLM chat

def _get_role_icon(tier: str) -> str:
    """Get emoji icon for agent tier"""
    tier_lower = tier.lower() if tier else ''
    icons = {
        'architect': '🏛️',
        'director': '💻',
        'manager': '⚙️',
        'programmer': '👨‍💻'
    }
    return icons.get(tier_lower, '🤖')


def _get_agent_name(agent_id: str) -> str:
    """Get display name for agent ID"""
    name_map = {
        'architect': 'Architect',
        'dir-code': 'Dir-Code',
        'dir-models': 'Dir-Models',
        'dir-data': 'Dir-Data',
        'dir-devsecops': 'Dir-DevSecOps',
        'dir-docs': 'Dir-Docs'
    }
    return name_map.get(agent_id.lower(), agent_id.title())


def _get_parent_role(agent_id: str) -> str:
    """Determine parent role based on target agent"""
    agent_lower = agent_id.lower()
    if agent_lower == 'architect':
        return 'PAS Root'
    elif agent_lower.startswith('dir-'):
        return 'Architect'
    elif agent_lower.startswith('mgr-'):
        return 'Director'
    elif agent_lower.startswith('programmer-'):
        return 'Manager'
    else:
        return 'User'


# === Agent Chat Test API ===
# These endpoints allow browser-based tests to create threads and send messages
# using the real agent chat system

agent_chat_client = AgentChatClient()
comms_logger = CommsLogger()


@app.route('/api/agent-chat/test/create-thread', methods=['POST'])
def api_create_test_thread():
    """
    Create a test agent chat thread.

    Request body:
    {
        "parent_agent": "HMI-Test",
        "child_agent": "Dir-Code",
        "message": "Test message content"
    }

    Returns thread_id for subsequent messages.
    """
    try:
        data = request.get_json()
        parent_agent = data.get('parent_agent', 'HMI-Test')
        child_agent = data.get('child_agent')
        message = data.get('message', 'Test message')

        if not child_agent:
            return jsonify({'error': 'child_agent is required'}), 400

        # Create thread using agent chat client
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        thread = loop.run_until_complete(
            agent_chat_client.create_thread(
                run_id=f"test-{int(time.time())}",
                parent_agent_id=parent_agent,
                child_agent_id=child_agent,
                initial_message=message,
                metadata={'test': True, 'source': 'hmi-test'}
            )
        )

        loop.close()

        return jsonify({
            'status': 'created',
            'thread_id': thread.thread_id,
            'parent_agent': parent_agent,
            'child_agent': child_agent,
            'message_count': len(thread.messages)
        })

    except Exception as e:
        logger.error(f"Failed to create test thread: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/agent-chat/test/send-to-agent', methods=['POST'])
def api_send_test_message_to_agent():
    """
    Send a test message to an agent via its /agent_chat/receive endpoint.

    Creates a thread, sends initial message, then forwards to the agent.

    Request body:
    {
        "agent_port": 6111,
        "agent_name": "Dir-Code",
        "message": "Test message",
        "message_type": "question"
    }
    """
    try:
        data = request.get_json()
        agent_port = data.get('agent_port')
        agent_name = data.get('agent_name', f'Agent-{agent_port}')
        message = data.get('message', 'Test message')
        message_type = data.get('message_type', 'question')

        if not agent_port:
            return jsonify({'error': 'agent_port is required'}), 400

        # Create thread
        import asyncio
        import uuid
        from datetime import datetime, timezone

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        run_id = f"test-{int(time.time())}"

        thread = loop.run_until_complete(
            agent_chat_client.create_thread(
                run_id=run_id,
                parent_agent_id='HMI-Test',
                child_agent_id=agent_name,
                initial_message=message,
                metadata={'test': True, 'agent_port': agent_port}
            )
        )

        # Log thread creation
        comms_logger.log_cmd(
            from_agent='HMI-Test',
            to_agent=agent_name,
            message=f"Agent Family Test: {message}",
            run_id=run_id,
            metadata={
                'thread_id': thread.thread_id,
                'agent_port': agent_port,
                'test_type': 'agent_family'
            }
        )

        # Get the initial message to send to agent
        initial_msg = thread.messages[0]

        # Try /agent/chat/send first (Architect), fallback to /agent_chat/receive (Directors/Managers)
        # Architect uses simple AgentChatRequest, Directors/Managers use full AgentChatMessage

        # Try Architect endpoint first
        agent_url = f'http://localhost:{agent_port}/agent/chat/send'
        msg_payload = {
            'sender_agent': 'HMI-Test',
            'message_type': message_type,
            'content': message,
            'metadata': {
                'thread_id': thread.thread_id,
                'test': True,
                'agent_port': agent_port
            }
        }

        response = requests.post(agent_url, json=msg_payload, timeout=5)

        # If 404, try Director/Manager endpoint
        if response.status_code == 404:
            agent_url = f'http://localhost:{agent_port}/agent_chat/receive'
            msg_payload = {
                'message_id': initial_msg.message_id,
                'thread_id': thread.thread_id,
                'from_agent': 'HMI-Test',
                'to_agent': agent_name,
                'message_type': message_type,
                'content': message,
                'created_at': initial_msg.created_at,
                'metadata': {
                    'thread_id': thread.thread_id,
                    'test': True,
                    'agent_port': agent_port
                }
            }
            response = requests.post(agent_url, json=msg_payload, timeout=5)

        loop.close()

        if response.ok:
            # Log successful response
            comms_logger.log_response(
                from_agent=agent_name,
                to_agent='HMI-Test',
                message=f"Test accepted: HTTP {response.status_code}",
                run_id=run_id,
                status='success',
                metadata={
                    'thread_id': thread.thread_id,
                    'agent_port': agent_port,
                    'http_status': response.status_code
                }
            )
            return jsonify({
                'status': 'sent',
                'thread_id': thread.thread_id,
                'agent': agent_name,
                'port': agent_port,
                'response': response.json() if response.content else None
            })
        else:
            # Log failed response
            comms_logger.log_response(
                from_agent=agent_name,
                to_agent='HMI-Test',
                message=f"Test failed: HTTP {response.status_code}",
                run_id=run_id,
                status='failed',
                metadata={
                    'thread_id': thread.thread_id,
                    'agent_port': agent_port,
                    'http_status': response.status_code,
                    'error': response.text[:200] if response.text else None
                }
            )
            return jsonify({
                'status': 'failed',
                'error': f'HTTP {response.status_code}',
                'thread_id': thread.thread_id
            }), response.status_code

    except Exception as e:
        logger.error(f"Failed to send test message: {e}")

        # Log exception
        comms_logger.log_response(
            from_agent='HMI-Test',
            to_agent='System',
            message=f"Test exception: {str(e)[:100]}",
            run_id=data.get('run_id', 'unknown') if 'data' in locals() else 'unknown',
            status='error',
            metadata={'error_type': type(e).__name__, 'error': str(e)[:500]}
        )

        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("Starting Flask HMI App on port 6101...")

    # Initialize last_known_log_id from database
    try:
        db_path = get_db_path()
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(log_id) FROM action_logs")
            result = cursor.fetchone()
            conn.close()
            if result and result[0]:
                last_known_log_id = result[0]
                logger.info(f"Initialized last_known_log_id to {last_known_log_id}")
    except Exception as e:
        logger.warning(f"Could not initialize last_known_log_id: {e}")

    # Start background polling thread for action_logs
    polling_thread = threading.Thread(target=poll_action_logs, daemon=True)
    polling_thread.start()
    logger.info("Started background action_logs polling thread")

    app.run(host='127.0.0.1', port=6101, debug=True)

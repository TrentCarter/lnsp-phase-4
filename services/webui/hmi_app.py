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

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import logging
from datetime import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'pas-hmi-secret'
CORS(app)

# Service endpoints
REGISTRY_URL = 'http://localhost:6121'
HEARTBEAT_MONITOR_URL = 'http://localhost:6109'
RESOURCE_MANAGER_URL = 'http://localhost:6104'
TOKEN_GOVERNOR_URL = 'http://localhost:6105'
EVENT_STREAM_URL = 'http://localhost:6102'
PROVIDER_ROUTER_URL = 'http://localhost:6103'
GATEWAY_URL = 'http://localhost:6120'


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


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'hmi_app',
        'port': 6101,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/services', methods=['GET'])
def get_services():
    """Get all registered services from Registry"""
    try:
        response = requests.get(f'{REGISTRY_URL}/services', timeout=5)
        response.raise_for_status()
        data = response.json()
        # Transform 'items' to 'services' for frontend compatibility
        return jsonify({'services': data.get('items', [])})
    except Exception as e:
        logger.error(f"Error fetching services: {e}")
        return jsonify({'error': str(e), 'services': []}), 500


@app.route('/api/tree', methods=['GET'])
def get_tree_data():
    """
    Get agent hierarchy tree data.

    Returns tree structure compatible with D3.js:
    {
        "name": "Root",
        "children": [...]
    }
    """
    try:
        # Fetch services from Registry
        response = requests.get(f'{REGISTRY_URL}/services', timeout=5)
        response.raise_for_status()
        services = response.json().get('items', [])

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
        parent_name = service.get('labels', {}).get('parent', '').strip()

        if not parent_name or parent_name == 'null':
            # No parent - add to root
            if service_id in nodes:
                root_children.append(nodes[service_id])
        else:
            # Find parent by matching name
            parent_id = None
            for sid, node in nodes.items():
                # Match by service_id suffix (e.g., "agent-architect" matches "architect")
                if sid.endswith(f"-{parent_name}") or node['name'].lower() == parent_name.lower():
                    parent_id = sid
                    break

            if parent_id and parent_id in nodes:
                nodes[parent_id]['children'].append(nodes[service_id])
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

        metrics['summary'] = {
            'total_services': total_services,
            'healthy_services': healthy_services,
            'health_percentage': (healthy_services / total_services * 100) if total_services > 0 else 0
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
    try:
        # Fetch agents from Registry
        response = requests.get(f'{REGISTRY_URL}/services', timeout=5)
        response.raise_for_status()
        services = response.json().get('items', [])

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
            # Generate mock tasks for demo if event stream is unavailable
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
    """Get all tasks from action logs"""
    try:
        response = requests.get(f'{REGISTRY_URL}/action_logs/tasks', timeout=5)
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        logger.error(f"Error fetching action tasks: {e}")
        return jsonify({'error': str(e), 'items': []}), 500


@app.route('/api/actions/task/<task_id>', methods=['GET'])
def get_task_actions(task_id):
    """Get hierarchical actions for a specific task"""
    try:
        response = requests.get(f'{REGISTRY_URL}/action_logs/task/{task_id}', timeout=5)
        response.raise_for_status()
        return jsonify(response.json())
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


if __name__ == '__main__':
    logger.info("Starting Flask HMI App on port 6101...")
    app.run(host='127.0.0.1', port=6101, debug=True)

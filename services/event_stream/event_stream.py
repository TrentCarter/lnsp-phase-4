"""
Event Stream Service (Port 6102)
WebSocket server for broadcasting real-time events to HMI clients.

Receives events from Phase 0+1 services and broadcasts to all connected clients.
"""

from flask import Flask, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import logging
from datetime import datetime
from typing import Dict, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'pas-event-stream-secret'
CORS(app)

# Initialize SocketIO with CORS support
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# In-memory event buffer (last 100 events for new clients)
event_buffer = []
MAX_BUFFER_SIZE = 100

# Connected clients counter
connected_clients = 0


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'event_stream',
        'port': 6102,
        'connected_clients': connected_clients,
        'buffered_events': len(event_buffer),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/broadcast', methods=['POST'])
def broadcast_event():
    """
    HTTP endpoint for services to broadcast events.

    Expected JSON payload:
    {
        "event_type": "heartbeat|status_update|alert|...",
        "data": {...}
    }
    """
    from flask import request

    try:
        payload = request.get_json()
        if not payload:
            return jsonify({'error': 'No JSON payload'}), 400

        event_type = payload.get('event_type', 'unknown')
        event_data = payload.get('data', {})

        # Add timestamp if not present
        if 'timestamp' not in event_data:
            event_data['timestamp'] = datetime.now().isoformat()

        # Create full event
        full_event = {
            'event_type': event_type,
            'data': event_data,
            'server_timestamp': datetime.now().isoformat()
        }

        # Add to buffer
        add_to_buffer(full_event)

        # Broadcast to all connected clients (no broadcast parameter needed)
        socketio.emit('event', full_event)

        logger.info(f"Broadcast {event_type} to {connected_clients} clients")

        return jsonify({
            'status': 'broadcasted',
            'event_type': event_type,
            'clients': connected_clients
        })

    except Exception as e:
        logger.error(f"Broadcast error: {e}")
        return jsonify({'error': str(e)}), 500


def add_to_buffer(event: Dict[str, Any]):
    """Add event to circular buffer"""
    global event_buffer
    event_buffer.append(event)
    if len(event_buffer) > MAX_BUFFER_SIZE:
        event_buffer.pop(0)


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    global connected_clients
    connected_clients += 1
    logger.info(f"Client connected. Total clients: {connected_clients}")

    # Send connection acknowledgment
    emit('connected', {
        'status': 'connected',
        'server_time': datetime.now().isoformat(),
        'buffered_events': len(event_buffer)
    })

    # Send buffered events to new client
    if event_buffer:
        emit('event_history', event_buffer)


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    global connected_clients
    connected_clients = max(0, connected_clients - 1)
    logger.info(f"Client disconnected. Total clients: {connected_clients}")


@socketio.on('ping')
def handle_ping():
    """Handle ping from client"""
    emit('pong', {'timestamp': datetime.now().isoformat()})


@socketio.on('request_history')
def handle_request_history():
    """Send event history to client"""
    emit('event_history', event_buffer)


# Convenience methods for different event types
def broadcast_heartbeat(service_id: str, service_name: str, data: Dict[str, Any]):
    """Broadcast heartbeat event"""
    event = {
        'event_type': 'heartbeat',
        'data': {
            'service_id': service_id,
            'service_name': service_name,
            'timestamp': datetime.now().isoformat(),
            **data
        }
    }
    add_to_buffer(event)
    socketio.emit('event', event)


def broadcast_alert(alert_type: str, service_id: str, message: str, severity: str = 'warning'):
    """Broadcast alert event"""
    event = {
        'event_type': 'alert',
        'data': {
            'alert_type': alert_type,
            'service_id': service_id,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
    }
    add_to_buffer(event)
    socketio.emit('event', event)


def broadcast_status_update(service_id: str, status: str, data: Dict[str, Any]):
    """Broadcast status update event"""
    event = {
        'event_type': 'status_update',
        'data': {
            'service_id': service_id,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            **data
        }
    }
    add_to_buffer(event)
    socketio.emit('event', event)


if __name__ == '__main__':
    logger.info("Starting Event Stream Service on port 6102...")
    socketio.run(app, host='127.0.0.1', port=6102, debug=False, allow_unsafe_werkzeug=True)

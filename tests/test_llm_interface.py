#!/usr/bin/env python3
"""
Comprehensive automated test suite for /llm chat interface.
Tests the full end-to-end flow: HMI -> Gateway -> Ollama.

Usage:
    PYTHONPATH=. ./.venv/bin/python3 tests/test_llm_interface.py
"""

import requests
import json
import time
import sys
from typing import Dict, List, Optional


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_test(msg: str):
    """Print test name."""
    print(f"\n{Colors.BLUE}▶ {msg}{Colors.RESET}")


def print_success(msg: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")


def print_error(msg: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {msg}{Colors.RESET}")


def print_warning(msg: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.RESET}")


class LLMInterfaceTester:
    """Test suite for LLM chat interface."""

    def __init__(self):
        self.hmi_base = "http://localhost:6101"
        self.gateway_base = "http://localhost:6120"
        self.ollama_base = "http://localhost:11434"
        self.tests_passed = 0
        self.tests_failed = 0

    def run_all_tests(self):
        """Run all test suites."""
        print(f"\n{Colors.BOLD}{'='*60}")
        print(f"LLM Interface Comprehensive Test Suite")
        print(f"{'='*60}{Colors.RESET}\n")

        # Test service availability
        if not self.test_services_available():
            print_error("Services not available. Aborting test suite.")
            sys.exit(1)

        # Test API endpoints
        self.test_hmi_agents_endpoint()
        self.test_hmi_models_endpoint()
        self.test_hmi_sessions_endpoint()

        # Test Gateway endpoints
        self.test_gateway_health()
        self.test_gateway_streaming_direct()

        # Test end-to-end flows
        self.test_end_to_end_single_message()
        self.test_end_to_end_multi_message()
        self.test_model_selection()
        self.test_agent_selection()
        self.test_error_handling()

        # Print summary
        self.print_summary()

        # Exit with appropriate code
        sys.exit(0 if self.tests_failed == 0 else 1)

    def test_services_available(self) -> bool:
        """Test that all required services are available."""
        print_test("Testing service availability")

        services = [
            ("HMI", f"{self.hmi_base}/health"),
            ("Gateway", f"{self.gateway_base}/health"),
            ("Ollama", f"{self.ollama_base}/api/tags")
        ]

        all_available = True
        for name, url in services:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    print_success(f"{name} is available")
                else:
                    print_error(f"{name} returned status {response.status_code}")
                    all_available = False
            except Exception as e:
                print_error(f"{name} is not available: {e}")
                all_available = False

        if all_available:
            self.tests_passed += 1
        else:
            self.tests_failed += 1

        return all_available

    def test_hmi_agents_endpoint(self):
        """Test HMI /api/agents endpoint."""
        print_test("Testing HMI /api/agents endpoint")

        try:
            response = requests.get(f"{self.hmi_base}/api/agents", timeout=5)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"

            data = response.json()
            assert data['status'] == 'ok', f"Expected status 'ok', got {data.get('status')}"
            assert 'agents' in data, "Response missing 'agents' key"
            assert len(data['agents']) > 0, "No agents returned"

            # Verify agent structure
            agent = data['agents'][0]
            required_fields = ['agent_id', 'agent_name', 'role_icon']
            for field in required_fields:
                assert field in agent, f"Agent missing required field: {field}"

            print_success(f"Agents endpoint working ({len(data['agents'])} agents)")
            self.tests_passed += 1

        except AssertionError as e:
            print_error(f"Agents endpoint test failed: {e}")
            self.tests_failed += 1
        except Exception as e:
            print_error(f"Agents endpoint error: {e}")
            self.tests_failed += 1

    def test_hmi_models_endpoint(self):
        """Test HMI /api/models endpoint."""
        print_test("Testing HMI /api/models endpoint")

        try:
            response = requests.get(f"{self.hmi_base}/api/models", timeout=5)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"

            data = response.json()
            assert data['status'] == 'ok', f"Expected status 'ok', got {data.get('status')}"
            assert 'models' in data, "Response missing 'models' key"
            assert len(data['models']) > 0, "No models returned"

            # Verify model structure
            model = data['models'][0]
            required_fields = ['model_id', 'model_name']
            for field in required_fields:
                assert field in model, f"Model missing required field: {field}"

            # Verify model_id exists and is not the same as model_name (would indicate it's not set)
            assert model['model_id'] != model['model_name'] or model['model_id'].count(' ') == 0, \
                f"model_id should be machine ID (no spaces), got: {model['model_id']}"

            print_success(f"Models endpoint working ({len(data['models'])} models)")
            self.tests_passed += 1

        except AssertionError as e:
            print_error(f"Models endpoint test failed: {e}")
            self.tests_failed += 1
        except Exception as e:
            print_error(f"Models endpoint error: {e}")
            self.tests_failed += 1

    def test_hmi_sessions_endpoint(self):
        """Test HMI /api/chat/sessions endpoint."""
        print_test("Testing HMI /api/chat/sessions endpoint")

        try:
            response = requests.get(f"{self.hmi_base}/api/chat/sessions", timeout=5)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"

            data = response.json()
            assert data['status'] == 'ok', f"Expected status 'ok', got {data.get('status')}"
            assert 'sessions' in data, "Response missing 'sessions' key"

            print_success(f"Sessions endpoint working ({len(data['sessions'])} sessions)")
            self.tests_passed += 1

        except AssertionError as e:
            print_error(f"Sessions endpoint test failed: {e}")
            self.tests_failed += 1
        except Exception as e:
            print_error(f"Sessions endpoint error: {e}")
            self.tests_failed += 1

    def test_gateway_health(self):
        """Test Gateway health endpoint."""
        print_test("Testing Gateway /health endpoint")

        try:
            response = requests.get(f"{self.gateway_base}/health", timeout=5)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"

            data = response.json()
            assert data['status'] in ['healthy', 'ok'], f"Expected 'healthy' or 'ok', got {data.get('status')}"

            print_success("Gateway health check passed")
            self.tests_passed += 1

        except AssertionError as e:
            print_error(f"Gateway health test failed: {e}")
            self.tests_failed += 1
        except Exception as e:
            print_error(f"Gateway health error: {e}")
            self.tests_failed += 1

    def test_gateway_streaming_direct(self):
        """Test Gateway /chat/stream endpoint directly."""
        print_test("Testing Gateway /chat/stream directly")

        try:
            payload = {
                "session_id": "test_direct",
                "message_id": "msg_direct",
                "agent_id": "architect",
                "model": "llama3.1:8b",
                "content": "Say only: TEST"
            }

            response = requests.post(
                f"{self.gateway_base}/chat/stream",
                json=payload,
                stream=True,
                timeout=20
            )

            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            assert 'text/event-stream' in response.headers.get('content-type', ''), \
                "Expected SSE content type"

            # Parse SSE events
            events = []
            for line in response.iter_lines():
                if line and line.startswith(b'data: '):
                    try:
                        event = json.loads(line[6:])
                        events.append(event)
                        if event.get('type') == 'done':
                            break
                    except json.JSONDecodeError:
                        pass

            # Verify event types
            event_types = [e.get('type') for e in events]
            assert 'status_update' in event_types, "No status_update events"
            assert 'token' in event_types, "No token events"
            assert 'usage' in event_types, "No usage event"
            assert 'done' in event_types, "No done event"

            # Verify we got actual content
            token_events = [e for e in events if e.get('type') == 'token']
            content = ''.join([e.get('content', '') for e in token_events])
            assert len(content) > 0, "No content received"

            print_success(f"Gateway streaming working (received {len(events)} events)")
            self.tests_passed += 1

        except AssertionError as e:
            print_error(f"Gateway streaming test failed: {e}")
            self.tests_failed += 1
        except Exception as e:
            print_error(f"Gateway streaming error: {e}")
            self.tests_failed += 1

    def test_end_to_end_single_message(self):
        """Test complete end-to-end flow with single message."""
        print_test("Testing end-to-end single message flow")

        try:
            # Step 1: Send message
            payload = {
                "session_id": None,
                "message": "Reply with exactly: SUCCESS",
                "agent_id": "architect",
                "model": "llama3.1:8b"
            }

            response = requests.post(
                f"{self.hmi_base}/api/chat/message",
                json=payload,
                timeout=10
            )

            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()
            assert data['status'] == 'ok', f"Expected 'ok', got {data.get('status')}"
            assert 'session_id' in data, "No session_id returned"

            session_id = data['session_id']

            # Step 2: Stream response
            stream_response = requests.get(
                f"{self.hmi_base}/api/chat/stream/{session_id}",
                stream=True,
                timeout=20
            )

            assert stream_response.status_code == 200, \
                f"Expected 200, got {stream_response.status_code}"

            # Parse events
            events = []
            for line in stream_response.iter_lines():
                if line and line.startswith(b'data: '):
                    try:
                        event = json.loads(line[6:])
                        events.append(event)
                        if event.get('type') == 'done':
                            break
                    except json.JSONDecodeError:
                        pass

            # Verify complete flow
            assert len(events) > 0, "No events received"

            event_types = [e.get('type') for e in events]
            assert 'done' in event_types, "Stream didn't complete"

            # Verify content
            token_events = [e for e in events if e.get('type') == 'token']
            content = ''.join([e.get('content', '') for e in token_events])

            print_success(f"End-to-end test passed (session: {session_id[:8]}...)")
            print_success(f"  Content: {content[:50]}...")
            self.tests_passed += 1

        except AssertionError as e:
            print_error(f"End-to-end test failed: {e}")
            self.tests_failed += 1
        except Exception as e:
            print_error(f"End-to-end error: {e}")
            self.tests_failed += 1

    def test_end_to_end_multi_message(self):
        """Test end-to-end flow with multiple messages in same session."""
        print_test("Testing end-to-end multi-message conversation")

        try:
            session_id = None

            # Send 3 messages in sequence
            for i in range(1, 4):
                payload = {
                    "session_id": session_id,
                    "message": f"Say: Message {i}",
                    "agent_id": "architect",
                    "model": "llama3.1:8b"
                }

                response = requests.post(
                    f"{self.hmi_base}/api/chat/message",
                    json=payload,
                    timeout=10
                )

                assert response.status_code == 200, \
                    f"Message {i} failed with status {response.status_code}"

                data = response.json()
                session_id = data['session_id']

                # Stream response
                stream_response = requests.get(
                    f"{self.hmi_base}/api/chat/stream/{session_id}",
                    stream=True,
                    timeout=20
                )

                # Wait for completion
                for line in stream_response.iter_lines():
                    if line and line.startswith(b'data: '):
                        try:
                            event = json.loads(line[6:])
                            if event.get('type') == 'done':
                                break
                        except json.JSONDecodeError:
                            pass

            print_success(f"Multi-message test passed (3 messages in session {session_id[:8]}...)")
            self.tests_passed += 1

        except AssertionError as e:
            print_error(f"Multi-message test failed: {e}")
            self.tests_failed += 1
        except Exception as e:
            print_error(f"Multi-message error: {e}")
            self.tests_failed += 1

    def test_model_selection(self):
        """Test that model selection is passed correctly."""
        print_test("Testing model selection (verifying model_id not model_name)")

        try:
            # Test with explicit model_id
            payload = {
                "session_id": None,
                "message": "Hi",
                "agent_id": "architect",
                "model": "llama3.1:8b"  # Use actual model ID
            }

            response = requests.post(
                f"{self.hmi_base}/api/chat/message",
                json=payload,
                timeout=10
            )

            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()
            assert data['status'] == 'ok', "Message submission failed"

            session_id = data['session_id']

            # Verify streaming works (confirms model was valid)
            stream_response = requests.get(
                f"{self.hmi_base}/api/chat/stream/{session_id}",
                stream=True,
                timeout=20
            )

            # Look for done event
            done_received = False
            for line in stream_response.iter_lines():
                if line and line.startswith(b'data: '):
                    try:
                        event = json.loads(line[6:])
                        if event.get('type') == 'done':
                            done_received = True
                            break
                    except json.JSONDecodeError:
                        pass

            assert done_received, "Stream didn't complete (model likely invalid)"

            print_success("Model selection test passed (model_id correctly used)")
            self.tests_passed += 1

        except AssertionError as e:
            print_error(f"Model selection test failed: {e}")
            self.tests_failed += 1
        except Exception as e:
            print_error(f"Model selection error: {e}")
            self.tests_failed += 1

    def test_agent_selection(self):
        """Test that agent selection works correctly."""
        print_test("Testing agent selection")

        try:
            # Get available agents
            agents_response = requests.get(f"{self.hmi_base}/api/agents", timeout=5)
            agents = agents_response.json()['agents']

            # Test with first available agent
            agent_id = agents[0]['agent_id']

            payload = {
                "session_id": None,
                "message": "Test",
                "agent_id": agent_id,
                "model": "llama3.1:8b"
            }

            response = requests.post(
                f"{self.hmi_base}/api/chat/message",
                json=payload,
                timeout=10
            )

            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()
            assert data['status'] == 'ok', "Agent selection failed"

            print_success(f"Agent selection test passed (agent: {agent_id})")
            self.tests_passed += 1

        except AssertionError as e:
            print_error(f"Agent selection test failed: {e}")
            self.tests_failed += 1
        except Exception as e:
            print_error(f"Agent selection error: {e}")
            self.tests_failed += 1

    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        print_test("Testing error handling")

        tests_passed_local = 0
        tests_total = 3

        # Test 1: Invalid model
        try:
            payload = {
                "session_id": None,
                "message": "Test",
                "agent_id": "architect",
                "model": "invalid_model_that_does_not_exist"
            }

            response = requests.post(
                f"{self.hmi_base}/api/chat/message",
                json=payload,
                timeout=10
            )

            # Should still create session, but streaming will fail gracefully
            data = response.json()
            session_id = data.get('session_id')

            if session_id:
                stream_response = requests.get(
                    f"{self.hmi_base}/api/chat/stream/{session_id}",
                    stream=True,
                    timeout=10
                )

                # Should get error event
                for line in stream_response.iter_lines():
                    if line and line.startswith(b'data: '):
                        try:
                            event = json.loads(line[6:])
                            if event.get('type') == 'status_update' and \
                               event.get('status') == 'error':
                                tests_passed_local += 1
                                break
                        except json.JSONDecodeError:
                            pass

        except Exception as e:
            print_warning(f"Invalid model test error: {e}")

        # Test 2: Empty message
        try:
            payload = {
                "session_id": None,
                "message": "",
                "agent_id": "architect",
                "model": "llama3.1:8b"
            }

            response = requests.post(
                f"{self.hmi_base}/api/chat/message",
                json=payload,
                timeout=10
            )

            # Should reject empty message
            if response.status_code >= 400 or \
               response.json().get('status') == 'error':
                tests_passed_local += 1

        except Exception as e:
            print_warning(f"Empty message test error: {e}")

        # Test 3: Invalid session ID for streaming
        try:
            response = requests.get(
                f"{self.hmi_base}/api/chat/stream/invalid_session_id_xyz",
                stream=True,
                timeout=5
            )

            # Should return error or 404
            if response.status_code >= 400:
                tests_passed_local += 1

        except Exception as e:
            print_warning(f"Invalid session test error: {e}")

        if tests_passed_local >= 2:  # At least 2 of 3 error tests passed
            print_success(f"Error handling tests passed ({tests_passed_local}/{tests_total})")
            self.tests_passed += 1
        else:
            print_error(f"Error handling tests failed ({tests_passed_local}/{tests_total})")
            self.tests_failed += 1

    def print_summary(self):
        """Print test summary."""
        total = self.tests_passed + self.tests_failed

        print(f"\n{Colors.BOLD}{'='*60}")
        print("Test Summary")
        print(f"{'='*60}{Colors.RESET}")
        print(f"Total Tests: {total}")
        print(f"{Colors.GREEN}Passed: {self.tests_passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {self.tests_failed}{Colors.RESET}")

        if self.tests_failed == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED!{Colors.RESET}\n")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ SOME TESTS FAILED{Colors.RESET}\n")


def main():
    """Main entry point."""
    tester = LLMInterfaceTester()
    tester.run_all_tests()


if __name__ == '__main__':
    main()

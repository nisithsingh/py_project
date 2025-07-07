"""
Simplified unit tests for main.py CPACConsistencyService
Tests only the core functions with minimal mocking
"""
import os
import sys
import unittest
import asyncio
import json
import tempfile
import traceback
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from main import CPACConsistencyService
    from ai_foundation.protocol.luma_protocol.models import (
        LumaMessage, MessageType, Source, Target, Payload, Task, Status
    )
    from data_model.schemas import ClaimCategory
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    IMPORTS_AVAILABLE = False


@unittest.skipIf(not IMPORTS_AVAILABLE, "Required imports not available")
class TestCPACConsistencyServiceCore(unittest.TestCase):
    """Test core CPACConsistencyService functions with minimal mocking"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.service = CPACConsistencyService()
        self.service.agent_name = "test_agent"
        
        # Create sample LUMA message for testing
        self.sample_message = LumaMessage(
            message_type=MessageType.REQEST,
            source=Source(
                created_ts=datetime.now().isoformat() + 'Z',
                name="orchestrator"
            ),
            target=Target(name="cpac_consistency_reviewer"),
            payload=Payload(
                thread_id="test_thread",
                job_id="test_job", 
                task_id="test_task",
                conversation_id=12345,
                task=Task(
                    request={
                        "claim_category": "employment",
                        "claims": [
                            {
                                "claim_id": "1",
                                "claim_type": "original_cpac",
                                "cpac_data": {
                                    "employer_name": "Test Company",
                                    "job_title": "Engineer",
                                    "start_date": "2020-01-01",
                                    "end_date": "2023-12-31",
                                    "annual_compensation": "USD 100000"
                                }
                            }
                        ],
                        "cpac_text": "Test CPAC text"
                    }
                )
            ),
            conversation_history="",
            shared_resources={}
        )
    
    async def test_handle_message_success(self):
        """Test _handle_message with successful processing"""
        # Mock only the graph and protocol
        mock_graph = AsyncMock()
        mock_protocol = AsyncMock()
        
        # Set up successful graph response
        mock_result = {
            "claim_category": "employment",
            "discrepancies": [],
            "review_metadata": {"final_decision": True}
        }
        mock_graph.process_request.return_value = mock_result
        
        # Inject mocks into service
        self.service.graph = mock_graph
        self.service.protocol = mock_protocol
        
        # Test the function
        await self.service._handle_message(self.sample_message)
        
        # Verify behavior
        mock_graph.process_request.assert_called_once()
        mock_protocol.send_result.assert_called_once()
        
        # Verify the LUMA message structure sent
        sent_message = mock_protocol.send_result.call_args[0][0]
        self.assertEqual(sent_message.message_type, MessageType.RESULT)
        self.assertEqual(sent_message.payload.task.status, Status.COMPLETED)
    
    async def test_handle_message_validation_error(self):
        """Test _handle_message with invalid message type"""
        # Create message with wrong type
        invalid_message = LumaMessage(
            message_type=MessageType.RESULT,  # Wrong type
            source=Source(
                created_ts=datetime.now().isoformat() + 'Z',
                name="orchestrator"
            ),
            target=Target(name="cpac_consistency_reviewer"),
            payload=Payload(
                thread_id="test_thread",
                job_id="test_job",
                task_id="test_task",
                conversation_id=12345
            ),
            conversation_history="",
            shared_resources={}
        )
        
        # Mock protocol
        mock_protocol = AsyncMock()
        self.service.protocol = mock_protocol
        
        # Test the function
        await self.service._handle_message(invalid_message)
        
        # Verify error result was sent
        mock_protocol.send_result.assert_called_once()
        sent_message = mock_protocol.send_result.call_args[0][0]
        self.assertEqual(sent_message.payload.task.status, Status.FAILED)
        self.assertIn("Invalid message type", sent_message.payload.task.error["reason"])
    
    async def test_handle_message_graph_error(self):
        """Test _handle_message with graph processing error"""
        # Mock graph to raise exception
        mock_graph = AsyncMock()
        mock_graph.process_request.side_effect = RuntimeError("Graph processing failed")
        mock_protocol = AsyncMock()
        
        self.service.graph = mock_graph
        self.service.protocol = mock_protocol
        
        # Test the function
        await self.service._handle_message(self.sample_message)
        
        # Verify error handling
        mock_protocol.send_result.assert_called_once()
        sent_message = mock_protocol.send_result.call_args[0][0]
        self.assertEqual(sent_message.payload.task.status, Status.FAILED)
        self.assertIn("Graph processing failed", sent_message.payload.task.error["reason"])
    
    async def test_send_result(self):
        """Test _send_result function"""
        # Mock protocol
        mock_protocol = AsyncMock()
        self.service.protocol = mock_protocol
        
        # Test data
        test_result = {
            "claim_category": "employment",
            "discrepancies": [],
            "review_metadata": {"final_decision": True}
        }
        
        # Call function
        await self.service._send_result(self.sample_message, test_result)
        
        # Verify protocol was called
        mock_protocol.send_result.assert_called_once()
        
        # Verify message structure
        sent_message = mock_protocol.send_result.call_args[0][0]
        self.assertEqual(sent_message.message_type, MessageType.RESULT)
        self.assertEqual(sent_message.source.name, "test_agent")
        self.assertEqual(sent_message.target.name, "orchestrator")
        self.assertEqual(sent_message.payload.task.result, test_result)
        self.assertEqual(sent_message.payload.task.status, Status.COMPLETED)
        
        # Verify payload fields are preserved
        self.assertEqual(sent_message.payload.thread_id, "test_thread")
        self.assertEqual(sent_message.payload.job_id, "test_job")
        self.assertEqual(sent_message.payload.task_id, "test_task")
        self.assertEqual(sent_message.payload.conversation_id, 12345)
    
    async def test_send_error_result(self):
        """Test _send_error_result function"""
        # Mock protocol
        mock_protocol = AsyncMock()
        self.service.protocol = mock_protocol
        
        # Test error details
        error_details = {
            "type": "technical error",
            "reason": "Test error with location details | Task: test_task"
        }
        
        # Call function
        await self.service._send_error_result(self.sample_message, error_details)
        
        # Verify protocol was called
        mock_protocol.send_result.assert_called_once()
        
        # Verify message structure
        sent_message = mock_protocol.send_result.call_args[0][0]
        self.assertEqual(sent_message.message_type, MessageType.RESULT)
        self.assertEqual(sent_message.source.name, "test_agent")
        self.assertEqual(sent_message.target.name, "orchestrator")
        self.assertEqual(sent_message.payload.task.status, Status.FAILED)
        
        # Verify error format (LUMA error format in task.error)
        task_error = sent_message.payload.task.error
        self.assertEqual(task_error["type"], "technical error")
        self.assertEqual(task_error["reason"], "Test error with location details | Task: test_task")
        
        # Verify payload fields are preserved
        self.assertEqual(sent_message.payload.thread_id, "test_thread")
        self.assertEqual(sent_message.payload.job_id, "test_job")
        self.assertEqual(sent_message.payload.task_id, "test_task")
    
    def test_format_error_reason_basic(self):
        """Test _format_error_reason with basic exception"""
        # Create test exception
        try:
            raise ValueError("Test error message")
        except Exception as e:
            result = self.service._format_error_reason(e, "test_task_123")
        
        # Verify format
        self.assertIn("ValueError: Test error message", result)
        self.assertIn("Task: test_task_123", result)
    
    def test_format_error_reason_with_traceback(self):
        """Test _format_error_reason with traceback location"""
        # Create exception with traceback in our codebase
        def fake_agent_function():
            raise RuntimeError("Agent processing failed")
        
        try:
            fake_agent_function()
        except Exception as e:
            # Mock the traceback to simulate coming from agent module
            with patch('traceback.extract_tb') as mock_extract:
                # Mock traceback frame that looks like it's from agent module
                mock_frame = Mock()
                mock_frame.filename = "/path/to/agent/executor_node.py"
                mock_frame.name = "analyze_claims"
                mock_frame.lineno = 123
                mock_extract.return_value = [mock_frame]
                
                result = self.service._format_error_reason(e, "test_task_456")
        
        # Verify format includes location
        self.assertIn("RuntimeError: Agent processing failed", result)
        self.assertIn("Location: analyze_claims() in executor_node.py:123", result)
        self.assertIn("Task: test_task_456", result)
    
    def test_format_error_reason_multiple_frames(self):
        """Test _format_error_reason with multiple traceback frames"""
        try:
            raise ConnectionError("Network failure")
        except Exception as e:
            # Mock traceback with multiple frames, should pick the agent one
            with patch('traceback.extract_tb') as mock_extract:
                # Create frames - should pick the agent one
                frame1 = Mock()
                frame1.filename = "/system/lib/network.py"
                frame1.name = "connect"
                frame1.lineno = 50
                
                frame2 = Mock()
                frame2.filename = "/project/graph/cpac_graph.py"
                frame2.name = "process_request"
                frame2.lineno = 200
                
                frame3 = Mock()
                frame3.filename = "/system/asyncio/events.py"
                frame3.name = "run"
                frame3.lineno = 80
                
                mock_extract.return_value = [frame1, frame2, frame3]
                
                result = self.service._format_error_reason(e, "network_task")
        
        # Should pick the graph frame (our codebase)
        self.assertIn("ConnectionError: Network failure", result)
        self.assertIn("Location: process_request() in cpac_graph.py:200", result)
        self.assertIn("Task: network_task", result)


def run_async_test(test_method):
    """Helper to run async test methods"""
    def wrapper(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(test_method(self))
        finally:
            loop.close()
    return wrapper


if __name__ == '__main__':
    # Convert async test methods
    TestCPACConsistencyServiceCore.test_handle_message_success = run_async_test(
        TestCPACConsistencyServiceCore.test_handle_message_success
    )
    TestCPACConsistencyServiceCore.test_handle_message_validation_error = run_async_test(
        TestCPACConsistencyServiceCore.test_handle_message_validation_error
    )
    TestCPACConsistencyServiceCore.test_handle_message_graph_error = run_async_test(
        TestCPACConsistencyServiceCore.test_handle_message_graph_error
    )
    TestCPACConsistencyServiceCore.test_send_result = run_async_test(
        TestCPACConsistencyServiceCore.test_send_result
    )
    TestCPACConsistencyServiceCore.test_send_error_result = run_async_test(
        TestCPACConsistencyServiceCore.test_send_error_result
    )
    
    unittest.main(verbosity=2)
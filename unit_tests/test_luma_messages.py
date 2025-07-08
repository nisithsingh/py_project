"""
Simplified LUMA message handling tests for CPACConsistencyService
Tests three key scenarios using actual protocol and graph with proper setup/teardown
"""
import os
import sys
import unittest
import asyncio
import tempfile
import shutil
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from main import CPACConsistencyService
    from ai_foundation.protocol.luma_protocol.models import (
        LumaMessage, MessageType, Source, Target, Payload, Task, Status
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    IMPORTS_AVAILABLE = False


@unittest.skipIf(not IMPORTS_AVAILABLE, "Required imports not available")
class TestLumaMessageHandling(unittest.TestCase):
    """Test LUMA message handling with actual protocol and graph"""
    
    def setUp(self):
        """Set up test environment with required resources"""
        # Create temporary directory for test resources
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        
        # Set environment variables for testing
        os.environ['ENVIRONMENT'] = 'local'
        os.environ['SERVICE_BUS_NAMESPACE'] = os.path.join(self.test_dir, 'service_bus')
        os.environ['RESULT_QUEUE'] = 'test_orchestrator'
        os.environ['BLOB_STORAGE_ACCOUNT'] = 'test_storage'
        os.environ['BLOB_STORAGE_CONTAINER'] = os.path.join(self.test_dir, 'blob_storage')
        
        # Mock OpenAI API key if not set
        if 'OPENAI_API_KEY' not in os.environ:
            os.environ['OPENAI_API_KEY'] = 'test-key-for-testing'
        
        # Change to test directory and copy required files
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Sample valid LUMA message
        self.valid_message = LumaMessage(
            message_type=MessageType.REQEST,
            source=Source(
                created_ts=datetime.now().isoformat() + 'Z',
                name="orchestrator"
            ),
            target=Target(name="cpac_consistency_reviewer"),
            payload=Payload(
                thread_id="test_thread",
                job_id=12345,
                task_id=67890,
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
        
        # Invalid message (wrong type)
        self.invalid_message = LumaMessage(
            message_type=MessageType.RESULT,  # Wrong type
            source=Source(
                created_ts=datetime.now().isoformat() + 'Z',
                name="orchestrator"
            ),
            target=Target(name="cpac_consistency_reviewer"),
            payload=Payload(
                thread_id="test_thread",
                job_id=12345,
                task_id=67890,
                conversation_id=12345
            ),
            conversation_history="",
            shared_resources={}
        )
        
        # Message with missing required fields
        self.missing_fields_message = LumaMessage(
            message_type=MessageType.REQEST,
            source=Source(
                created_ts=datetime.now().isoformat() + 'Z',
                name="orchestrator"
            ),
            target=Target(name="cpac_consistency_reviewer"),
            payload=Payload(
                thread_id="test_thread",
                job_id=12345,
                task_id=67890,
                conversation_id=12345,
                task=Task(
                    request={
                        # Missing claim_category and claims
                        "cpac_text": "Test CPAC text"
                    }
                )
            ),
            conversation_history="",
            shared_resources={}
        )
    
    def tearDown(self):
        """Clean up test resources"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
        # Clean up environment variables
        for key in ['ENVIRONMENT', 'SERVICE_BUS_NAMESPACE', 'RESULT_QUEUE', 
                   'BLOB_STORAGE_ACCOUNT', 'BLOB_STORAGE_CONTAINER']:
            os.environ.pop(key, None)
    
    def test_valid_request_success(self):
        """Test case 1: Valid request processed successfully"""
        async def run_test():
            # Create service instance
            service = CPACConsistencyService()
            
            # Manual initialization without blocking listener
            service.agent_name = "cpac_consistency_reviewer"
            
            # Override the graph to prevent actual LLM calls
            class TestableGraph:
                def __init__(self, *args, **kwargs):
                    pass
                
                async def process_request(self, luma_message, request_data):
                    # Return a valid response structure
                    return {
                        "claim_category": request_data.claim_category,
                        "discrepancies": [],
                        "review_metadata": {"final_decision": True}
                    }
            
            # Create test protocol that captures messages
            class TestProtocol:
                def __init__(self):
                    self.sent_messages = []
                
                async def send_result(self, message):
                    self.sent_messages.append(message)
            
            # Set up service components
            service.graph = TestableGraph()
            service.protocol = TestProtocol()
            
            # Process the message
            await service._handle_message(self.valid_message)
            
            # Assert response structure
            self.assertEqual(len(service.protocol.sent_messages), 1)
            sent_message = service.protocol.sent_messages[0]
            
            self.assertEqual(sent_message.message_type, MessageType.RESULT)
            self.assertEqual(sent_message.payload.task.status, Status.COMPLETED)
            self.assertEqual(sent_message.source.name, "cpac_consistency_reviewer")
            self.assertEqual(sent_message.target.name, "orchestrator")
            self.assertIsNotNone(sent_message.payload.task.result)
            self.assertEqual(sent_message.payload.task.result["claim_category"], "employment")
            
            # Pretty print the response for debugging/verification
            return sent_message
        
        result = asyncio.run(run_test())
        
        print("\n" + "="*80)
        print("TEST 1: VALID REQUEST SUCCESS - FULL LUMA RESPONSE")
        print("="*80)
        print("Complete LUMA Message Response:")
        print(json.dumps(result.model_dump(), indent=2, default=str))
        print("="*80 + "\n")
    
    def test_invalid_request_error(self):
        """Test case 2: Invalid request returns error without graph processing"""
        async def run_test():
            # Create service instance
            service = CPACConsistencyService()
            
            # Manual initialization without blocking listener
            service.agent_name = "cpac_consistency_reviewer"
            
            # Create test protocol that captures messages
            class TestProtocol:
                def __init__(self):
                    self.sent_messages = []
                
                async def send_result(self, message):
                    self.sent_messages.append(message)
            
            # Set up service components
            service.protocol = TestProtocol()
            
            # Process invalid message
            await service._handle_message(self.invalid_message)
            
            # Assert error response structure
            self.assertEqual(len(service.protocol.sent_messages), 1)
            sent_message = service.protocol.sent_messages[0]
            
            self.assertEqual(sent_message.message_type, MessageType.RESULT)
            self.assertEqual(sent_message.payload.task.status, Status.FAILED)
            self.assertIn("Invalid message type", sent_message.payload.task.error["reason"])
            self.assertEqual(sent_message.payload.task.error["type"], "technical error")
            
            # Pretty print the response for debugging/verification
            return sent_message
        
        result = asyncio.run(run_test())
        
        print("\n" + "="*80)
        print("TEST 2: INVALID REQUEST ERROR - FULL LUMA RESPONSE")
        print("="*80)
        print("Complete LUMA Message Response:")
        print(json.dumps(result.model_dump(), indent=2, default=str))
        print("="*80 + "\n")
    
    def test_valid_request_graph_exception(self):
        """Test case 3: Valid request but graph processing raises exception"""
        async def run_test():
            # Create service instance
            service = CPACConsistencyService()
            
            # Manual initialization without blocking listener
            service.agent_name = "cpac_consistency_reviewer"
            
            # Override the graph to raise exception
            class FailingGraph:
                def __init__(self, *args, **kwargs):
                    pass
                
                async def process_request(self, luma_message, request_data):
                    raise RuntimeError("Graph processing failed")
            
            # Create test protocol that captures messages
            class TestProtocol:
                def __init__(self):
                    self.sent_messages = []
                
                async def send_result(self, message):
                    self.sent_messages.append(message)
            
            # Set up service components
            service.graph = FailingGraph()
            service.protocol = TestProtocol()
            
            # Process the message
            await service._handle_message(self.valid_message)
            
            # Assert error response structure
            self.assertEqual(len(service.protocol.sent_messages), 1)
            sent_message = service.protocol.sent_messages[0]
            
            self.assertEqual(sent_message.message_type, MessageType.RESULT)
            self.assertEqual(sent_message.payload.task.status, Status.FAILED)
            self.assertIn("Graph processing failed", sent_message.payload.task.error["reason"])
            self.assertEqual(sent_message.payload.task.error["type"], "technical error")
            
            # Pretty print the response for debugging/verification
            return sent_message
        
        result = asyncio.run(run_test())
        
        print("\n" + "="*80)
        print("TEST 3: VALID REQUEST GRAPH EXCEPTION - FULL LUMA RESPONSE")
        print("="*80)
        print("Complete LUMA Message Response:")
        print(json.dumps(result.model_dump(), indent=2, default=str))
        print("="*80 + "\n")
    
    def test_missing_required_fields_error(self):
        """Test case 4: Valid message type but missing required fields"""
        async def run_test():
            # Create service instance
            service = CPACConsistencyService()
            
            # Manual initialization without blocking listener
            service.agent_name = "cpac_consistency_reviewer"
            
            # Create test protocol that captures messages
            class TestProtocol:
                def __init__(self):
                    self.sent_messages = []
                
                async def send_result(self, message):
                    self.sent_messages.append(message)
            
            # Set up service components
            service.protocol = TestProtocol()
            
            # Process message with missing fields
            await service._handle_message(self.missing_fields_message)
            
            # Assert error response structure
            self.assertEqual(len(service.protocol.sent_messages), 1)
            sent_message = service.protocol.sent_messages[0]
            
            self.assertEqual(sent_message.message_type, MessageType.RESULT)
            self.assertEqual(sent_message.payload.task.status, Status.FAILED)
            self.assertIn("Missing required field", sent_message.payload.task.error["reason"])
            self.assertEqual(sent_message.payload.task.error["type"], "technical error")
            
            # Pretty print the response for debugging/verification
            return sent_message
        
        result = asyncio.run(run_test())
        
        print("\n" + "="*80)
        print("TEST 4: MISSING REQUIRED FIELDS ERROR - FULL LUMA RESPONSE")
        print("="*80)
        print("Complete LUMA Message Response:")
        print(json.dumps(result.model_dump(), indent=2, default=str))
        print("="*80 + "\n")


if __name__ == '__main__':
    unittest.main(verbosity=2)
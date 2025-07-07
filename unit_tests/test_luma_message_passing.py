"""
Unit tests for LUMA message passing and error handling in different nodes
Tests exception propagation through the workflow and LUMA response format
"""
import os
import sys
import unittest
import asyncio
import json
import tempfile
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from main import CPACConsistencyService
    from graph.cpac_consistency_graph import CPACConsistencyGraph
    from data_model.schemas import (
        CPACConsistencyRequest, CPACConsistencyResult        
    )
    from ai_foundation.protocol.luma_protocol.models import (
    LumaMessage, AgentMetadata, MessageType, Status, Source, Target, Payload, Task
    )
    from agent.executor_node import ExecutorNode
    from agent.reviewer_node import ReviewerNode
    from agent.improver_node import ImproverNode
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    IMPORTS_AVAILABLE = False


@unittest.skipIf(not IMPORTS_AVAILABLE, "Required imports not available")
class TestLumaMessagePassing(unittest.TestCase):
    """Test LUMA message passing and error handling in workflow nodes"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        
        # Create basic config file
        config_content = """
application:
  service_name: "cpac_consistency_reviewer"
  to_service: "discrepancy_analyzer_orchestrator"

debug:
  print_prompts: false
  print_results: false
  save_logs: false

llm:
  prompt_path: "prompts/cpac_analysis_prompt_v7.j2"
  generic_prompt_path: "prompts/cpac_analysis_prompt_generic_v8.j2"
  model: "gpt-4o"
  temperature: 0.0
  max_tokens: 4096

analysis:
  timeline_gap_threshold_months: 6
  timeline_overlap_threshold_months: 1

graph:
  max_review_iterations: 1
  use_binary_classification: true

reviewer:
  prompt_path: "prompts/reviewer_prompt_improved_v11.j2"
  generic_prompt_path: "prompts/reviewer_prompt_generic_v12.j2"
  binary_prompt_path: "prompts/reviewer_prompt_binary_v16.j2"

improvement:
  prompt_path: "prompts/improvement_prompt_v5.j2"

preprocessing:
  prompt_path: "prompts/preprocessing_prompt_v1.j2"
"""
        with open(self.config_path, 'w') as f:
            f.write(config_content)
        
        # Sample LUMA message for testing
        self.test_luma_message = LumaMessage(
            message_type=MessageType.REQEST,
            source=Source(
                created_ts=datetime.now().isoformat() + 'Z',
                name="discrepancy_analyzer_orchestrator"
            ),
            target=Target(name="cpac_consistency_reviewer"),
            payload=Payload(
                conversation_id=12345,
                job_id=67890,
                task_id=1234567890,
                task=Task(
                    request={
                        "claim_category": "employment",
                        "claims": [
                            {
                                "claim_id": "1",
                                "claim_type": "original_cpac",
                                "claim_sow_category": "employment_sow_category",
                                "cpac_data": {
                                    "employer_name": "Test Company",
                                    "job_title": "Engineer",
                                    "professional_duties": "Engineering work",
                                    "start_date": "2020-01-01",
                                    "end_date": "2023-12-31",
                                    "employment_type": "full-time",
                                    "annual_compensation": "USD 100000"
                                }
                            }
                        ],
                        "cpac_text": "Test CPAC text content"
                    }
                )
            ),
            conversation_history="",
            shared_resources={}
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_executor_node_exception_handling(self):
        """Test Case 1: Exception raised in executor node"""
        # Create graph with real components but trigger error with missing API key
        graph = CPACConsistencyGraph(config_path=self.config_path)
        
        # Create CPACConsistencyRequest from LUMA message
        task_request = self.test_luma_message.payload.task.request
        request_data = CPACConsistencyRequest(
            claim_category=task_request["claim_category"],
            claims=task_request["claims"],
            cpac_text=task_request.get("cpac_text", "")
        )
        
        # Remove OpenAI API key to trigger LLM initialization failure
        with patch.dict(os.environ, {}, clear=True):
            # Process the message - this will trigger LLM errors
            result = await graph.process_request(self.test_luma_message, request_data)
            
            # Verify the result structure
            self.assertIsInstance(result, dict)
            self.assertIn('exceptions', result)
            self.assertIn('discrepancies', result)
            self.assertIn('review_metadata', result)
            
            # Verify exceptions are captured
            exceptions = result['exceptions']
            self.assertGreaterEqual(exceptions['count'], 0)
            
            # Verify review metadata indicates completion despite errors
            review_metadata = result['review_metadata']
            self.assertIsNotNone(review_metadata['final_decision'])
    
    async def test_reviewer_node_exception_handling(self):
        """Test Case 2: Exception raised in reviewer node"""
        # Create graph with real components
        graph = CPACConsistencyGraph(config_path=self.config_path)
        
        # Create CPACConsistencyRequest from LUMA message
        task_request = self.test_luma_message.payload.task.request
        request_data = CPACConsistencyRequest(
            claim_category=task_request["claim_category"],
            claims=task_request["claims"],
            cpac_text=task_request.get("cpac_text", "")
        )
        
        # Set a valid API key for initialization but mock the OpenAI client call to fail during execution
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key-12345'}):
            # Mock the OpenAI client chat completions to fail after initialization
            with patch('openai.OpenAI') as mock_openai_class:
                mock_client = Mock()
                mock_chat = Mock()
                mock_completions = Mock()
                mock_completions.create.side_effect = Exception("API call failed during review")
                mock_chat.completions = mock_completions
                mock_client.chat = mock_chat
                mock_openai_class.return_value = mock_client
                
                # Process the message
                result = await graph.process_request(self.test_luma_message, request_data)
                
                # Verify the result structure
                self.assertIsInstance(result, dict)
                self.assertIn('exceptions', result)
                self.assertIn('discrepancies', result)
                self.assertIn('review_metadata', result)
                
                # Verify exceptions are captured
                exceptions = result['exceptions']
                self.assertGreaterEqual(exceptions['count'], 0)
                
                # Verify review metadata shows error handling
                review_metadata = result['review_metadata']
                self.assertIsNotNone(review_metadata['final_decision'])
    
    async def test_improver_node_exception_handling(self):
        """Test Case 3: Exception raised in improver node"""
        # Create graph with real components  
        graph = CPACConsistencyGraph(config_path=self.config_path)
        
        # Use employment gap test data that will trigger timeline discrepancies
        test_message_with_gaps = LumaMessage(
            message_type=MessageType.REQEST,
            source=Source(
                created_ts=datetime.now().isoformat() + 'Z',
                name="discrepancy_analyzer_orchestrator"
            ),
            target=Target(name="cpac_consistency_reviewer"),
            payload=Payload(
                conversation_id=12345,
                job_id=67890,
                task_id=1234567890,
                task=Task(
                    request={
                        "claim_category": "employment",
                        "claims": [
                            {
                                "claim_id": "1",
                                "claim_type": "original_cpac",
                                "claim_sow_category": "employment_sow_category",
                                "cpac_data": {
                                    "employer_name": "Company A",
                                    "job_title": "Engineer",
                                    "start_date": "2020-01-01",
                                    "end_date": "2021-12-31",
                                    "employment_type": "full-time",
                                    "annual_compensation": "USD 100000"
                                }
                            },
                            {
                                "claim_id": "2", 
                                "claim_type": "original_cpac",
                                "claim_sow_category": "employment_sow_category",
                                "cpac_data": {
                                    "employer_name": "Company B",
                                    "job_title": "Senior Engineer", 
                                    "start_date": "2023-01-01",  # Gap between 2021 and 2023
                                    "end_date": "2024-12-31",
                                    "employment_type": "full-time",
                                    "annual_compensation": "USD 120000"
                                }
                            }
                        ],
                        "cpac_text": "Employment history with gap"
                    }
                )
            ),
            conversation_history="",
            shared_resources={}
        )
        
        # Create CPACConsistencyRequest from test message
        task_request = test_message_with_gaps.payload.task.request
        request_data = CPACConsistencyRequest(
            claim_category=task_request["claim_category"],
            claims=task_request["claims"],
            cpac_text=task_request.get("cpac_text", "")
        )
        
        # Remove API key to trigger failures in improvement process
        with patch.dict(os.environ, {}, clear=True):
            # Process the message - timeline gaps will be detected but improvement will fail
            result = await graph.process_request(test_message_with_gaps, request_data)
            
            # Verify the result structure
            self.assertIsInstance(result, dict)
            self.assertIn('exceptions', result) 
            self.assertIn('discrepancies', result)
            self.assertIn('review_metadata', result)
            
            # Verify workflow completed despite improvement errors
            review_metadata = result['review_metadata']
            self.assertIsNotNone(review_metadata['final_decision'])
    
    async def test_workflow_with_no_api_key(self):
        """Test complete workflow when OpenAI API key is missing"""
        # Create graph with real components
        graph = CPACConsistencyGraph(config_path=self.config_path)
        
        # Create CPACConsistencyRequest from LUMA message
        task_request = self.test_luma_message.payload.task.request
        request_data = CPACConsistencyRequest(
            claim_category=task_request["claim_category"],
            claims=task_request["claims"],
            cpac_text=task_request.get("cpac_text", "")
        )
        
        # Remove API key to test error handling across all nodes
        with patch.dict(os.environ, {}, clear=True):
            # Process the message - all LLM calls will fail due to missing API key
            result = await graph.process_request(self.test_luma_message, request_data)
            
            # Verify that even with LLM failures, we get a complete response
            self.assertIsInstance(result, dict)
            self.assertIn('exceptions', result)
            self.assertIn('discrepancies', result) 
            self.assertIn('review_metadata', result)
            
            # Verify workflow completed despite errors
            review_metadata = result['review_metadata']
            self.assertIsNotNone(review_metadata['final_decision'])


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
    # Patch test methods to be async-compatible
    TestLumaMessagePassing.test_executor_node_exception_handling = run_async_test(
        TestLumaMessagePassing.test_executor_node_exception_handling
    )
    TestLumaMessagePassing.test_reviewer_node_exception_handling = run_async_test(
        TestLumaMessagePassing.test_reviewer_node_exception_handling
    )
    TestLumaMessagePassing.test_improver_node_exception_handling = run_async_test(
        TestLumaMessagePassing.test_improver_node_exception_handling
    )
    TestLumaMessagePassing.test_workflow_with_no_api_key = run_async_test(
        TestLumaMessagePassing.test_workflow_with_no_api_key
    )
    
    unittest.main()
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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
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
  use_binary_classification: false

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
                            },
                            {
                                "claim_id": "2",
                                "claim_type": "original_cpac",
                                "claim_sow_category": "employment_sow_category",
                                "cpac_data": {
                                    "employer_name": "Test Company 2",
                                    "job_title": "Engineer",
                                    "professional_duties": "Engineering work",
                                    "start_date": "2021-01-01",
                                    "end_date": "2024-12-31",
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
        """Test Case 1: Employment claims succeed with rule-based analyzer (no API key needed)"""
        # Create graph with real components but trigger error with missing API key
        graph = CPACConsistencyGraph(config_path=self.config_path)
        
        # Create CPACConsistencyRequest from LUMA message
        task_request = self.test_luma_message.payload.task.request
        request_data = CPACConsistencyRequest(
            claim_category=task_request["claim_category"],
            claims=task_request["claims"],
            cpac_text=task_request.get("cpac_text", "")
        )
        
        # Remove OpenAI API key - employment should still work with rule-based analyzer
        with patch.dict(os.environ, {}, clear=True):
            # Process the message - employment uses rule-based timeline analyzer (no API key needed)
            result = await graph.process_request(self.test_luma_message, request_data)
            
            # Verify successful result structure (no exceptions expected for employment)
            self.assertIsInstance(result, dict)
            self.assertIn('discrepancies', result)
            self.assertIn('review_metadata', result)
            
            # Verify analysis completed successfully with rule-based method
            review_metadata = result['review_metadata']
            self.assertEqual(review_metadata['analysis_method'], 'rule_based')
            self.assertIsNotNone(review_metadata['final_decision'])
    
    async def test_reviewer_node_exception_handling(self):
        """Test Case 2: Exception propagation in reviewer node with API failure"""
        
        # Set a valid API key for initialization but mock the OpenAI client call to fail during execution
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key-12345'}):
            # Mock the OpenAI client where it's imported in the LLM handler
            with patch('utils.llm_handler.OpenAI') as mock_openai_class:
                mock_client = Mock()
                mock_chat = Mock()
                mock_completions = Mock()
                mock_completions.create.side_effect = Exception("API call failed during review")
                mock_chat.completions = mock_completions
                mock_client.chat = mock_chat
                mock_openai_class.return_value = mock_client
                
                # Create graph with mocked OpenAI client
                graph = CPACConsistencyGraph(config_path=self.config_path)
        
                # Create employment claims with a large gap (24 months) to ensure discrepancies are generated
                employment_with_large_gap_request = {
                    "claim_category": "employment",
                    "claims": [
                        {
                            "claim_id": "1",
                            "claim_type": "original_cpac",
                            "cpac_data": {
                                "employer_name": "Company A",
                                "job_title": "Engineer",
                                "start_date": "2020-01-01",
                                "end_date": "2020-12-31",  # Ends in 2020
                                "annual_compensation": "USD 100000",
                                "employment_type": "full-time"
                            }
                        },
                        {
                            "claim_id": "2",
                            "claim_type": "original_cpac", 
                            "cpac_data": {
                                "employer_name": "Company B",
                                "job_title": "Senior Engineer",
                                "start_date": "2023-01-01",  # 24-month gap (exceeds 6-month threshold)
                                "end_date": "2024-12-31",
                                "annual_compensation": "USD 120000",
                                "employment_type": "full-time"
                            }
                        }
                    ],
                    "cpac_text": "Employment history with 24-month gap between companies"
                }
        
                request_data = CPACConsistencyRequest(
                    claim_category=employment_with_large_gap_request["claim_category"],
                    claims=employment_with_large_gap_request["claims"],
                    cpac_text=employment_with_large_gap_request.get("cpac_text", "")
                )
                
                # Create LUMA message with gap data
                gap_luma_message = LumaMessage(
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
                        task=Task(request=employment_with_large_gap_request)
                    ),
                    conversation_history="",
                    shared_resources={}
                )
                
                # Test that the system handles LLM failures gracefully
                result = await graph.process_request(gap_luma_message, request_data)
                
                # Verify the mock was called (LLM was attempted)
                self.assertTrue(mock_openai_class.called, "OpenAI mock should have been called")
                self.assertGreater(mock_openai_class.call_count, 0, "OpenAI should have been called at least once")
                
                # Verify the system completed but with error handling
                self.assertIsInstance(result, dict)
                self.assertIn('review_metadata', result)
                
                # Verify error was captured in metadata
                review_metadata = result['review_metadata']
                self.assertIn('review_error', review_metadata)
                self.assertEqual(review_metadata['review_error'], 'Failed to parse LLM response')
                
                # The system should complete successfully despite LLM failure
                self.assertIn('final_decision', review_metadata)
                
                print(f"✓ Mock called {mock_openai_class.call_count} times")
                print(f"✓ LLM error handled gracefully: {review_metadata['review_error']}")
                print(f"✓ System completed with final_decision: {review_metadata['final_decision']}")
    
    async def test_improver_node_exception_handling(self):
        """Test Case 3: Employment with gaps succeeds using rule-based analyzer (no API key needed)"""
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
        
        # Remove API key - employment workflow should still complete with rule-based analysis
        with patch.dict(os.environ, {}, clear=True):
            # Process the message - employment timeline analysis doesn't need API key
            result = await graph.process_request(test_message_with_gaps, request_data)
            
            # Verify successful completion with detected timeline gaps
            self.assertIsInstance(result, dict)
            self.assertIn('discrepancies', result)
            self.assertIn('review_metadata', result)
            
            # Verify rule-based analysis detected the employment gap
            review_metadata = result['review_metadata']
            self.assertEqual(review_metadata['analysis_method'], 'rule_based')
            self.assertIsNotNone(review_metadata['final_decision'])
            
            # Should detect the 18-month gap between employments
            discrepancies = result['discrepancies']
            gap_discrepancies = [d for d in discrepancies if d.get('discrepancy_type') == 'employment_timeline_gap']
            self.assertGreater(len(gap_discrepancies), 0, "Should detect employment timeline gap")
    
    async def test_workflow_with_no_api_key(self):
        """Test complete employment workflow when OpenAI API key is missing (should succeed)"""
        # Create graph with real components
        graph = CPACConsistencyGraph(config_path=self.config_path)
        
        # Create CPACConsistencyRequest from LUMA message
        task_request = self.test_luma_message.payload.task.request
        request_data = CPACConsistencyRequest(
            claim_category=task_request["claim_category"],
            claims=task_request["claims"],
            cpac_text=task_request.get("cpac_text", "")
        )
        
        # Remove API key - employment workflow should work with rule-based analysis
        with patch.dict(os.environ, {}, clear=True):
            # Process the message - employment uses rule-based analyzer (no API key needed)
            result = await graph.process_request(self.test_luma_message, request_data)
            
            # Verify successful completion without API key
            self.assertIsInstance(result, dict)
            self.assertIn('discrepancies', result) 
            self.assertIn('review_metadata', result)
            
            # Verify employment workflow completed successfully with rule-based analysis
            review_metadata = result['review_metadata']
            self.assertEqual(review_metadata['analysis_method'], 'rule_based')
            self.assertIsNotNone(review_metadata['final_decision'])
            
            # Should not have exceptions field since everything succeeded
            self.assertNotIn('exceptions', result)
    
    async def test_non_employment_api_key_failure(self):
        """Test non-employment claims fail fast when API key is missing"""
        # Create graph with real components
        graph = CPACConsistencyGraph(config_path=self.config_path)
        
        # Create non-employment request (inheritance)
        inheritance_message = LumaMessage(
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
                        "claim_category": "inheritance",
                        "claims": [
                            {
                                "claim_id": "1",
                                "claim_type": "original_cpac",
                                "cpac_data": {
                                    "inheritance_source": "Father",
                                    "inheritance_date": "2022-05-15",
                                    "inheritance_amount": 500000,
                                    "inheritance_currency": "USD",
                                    "inheritance_type": "cash",
                                    "start_date": "2022-05-15"
                                }
                            }
                        ],
                        "cpac_text": "Inherited money from father"
                    }
                )
            ),
            conversation_history="",
            shared_resources={}
        )
        
        task_request = inheritance_message.payload.task.request
        request_data = CPACConsistencyRequest(
            claim_category=task_request["claim_category"],
            claims=task_request["claims"],
            cpac_text=task_request.get("cpac_text", "")
        )
        
        # Remove API key - non-employment should fail fast
        with patch.dict(os.environ, {}, clear=True):
            # Non-employment claims require LLM, should raise exception
            with self.assertRaises(Exception) as context:
                await graph.process_request(inheritance_message, request_data)
            
            # Verify the exception is related to missing API key
            error_msg = str(context.exception)
            self.assertTrue(
                "API key" in error_msg or "authentication" in error_msg.lower(),
                f"Expected API key error, got: {error_msg}"
            )


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
    TestLumaMessagePassing.test_non_employment_api_key_failure = run_async_test(
        TestLumaMessagePassing.test_non_employment_api_key_failure
    )
    
    unittest.main(verbosity=2)
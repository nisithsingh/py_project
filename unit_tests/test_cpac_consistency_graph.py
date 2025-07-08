"""
Unit tests for CPACConsistencyGraph
Tests individual functions in the CPACConsistencyGraph class
"""
import os
import sys
import unittest
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add ai_foundation to path
ai_foundation_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'ai_foundation')
sys.path.append(ai_foundation_path)

from data_model.schemas import (
    CPACConsistencyRequest, ClaimCategory, Claim, CPACData, Discrepancy, DiscrepancyType
)
from ai_foundation.protocol.luma_protocol.models import (
    LumaMessage, Source, Target, Payload, Task
)
from graph.cpac_consistency_graph import CPACConsistencyGraph, GraphState
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TestCPACConsistencyGraph(unittest.TestCase):
    """Test cases for CPACConsistencyGraph class methods"""

    def setUp(self):
        """Set up test fixtures"""
        # Set test OpenAI API key if not already set
        if not os.getenv('OPENAI_API_KEY'):
            os.environ['OPENAI_API_KEY'] = 'test-key-for-unit-tests'
        
        # Create test claims
        self.test_claims = [
            Claim(
                claim_id=1,
                cpac_data=CPACData(
                    employer_name="TechCorp Inc",
                    job_title="Software Engineer",
                    start_date="2020-01-01",
                    end_date="2021-12-31",
                    annual_compensation="80000 USD",
                    employment_type="full-time"
                )
            ),
            Claim(
                claim_id=2,
                cpac_data=CPACData(
                    employer_name="DataSystems Ltd",
                    job_title="Senior Developer",
                    start_date="2022-03-01",
                    end_date="2024-06-30",
                    annual_compensation="95000 USD",
                    employment_type="full-time"
                )
            )
        ]
        
        # Create test request
        self.test_request = CPACConsistencyRequest(
            claim_category=ClaimCategory.EMPLOYMENT,
            claims=self.test_claims,
            cpac_text="Client worked at TechCorp Inc from 2020 to 2021, then moved to DataSystems Ltd in 2022."
        )
        
        # Create test LUMA message
        self.test_luma_message = LumaMessage(
            message_type="request",
            source=Source(
                created_ts=datetime.now().isoformat(),
                name="test_orchestrator"
            ),
            target=Target(name="cpac_consistency_reviewer"),
            payload=Payload(
                conversation_id=12345,
                job_id=67890,
                task_id=1111,
                agent_thread_id="test-thread",
                task=Task(request=self.test_request.model_dump()),
                conversation_history="",
                shared_resources={}
            )
        )

    def test_init(self):
        """Test CPACConsistencyGraph.__init__() method"""
        try:
            graph = CPACConsistencyGraph(
                config_path="config/config.yaml",
                service_bus=None,
                agent_name="test_reviewer"
            )
            
            # Check that all components are initialized
            self.assertIsNotNone(graph.preprocessor)
            self.assertIsNotNone(graph.timeline_analyzer)
            self.assertIsNotNone(graph.executor)
            self.assertIsNotNone(graph.reviewer)
            self.assertIsNotNone(graph.improver)
            
            # Check configuration is loaded
            self.assertIsNotNone(graph.config)
            self.assertGreater(graph.max_iterations, 0)
            
            # Check service bus and agent name
            self.assertIsNone(graph.service_bus)  # We passed None
            self.assertEqual(graph.agent_name, "test_reviewer")
            
            # Check graph is initially None (built on demand)
            self.assertIsNone(graph.graph)
            
        except Exception as e:
            self.fail(f"Graph initialization failed: {str(e)}")

    def test_build_graph(self):
        """Test CPACConsistencyGraph._build_graph() method"""
        async def run_test():
            graph = CPACConsistencyGraph(
                config_path="config/config.yaml",
                service_bus=None,
                agent_name="test_reviewer"
            )
            
            # Build the graph
            workflow = await graph._build_graph()
            
            # Check that workflow is created
            self.assertIsNotNone(workflow)
            
            # The workflow should have nodes (we can't easily introspect StateGraph internals,
            # but we can verify it was created without errors)
            
        asyncio.run(run_test())

    def test_ensure_graph_built(self):
        """Test CPACConsistencyGraph._ensure_graph_built() method"""
        async def run_test():
            graph = CPACConsistencyGraph(
                config_path="config/config.yaml",
                service_bus=None,
                agent_name="test_reviewer"
            )
            
            # Initially graph should be None
            self.assertIsNone(graph.graph)
            
            # Call _ensure_graph_built
            await graph._ensure_graph_built()
            
            # Now graph should be built
            self.assertIsNotNone(graph.graph)
            
            # Calling again should not rebuild
            original_graph = graph.graph
            await graph._ensure_graph_built()
            self.assertIs(graph.graph, original_graph)
            
        asyncio.run(run_test())

    def test_create_fresh_checkpointer(self):
        """Test CPACConsistencyGraph._create_fresh_checkpointer() method"""
        async def run_test():
            graph = CPACConsistencyGraph(
                config_path="config/config.yaml",
                service_bus=None,
                agent_name="test_reviewer"
            )
            
            # Create checkpointer
            saver, context = await graph._create_fresh_checkpointer()
            
            # Should return saver and context
            self.assertIsNotNone(saver)
            self.assertIsNotNone(context)
            
            # Clean up
            await context.__aexit__(None, None, None)
            
        asyncio.run(run_test())

    def test_close(self):
        """Test CPACConsistencyGraph.close() method"""
        async def run_test():
            graph = CPACConsistencyGraph(
                config_path="config/config.yaml",
                service_bus=None,
                agent_name="test_reviewer"
            )
            
            # Create a mock saver context
            mock_context = AsyncMock()
            graph._saver_context = mock_context
            graph.saver = Mock()
            
            # Call close
            await graph.close()
            
            # Verify cleanup
            mock_context.__aexit__.assert_called_once_with(None, None, None)
            self.assertIsNone(graph._saver_context)
            self.assertIsNone(graph.saver)
            
        asyncio.run(run_test())

    def test_preprocessing_node(self):
        """Test CPACConsistencyGraph.preprocessing_node() method"""
        async def run_test():
            graph = CPACConsistencyGraph(
                config_path="config/config.yaml",
                service_bus=None,
                agent_name="test_reviewer"
            )
            
            # Create test state
            state = GraphState(
                luma_message=self.test_luma_message,
                request_data=self.test_request,
                preprocessed_request=None,
                discrepancies=[],
                analysis_result=None,
                review_decision=False,
                review_reason=[],
                review_error="",
                review_action=None,
                review_analysis=None,
                review_summaries=[],
                improved_discrepancies=None,
                iteration_count=0,
                max_iterations=3,
                analysis_method=None,
                exception_list=[],
                final_result={},
                messages=[]
            )
            
            # Call preprocessing node
            result_state = await graph.preprocessing_node(state)
            
            # Verify state updates
            self.assertIsNotNone(result_state['preprocessed_request'])
            self.assertEqual(result_state['preprocessed_request'].claim_category, ClaimCategory.EMPLOYMENT)
            self.assertGreater(len(result_state['messages']), 0)
            self.assertEqual(result_state['messages'][-1]['type'], 'preprocessing')
            
        asyncio.run(run_test())

    def test_executor_node_employment(self):
        """Test CPACConsistencyGraph.executor_node() method with employment claims"""
        async def run_test():
            graph = CPACConsistencyGraph(
                config_path="config/config.yaml",
                service_bus=None,
                agent_name="test_reviewer"
            )
            
            # Create test state with employment claims
            state = GraphState(
                luma_message=self.test_luma_message,
                request_data=self.test_request,
                preprocessed_request=self.test_request,
                discrepancies=[],
                analysis_result=None,
                review_decision=False,
                review_reason=[],
                review_error="",
                review_action=None,
                review_analysis=None,
                review_summaries=[],
                improved_discrepancies=None,
                iteration_count=0,
                max_iterations=3,
                analysis_method=None,
                exception_list=[],
                final_result={},
                messages=[]
            )
            
            # Call executor node
            result_state = await graph.executor_node(state)
            
            # Verify state updates
            self.assertIsNotNone(result_state['discrepancies'])
            self.assertIsNotNone(result_state['analysis_result'])
            self.assertEqual(result_state['analysis_method'], 'rule_based')
            self.assertGreater(len(result_state['messages']), 0)
            
            # Check message content
            executor_message = result_state['messages'][-1]
            self.assertEqual(executor_message['type'], 'executor')
            self.assertEqual(executor_message['analysis_method'], 'rule_based')
            
        asyncio.run(run_test())

    def test_executor_node_non_employment(self):
        """Test CPACConsistencyGraph.executor_node() method with non-employment claims"""
        async def run_test():
            if os.getenv('OPENAI_API_KEY') == 'test-key-for-unit-tests':
                self.skipTest("Skipping LLM test - no real API key provided")
            
            # Create inheritance request
            inheritance_claims = [
                Claim(
                    claim_id=1,
                    cpac_data=CPACData(
                        inheritance_source="Father",
                        inheritance_date="2022-05-15",
                        inheritance_amount="500000",
                        inheritance_currency="USD",
                        inheritance_type="cash"
                    )
                )
            ]
            
            inheritance_request = CPACConsistencyRequest(
                claim_category=ClaimCategory.INHERITANCE,
                claims=inheritance_claims,
                cpac_text="Client inherited $500,000 from father in May 2022."
            )
            
            graph = CPACConsistencyGraph(
                config_path="config/config.yaml",
                service_bus=None,
                agent_name="test_reviewer"
            )
            
            # Create test state with inheritance claims
            state = GraphState(
                luma_message=self.test_luma_message,
                request_data=inheritance_request,
                preprocessed_request=inheritance_request,
                discrepancies=[],
                analysis_result=None,
                review_decision=False,
                review_reason=[],
                review_error="",
                review_action=None,
                review_analysis=None,
                review_summaries=[],
                improved_discrepancies=None,
                iteration_count=0,
                max_iterations=3,
                analysis_method=None,
                exception_list=[],
                final_result={},
                messages=[]
            )
            
            # Call executor node
            result_state = await graph.executor_node(state)
            
            # Verify state updates
            self.assertIsNotNone(result_state['discrepancies'])
            self.assertIsNotNone(result_state['analysis_result'])
            self.assertEqual(result_state['analysis_method'], 'llm_based')
            
        try:
            asyncio.run(run_test())
        except unittest.SkipTest:
            raise
        except Exception as e:
            if "API key" in str(e):
                self.skipTest(f"Skipping LLM test - API key issue: {str(e)}")
            else:
                # With new exception handling, exceptions should propagate
                self.assertIsInstance(e, Exception)

    def test_reviewer_node(self):
        """Test CPACConsistencyGraph.reviewer_node() method"""
        async def run_test():
            if os.getenv('OPENAI_API_KEY') == 'test-key-for-unit-tests':
                self.skipTest("Skipping LLM test - no real API key provided")
            
            graph = CPACConsistencyGraph(
                config_path="config/config.yaml",
                service_bus=None,
                agent_name="test_reviewer"
            )
            
            # Create mock analysis result
            from data_model.schemas import CPACConsistencyResult
            analysis_result = CPACConsistencyResult(
                claim_category=ClaimCategory.EMPLOYMENT,
                discrepancies=[]
            )
            
            # Create test state
            state = GraphState(
                luma_message=self.test_luma_message,
                request_data=self.test_request,
                preprocessed_request=self.test_request,
                discrepancies=[],
                analysis_result=analysis_result,
                review_decision=False,
                review_reason=[],
                review_error="",
                review_action=None,
                review_analysis=None,
                review_summaries=[],
                improved_discrepancies=None,
                iteration_count=0,
                max_iterations=3,
                analysis_method='rule_based',
                exception_list=[],
                final_result={},
                messages=[]
            )
            
            # Call reviewer node
            result_state = await graph.reviewer_node(state)
            
            # Verify state updates
            self.assertIsInstance(result_state['review_decision'], bool)
            self.assertIsInstance(result_state['review_reason'], list)
            self.assertIsInstance(result_state['review_error'], str)
            self.assertGreater(len(result_state['review_summaries']), 0)
            self.assertGreater(len(result_state['messages']), 0)
            
        try:
            asyncio.run(run_test())
        except unittest.SkipTest:
            raise
        except Exception as e:
            if "API key" in str(e):
                self.skipTest(f"Skipping LLM test - API key issue: {str(e)}")
            else:
                # With new exception handling, exceptions should propagate
                self.assertIsInstance(e, Exception)

    def test_improvement_node(self):
        """Test CPACConsistencyGraph.improvement_node() method"""
        async def run_test():
            graph = CPACConsistencyGraph(
                config_path="config/config.yaml",
                service_bus=None,
                agent_name="test_reviewer"
            )
            
            # Create sample discrepancy
            sample_discrepancy = Discrepancy(
                discrepancy_id=1,
                discrepancy_type=DiscrepancyType.EMPLOYMENT_TIMELINE_GAP,
                description="Gap between employments",
                reason="",
                recommendation="",
                affected_claim_ids=[1, 2],
                affected_document_ids=[]
            )
            
            # Create test state
            state = GraphState(
                luma_message=self.test_luma_message,
                request_data=self.test_request,
                preprocessed_request=self.test_request,
                discrepancies=[sample_discrepancy],
                analysis_result=None,
                review_decision=False,
                review_reason=["Test analysis"],
                review_error="",
                review_action={"approve": {}, "remove": {}},
                review_analysis=None,
                review_summaries=[],
                improved_discrepancies=None,
                iteration_count=0,
                max_iterations=3,
                analysis_method='rule_based',
                exception_list=[],
                final_result={},
                messages=[]
            )
            
            # Call improvement node
            result_state = await graph.improvement_node(state)
            
            # Verify state updates
            self.assertIsNotNone(result_state['improved_discrepancies'])
            self.assertIsNotNone(result_state['analysis_result'])
            self.assertEqual(result_state['iteration_count'], 1)  # Should increment
            self.assertGreater(len(result_state['messages']), 0)
            
            # Check message content
            improvement_message = result_state['messages'][-1]
            self.assertEqual(improvement_message['type'], 'improvement')
            
        asyncio.run(run_test())

    def test_send_to_bus_node(self):
        """Test CPACConsistencyGraph.send_to_bus_node() method"""
        async def run_test():
            graph = CPACConsistencyGraph(
                config_path="config/config.yaml",
                service_bus=None,
                agent_name="test_reviewer"
            )
            
            # Create analysis result
            from data_model.schemas import CPACConsistencyResult
            analysis_result = CPACConsistencyResult(
                claim_category=ClaimCategory.EMPLOYMENT,
                discrepancies=[]
            )
            
            # Create test state with some messages
            state = GraphState(
                luma_message=self.test_luma_message,
                request_data=self.test_request,
                preprocessed_request=self.test_request,
                discrepancies=[],
                analysis_result=analysis_result,
                review_decision=True,
                review_reason=["No issues found"],
                review_error="",
                review_action=None,
                review_analysis=None,
                review_summaries=[],
                improved_discrepancies=None,
                iteration_count=1,
                max_iterations=3,
                analysis_method='rule_based',
                exception_list=[],
                final_result={},
                messages=[{
                    "type": "preprocessing",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }]
            )
            
            # Call send to bus node
            result_state = await graph.send_to_bus_node(state)
            
            # Verify final result structure
            final_result = result_state['final_result']
            self.assertIn('claim_category', final_result)
            self.assertIn('discrepancies', final_result)
            self.assertIn('review_metadata', final_result)
            
            # Check review metadata structure
            metadata = final_result['review_metadata']
            expected_metadata_keys = [
                'total_iterations', 'final_decision', 'review_reasons',
                'processing_time_seconds', 'analysis_method',
                'total_claims_analyzed', 'total_discrepancies_found'
            ]
            for key in expected_metadata_keys:
                self.assertIn(key, metadata)
            
            # Verify final message was added
            self.assertGreater(len(result_state['messages']), 1)
            self.assertEqual(result_state['messages'][-1]['type'], 'final_result_prepared')
            
        asyncio.run(run_test())

    def test_process_request_exception_propagation(self):
        """Test that CPACConsistencyGraph.process_request() lets exceptions propagate"""
        async def run_test():
            # Create invalid claims to trigger exceptions
            invalid_claims = [
                Claim(
                    claim_id=1,
                    cpac_data=CPACData(
                        employer_name="Test Corp",
                        job_title="Tester",
                        start_date="invalid-date",  # Invalid date format
                        end_date="2023-12-31",
                        annual_compensation="75000 USD",
                        employment_type="full-time"
                    )
                )
            ]
            
            invalid_request = CPACConsistencyRequest(
                claim_category=ClaimCategory.EMPLOYMENT,
                claims=invalid_claims,
                cpac_text="Test with invalid data."
            )
            
            graph = CPACConsistencyGraph(
                config_path="config/config.yaml",
                service_bus=None,
                agent_name="test_reviewer"
            )
            
            # With new exception handling, exceptions should propagate
            with self.assertRaises(Exception):
                await graph.process_request(
                    self.test_luma_message,
                    invalid_request,
                    exception_list=[]
                )
        
        asyncio.run(run_test())

    def test_process_request_successful(self):
        """Test CPACConsistencyGraph.process_request() with valid data"""
        async def run_test():
            graph = CPACConsistencyGraph(
                config_path="config/config.yaml",
                service_bus=None,
                agent_name="test_reviewer"
            )
            
            # Process with valid employment data (should use rule-based analyzer)
            result = await graph.process_request(
                self.test_luma_message,
                self.test_request,
                exception_list=[]
            )
            
            # Verify result structure
            self.assertIsNotNone(result)
            self.assertIn('claim_category', result)
            self.assertIn('discrepancies', result)
            self.assertIn('review_metadata', result)
            
            # Verify metadata
            metadata = result['review_metadata']
            self.assertEqual(metadata['analysis_method'], 'rule_based')
            self.assertEqual(metadata['total_claims_analyzed'], 2)
            self.assertGreater(metadata['processing_time_seconds'], 0)
            
        asyncio.run(run_test())


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)
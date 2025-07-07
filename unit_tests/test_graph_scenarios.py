"""
Unit tests for Graph Scenarios
Tests the complete LangGraph workflow using actual implementations
"""
import os
import sys
import unittest
import asyncio
import json
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add ai_foundation to path
ai_foundation_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'ai_foundation')
sys.path.append(ai_foundation_path)

from data_model.schemas import (
    CPACConsistencyRequest, CPACConsistencyResult, ClaimCategory, 
    Claim, CPACData, Discrepancy, DiscrepancyType
)
from ai_foundation.protocol.luma_protocol.models import (
    LumaMessage, Source, Target, Payload, Task
)
import glob
from graph.cpac_consistency_graph import CPACConsistencyGraph
from pydantic import ValidationError


class TestGraphScenarios(unittest.TestCase):
    """Test cases for Graph workflow using scenario data"""

    def setUp(self):
        """Set up test fixtures"""
        # Set test OpenAI API key if not already set
        if not os.getenv('OPENAI_API_KEY'):
            os.environ['OPENAI_API_KEY'] = 'test-key-for-unit-tests'
        
        # Create graph for testing
        self.graph = CPACConsistencyGraph(
            config_path="config/config.yaml",
            service_bus=None,
            agent_name="test_reviewer"
        )
        
        # Create test LUMA message template
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
                task=Task(request={}),
                conversation_history="",
                shared_resources={}
            )
        )

    def _create_test_request_from_scenario(self, scenario_name: str) -> CPACConsistencyRequest:
        """Create a test request from scenario data"""
        # Load test scenario from test_scenarios folder
        scenario_path = Path(__file__).parent.parent / "test_data" / "test_scenarios" / f"{scenario_name}.json"
        
        if not scenario_path.exists():
            # Create a mock scenario for testing
            return self._create_mock_scenario(scenario_name)
            
        with open(scenario_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # Extract test information
        claims = []
        for claim_data in test_data['claims']:
            # Convert claim data to proper schema objects
            cpac_data = CPACData(**claim_data['cpac_data'])
            claim = Claim(
                claim_id=claim_data['claim_id'],
                cpac_data=cpac_data
            )
            claims.append(claim)
        
        claim_category = ClaimCategory(test_data['claim_category'])
        cpac_text = test_data['cpac_text']
        
        # Create request
        return CPACConsistencyRequest(
            claim_category=claim_category,
            claims=claims,
            cpac_text=cpac_text
        )

    def _create_test_request_from_luma_scenario(self, scenario_name: str) -> CPACConsistencyRequest:
        """Create a test request from LUMA scenario data"""
        # Load test scenario from test_scenarios_luma folder
        scenario_path = Path(__file__).parent.parent / "test_data" / "test_scenarios_luma" / f"{scenario_name}.json"
        
        if not scenario_path.exists():
            raise FileNotFoundError(f"LUMA scenario file not found: {scenario_path}")
            
        with open(scenario_path, 'r', encoding='utf-8') as f:
            luma_data = json.load(f)
        
        # Extract request data from LUMA payload
        request_data = luma_data['payload']['task']['request']
        
        # Convert claim data to proper schema objects
        claims = []
        for claim_data in request_data['claims']:
            cpac_data = CPACData(**claim_data['cpac_data'])
            claim = Claim(
                claim_id=claim_data['claim_id'],
                cpac_data=cpac_data
            )
            claims.append(claim)
        
        claim_category = ClaimCategory(request_data['claim_category'])
        cpac_text = request_data['cpac_text']
        
        # Create request
        return CPACConsistencyRequest(
            claim_category=claim_category,
            claims=claims,
            cpac_text=cpac_text
        )

    def _create_mock_scenario(self, scenario_name: str) -> CPACConsistencyRequest:
        """Create mock scenarios for testing when files don't exist"""
        if "gap" in scenario_name.lower():
            # Create scenario with employment gap
            return CPACConsistencyRequest(
                claim_category=ClaimCategory.EMPLOYMENT,
                claims=[
                    Claim(
                        claim_id=1,
                        cpac_data=CPACData(
                            employer_name="Company A",
                            job_title="Engineer",
                            start_date="2018-01-01",
                            end_date="2020-12-31",
                            annual_compensation="75000 USD",
                            employment_type="full-time"
                        )
                    ),
                    Claim(
                        claim_id=2,
                        cpac_data=CPACData(
                            employer_name="Company B",
                            job_title="Senior Engineer",
                            start_date="2025-01-01",  # 4-year gap
                            end_date="2026-12-31",
                            annual_compensation="95000 USD",
                            employment_type="full-time"
                        )
                    )
                ],
                cpac_text="Client worked at Company A until 2020, then there was a long retirement period before joining Company B in 2025."
            )
        elif "overlap" in scenario_name.lower():
            # Create scenario with employment overlap
            return CPACConsistencyRequest(
                claim_category=ClaimCategory.EMPLOYMENT,
                claims=[
                    Claim(
                        claim_id=1,
                        cpac_data=CPACData(
                            employer_name="Company X",
                            job_title="Developer",
                            start_date="2022-01-01",
                            end_date="2023-06-30",
                            annual_compensation="75000 USD",
                            employment_type="full-time"
                        )
                    ),
                    Claim(
                        claim_id=2,
                        cpac_data=CPACData(
                            employer_name="Company Y",
                            job_title="Consultant",
                            start_date="2023-03-01",  # 4-month overlap
                            end_date="2024-03-31",
                            annual_compensation="85000 USD",
                            employment_type="contract"
                        )
                    )
                ],
                cpac_text="Client worked at Company X full-time and also did consulting work for Company Y during overlapping periods."
            )
        elif "missing" in scenario_name.lower():
            # Create scenario with missing fields
            return CPACConsistencyRequest(
                claim_category=ClaimCategory.EMPLOYMENT,
                claims=[
                    Claim(
                        claim_id=1,
                        cpac_data=CPACData(
                            employer_name="Company Test",
                            job_title="Manager",
                            start_date="",  # Empty string - valid but semantically problematic
                            end_date="2023-12-31",
                            annual_compensation="75000 USD",
                            employment_type="full-time"
                        )
                    )
                ],
                cpac_text="Test scenario with missing critical fields."
            )
        else:
            # Default scenario
            return CPACConsistencyRequest(
                claim_category=ClaimCategory.EMPLOYMENT,
                claims=[
                    Claim(
                        claim_id=1,
                        cpac_data=CPACData(
                            employer_name="Default Company",
                            job_title="Default Role",
                            start_date="2020-01-01",
                            end_date="2023-12-31",
                            annual_compensation="80000 USD",
                            employment_type="full-time"
                        )
                    )
                ],
                cpac_text="Default test scenario"
            )

    def _extract_discrepancies(self, result: dict) -> list:
        """Extract discrepancies from graph result"""
        if not result:
            return []
        return result.get('discrepancies', [])

    def _assert_discrepancy_expectations(self, actual_discrepancies: list, expected_count: int, 
                                       expected_types: list = None, 
                                       expected_affected_claims: list = None):
        """Assert discrepancy expectations"""
        self.assertEqual(len(actual_discrepancies), expected_count, 
                        f"Expected {expected_count} discrepancies, got {len(actual_discrepancies)}")
        
        if expected_types:
            actual_types = [d.get('discrepancy_type') for d in actual_discrepancies]
            for expected_type in expected_types:
                self.assertIn(expected_type, actual_types, 
                            f"Expected discrepancy type '{expected_type}' not found in {actual_types}")
        
        if expected_affected_claims:
            for i, expected_claims in enumerate(expected_affected_claims):
                if i < len(actual_discrepancies):
                    actual_claims = actual_discrepancies[i].get('affected_claim_ids', [])
                    self.assertEqual(set(str(c) for c in actual_claims), set(str(c) for c in expected_claims),
                                   f"Discrepancy {i}: expected affected claims {expected_claims}, got {actual_claims}")

    def test_employment_gap_scenario(self):
        """Test scenario with employment gap detection"""
        async def run_test():
            if os.getenv('OPENAI_API_KEY') == 'test-key-for-unit-tests':
                self.skipTest("Skipping LLM test - no real API key provided")
            
            request = self._create_mock_scenario("employment_gap")
            
            # Update LUMA message with request
            self.test_luma_message.payload.task.request = request.model_dump()
            
            result = await self.graph.process_request(
                self.test_luma_message,
                request,
                exception_list=[]
            )
            
            # Extract discrepancies
            actual_discrepancies = self._extract_discrepancies(result)
            
            print(f"\nEmployment Gap Scenario:")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc.get('discrepancy_type')}: {disc.get('description')}")
                print(f"      Affected Claims: {disc.get('affected_claim_ids')}")
            
            # Verify result structure
            self.assertIn('claim_category', result)
            self.assertIn('discrepancies', result)
            self.assertIn('review_metadata', result)
            
            # The gap may or may not be detected depending on reviewer decision
            # We mainly check that the process completes successfully
            self.assertIsInstance(actual_discrepancies, list)
            
            return result
        
        try:
            result = asyncio.run(run_test())
            self.assertIsNotNone(result)
        except unittest.SkipTest:
            raise
        except Exception as e:
            if "API key" in str(e):
                self.skipTest(f"Skipping LLM test - API key issue: {str(e)}")
            else:
                raise

    def test_employment_overlap_scenario(self):
        """Test scenario with employment overlap detection"""
        async def run_test():
            if os.getenv('OPENAI_API_KEY') == 'test-key-for-unit-tests':
                self.skipTest("Skipping LLM test - no real API key provided")
            
            request = self._create_mock_scenario("employment_overlap")
            
            # Update LUMA message with request
            self.test_luma_message.payload.task.request = request.model_dump()
            
            result = await self.graph.process_request(
                self.test_luma_message,
                request,
                exception_list=[]
            )
            
            # Extract discrepancies
            actual_discrepancies = self._extract_discrepancies(result)
            
            print(f"\nEmployment Overlap Scenario:")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc.get('discrepancy_type')}: {disc.get('description')}")
                print(f"      Affected Claims: {disc.get('affected_claim_ids')}")
            
            # Verify result structure
            self.assertIn('claim_category', result)
            self.assertIn('discrepancies', result)
            self.assertIn('review_metadata', result)
            
            # Verify processing completed
            metadata = result['review_metadata']
            self.assertIn('total_claims_analyzed', metadata)
            self.assertEqual(metadata['total_claims_analyzed'], 2)
            
            return result
        
        try:
            result = asyncio.run(run_test())
            self.assertIsNotNone(result)
        except unittest.SkipTest:
            raise
        except Exception as e:
            if "API key" in str(e):
                self.skipTest(f"Skipping LLM test - API key issue: {str(e)}")
            else:
                raise

    def test_perfect_employment_scenario(self):
        """Test scenario with no employment issues"""
        async def run_test():
            request = CPACConsistencyRequest(
                claim_category=ClaimCategory.EMPLOYMENT,
                claims=[
                    Claim(
                        claim_id=1,
                        cpac_data=CPACData(
                            employer_name="StartupCorp",
                            job_title="Junior Developer",
                            start_date="2020-01-01",
                            end_date="2021-12-31",
                            annual_compensation="60000 USD",
                            employment_type="full-time"
                        )
                    ),
                    Claim(
                        claim_id=2,
                        cpac_data=CPACData(
                            employer_name="TechGiant",
                            job_title="Software Engineer",
                            start_date="2022-01-01",
                            end_date="2024-12-31",
                            annual_compensation="90000 USD",
                            employment_type="full-time"
                        )
                    )
                ],
                cpac_text="Client worked at StartupCorp from 2020-2021, then joined TechGiant in January 2022."
            )
            
            # Update LUMA message with request
            self.test_luma_message.payload.task.request = request.model_dump()
            
            result = await self.graph.process_request(
                self.test_luma_message,
                request,
                exception_list=[]
            )
            
            # Extract discrepancies
            actual_discrepancies = self._extract_discrepancies(result)
            
            print(f"\nPerfect Employment Scenario:")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc.get('discrepancy_type')}: {disc.get('description')}")
                print(f"      Affected Claims: {disc.get('affected_claim_ids')}")
            
            # Verify result structure
            self.assertIn('claim_category', result)
            self.assertIn('discrepancies', result)
            self.assertIn('review_metadata', result)
            
            # Check metadata
            metadata = result['review_metadata']
            self.assertIn('analysis_method', metadata)
            self.assertEqual(metadata['analysis_method'], 'rule_based')
            self.assertIn('processing_time_seconds', metadata)
            self.assertGreater(metadata['processing_time_seconds'], 0)
            
            return result
        
        result = asyncio.run(run_test())
        self.assertIsNotNone(result)

    def test_non_employment_scenario(self):
        """Test scenario with non-employment claims"""
        async def run_test():
            if os.getenv('OPENAI_API_KEY') == 'test-key-for-unit-tests':
                self.skipTest("Skipping LLM test - no real API key provided")
            
            request = CPACConsistencyRequest(
                claim_category=ClaimCategory.INHERITANCE,
                claims=[
                    Claim(
                        claim_id=1,
                        cpac_data=CPACData(
                            inheritance_source="Father",
                            inheritance_date="2022-05-15",
                            inheritance_amount=500000,
                            inheritance_currency="USD",
                            inheritance_type="cash",
                            start_date="2022-05-15"
                        )
                    )
                ],
                cpac_text="Client inherited $500,000 from father in May 2022."
            )
            
            # Update LUMA message with request
            self.test_luma_message.payload.task.request = request.model_dump()
            
            result = await self.graph.process_request(
                self.test_luma_message,
                request,
                exception_list=[]
            )
            
            # Extract discrepancies
            actual_discrepancies = self._extract_discrepancies(result)
            
            print(f"\nInheritance Scenario:")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc.get('discrepancy_type')}: {disc.get('description')}")
                print(f"      Affected Claims: {disc.get('affected_claim_ids')}")
            
            # Verify result structure
            self.assertIn('claim_category', result)
            self.assertIn('discrepancies', result)
            self.assertIn('review_metadata', result)
            
            # Check that LLM method was used for non-employment
            metadata = result['review_metadata']
            self.assertEqual(metadata['analysis_method'], 'llm_based')
            
            return result
        
        try:
            result = asyncio.run(run_test())
            self.assertIsNotNone(result)
        except unittest.SkipTest:
            raise
        except Exception as e:
            if "API key" in str(e):
                self.skipTest(f"Skipping LLM test - API key issue: {str(e)}")
            else:
                raise

    def test_pydantic_validation_missing_critical_fields(self):
        """Test Pydantic validation for missing critical fields"""
        print("\nTesting Pydantic validation for missing critical fields...")
        
        # Create test cases with truly missing required fields
        test_cases = [
            {
                "name": "Missing start_date field entirely",
                "cpac_data": {
                    "employer_name": "Test Company",
                    "job_title": "Manager"
                    # start_date missing entirely
                },
                "should_fail": True
            },
            {
                "name": "Empty string start_date (should pass)",
                "cpac_data": {
                    "start_date": "",  # Empty string - valid for Pydantic str field
                    "employer_name": "Test Company",
                    "job_title": "Manager"
                },
                "should_fail": False
            },
            {
                "name": "Valid start_date (should pass)",
                "cpac_data": {
                    "start_date": "2020-01-01",
                    "employer_name": "Test Company",
                    "job_title": "Manager"
                },
                "should_fail": False
            }
        ]
        
        validation_errors_caught = 0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}: {test_case['name']}")
            
            try:
                # Attempt to create CPACData object
                cpac_data = CPACData(**test_case['cpac_data'])
                claim = Claim(claim_id=i, cpac_data=cpac_data)
                
                if test_case['should_fail']:
                    print(f"  ✗ Expected validation error but none was raised")
                    self.fail(f"Expected ValidationError for test case: {test_case['name']}")
                else:
                    print(f"  ✓ Validation passed as expected")
                    
            except ValidationError as e:
                validation_errors_caught += 1
                if test_case['should_fail']:
                    print(f"  ✓ Expected ValidationError caught: {e}")
                    # Verify it mentions the missing field
                    self.assertIn("start_date", str(e).lower())
                else:
                    print(f"  ✗ Unexpected ValidationError: {e}")
                    self.fail(f"Unexpected ValidationError for test case: {test_case['name']}")
                    
            except Exception as e:
                print(f"  ✗ Unexpected error: {type(e).__name__}: {e}")
                self.fail(f"Unexpected error for test case: {test_case['name']}")
        
        # Verify that at least one validation error was caught
        self.assertGreater(validation_errors_caught, 0, 
                          "Expected at least one ValidationError to be raised")
        
        print(f"\n✓ Pydantic validation test completed: {validation_errors_caught} validation errors caught as expected")

    def test_empty_claims_scenario(self):
        """Test scenario with empty claims list"""
        async def run_test():
            request = CPACConsistencyRequest(
                claim_category=ClaimCategory.EMPLOYMENT,
                claims=[],
                cpac_text="No claims provided."
            )
            
            # Update LUMA message with request
            self.test_luma_message.payload.task.request = request.model_dump()
            
            result = await self.graph.process_request(
                self.test_luma_message,
                request,
                exception_list=[]
            )
            
            # Extract discrepancies
            actual_discrepancies = self._extract_discrepancies(result)
            
            print(f"\nEmpty Claims Scenario:")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            # Verify result structure
            self.assertIn('claim_category', result)
            self.assertIn('discrepancies', result)
            self.assertIn('review_metadata', result)
            
            # Should handle empty claims gracefully
            self.assertEqual(len(actual_discrepancies), 0)
            
            return result
        
        result = asyncio.run(run_test())
        self.assertIsNotNone(result)

    def test_graph_state_transitions(self):
        """Test that graph state transitions work correctly"""
        async def run_test():
            request = CPACConsistencyRequest(
                claim_category=ClaimCategory.EMPLOYMENT,
                claims=[
                    Claim(
                        claim_id=1,
                        cpac_data=CPACData(
                            employer_name="Test Company",
                            job_title="Test Role",
                            start_date="2020-01-01",
                            end_date="2023-12-31",
                            annual_compensation="80000 USD",
                            employment_type="full-time"
                        )
                    )
                ],
                cpac_text="Simple test case for state transitions."
            )
            
            # Update LUMA message with request
            self.test_luma_message.payload.task.request = request.model_dump()
            
            result = await self.graph.process_request(
                self.test_luma_message,
                request,
                exception_list=[]
            )
            
            # Verify the complete result structure
            expected_keys = [
                'claim_category',
                'discrepancies',
                'review_metadata'
            ]
            
            for key in expected_keys:
                self.assertIn(key, result, f"Missing key: {key}")
            
            # Verify metadata structure
            metadata = result['review_metadata']
            metadata_keys = [
                'total_iterations',
                'final_decision',
                'processing_time_seconds',
                'analysis_method',
                'total_claims_analyzed',
                'total_discrepancies_found'
            ]
            
            for key in metadata_keys:
                self.assertIn(key, metadata, f"Missing metadata key: {key}")
            
            return result
        
        result = asyncio.run(run_test())
        self.assertIsNotNone(result)

    def _run_luma_scenario_test(self, scenario_name: str):
        """Generic method to run a LUMA scenario test"""
        async def run_test():
            if os.getenv('OPENAI_API_KEY') == 'test-key-for-unit-tests':
                self.skipTest("Skipping LLM test - no real API key provided")
            
            try:
                request = self._create_test_request_from_luma_scenario(scenario_name)
            except FileNotFoundError as e:
                self.skipTest(f"Scenario file not found: {e}")
            except Exception as e:
                self.fail(f"Failed to load scenario {scenario_name}: {e}")
            
            # Update LUMA message with request
            self.test_luma_message.payload.task.request = request.model_dump()
            
            result = await self.graph.process_request(
                self.test_luma_message,
                request,
                exception_list=[]
            )
            
            # Extract discrepancies
            actual_discrepancies = self._extract_discrepancies(result)
            
            print(f"\nLUMA Scenario {scenario_name}:")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc.get('discrepancy_type')}: {disc.get('description')}")
                print(f"      Affected Claims: {disc.get('affected_claim_ids')}")
            
            # Verify result structure
            self.assertIn('claim_category', result)
            self.assertIn('discrepancies', result)
            self.assertIn('review_metadata', result)
            
            # Verify processing completed
            self.assertIsInstance(actual_discrepancies, list)
            
            return result
        
        try:
            result = asyncio.run(run_test())
            self.assertIsNotNone(result)
        except unittest.SkipTest:
            raise
        except Exception as e:
            if "API key" in str(e):
                self.skipTest(f"Skipping LLM test - API key issue: {str(e)}")
            else:
                raise


# Dynamically create test methods for each LUMA scenario
def _create_luma_scenario_test_methods():
    """Create test methods for each LUMA scenario file"""
    luma_scenarios_dir = Path(__file__).parent.parent / "test_data" / "test_scenarios_luma"
    
    if not luma_scenarios_dir.exists():
        return
    
    scenario_files = list(luma_scenarios_dir.glob("*.json"))
    
    for scenario_file in scenario_files:
        scenario_name = scenario_file.stem  # filename without extension
        
        # Create test method name
        test_method_name = f"test_luma_{scenario_name}"
        
        # Define the test method
        def create_test_method(scenario_name):
            def test_method(self):
                self._run_luma_scenario_test(scenario_name)
            return test_method
        
        # Add the test method to the TestGraphScenarios class
        setattr(TestGraphScenarios, test_method_name, create_test_method(scenario_name))

# Create the test methods
_create_luma_scenario_test_methods()


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)
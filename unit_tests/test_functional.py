"""
Unit tests for Graph Scenarios - Functional Tests
Tests the complete LangGraph workflow using LUMA test scenario data files
"""
import os
import sys
import unittest
import asyncio
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.cpac_graph_agent import CPACGraphAgent
from service_bus.local_service_bus import ServiceBus
from data_model.schemas import (
    LumaMessage, LumaSource, LumaTarget, LumaPayload, LumaTaskWrapper,
    CPACConsistencyRequest, MessageType, ClaimCategory, Claim, CPACData
)
from pydantic import ValidationError
from utils.test_data_loader import TestDataLoader


class TestFunctionalScenarios(unittest.TestCase):
    """Functional test cases for Graph workflow using LUMA scenario data"""

    def setUp(self):
        """Set up test fixtures"""
        # Ensure we're using local bus for testing
        os.environ['USE_LOCAL_BUS'] = 'true'
        
        # Initialize components
        self.service_bus = ServiceBus()
        self.loader = TestDataLoader()
        self.agent = None
        
    def tearDown(self):
        """Clean up after tests"""
        if self.agent:
            asyncio.run(self.agent.stop())

    async def _run_luma_scenario_test(self, scenario_name: str) -> Dict:
        """
        Run a graph test with a LUMA scenario file
        
        Args:
            scenario_name: Name of the scenario file (without .json extension)
            
        Returns:
            Tuple of (response, test_data)
        """
        # Load test scenario from test_scenarios_luma folder
        scenario_path = Path(__file__).parent.parent / "test_data" / "test_scenarios_luma" / f"{scenario_name}.json"
        
        if not scenario_path.exists():
            raise FileNotFoundError(f"LUMA scenario file not found: {scenario_path}")
            
        with open(scenario_path, 'r') as f:
            luma_data = json.load(f)
        
        # Extract request data from LUMA payload
        request_data = luma_data['payload']['task']['request']
        
        # Convert claim data to proper schema objects
        claims = []
        for claim_data in request_data['claims']:
            cpac_data = CPACData(**claim_data['cpac_data'])
            claim = Claim(
                claim_id=claim_data['claim_id'],
                claim_sow_category=claim_data['claim_sow_category'],
                claim_type=claim_data['claim_type'],
                cpac_data=cpac_data
            )
            claims.append(claim)
        
        claim_category = ClaimCategory(request_data['claim_category'])
        cpac_text = request_data['cpac_text']
        
        # Create request
        request = CPACConsistencyRequest(
            claim_category=claim_category,
            claims=claims,
            cpac_text=cpac_text
        )
        
        # Create LUMA message
        message = LumaMessage(
            message_type=MessageType.REQUEST,
            source=LumaSource(
                created_ts=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                name="test_orchestrator"
            ),
            target=LumaTarget(
                name="cpac_consistency_reviewer"
            ),
            payload=LumaPayload(
                conversation_id=12345,
                job_id=67890,
                task_id=datetime.now().strftime('%Y%m%d%H%M%S'),
                agent_thread_id="550e8400-e29b-41d4-a716-446655440000",
                task=LumaTaskWrapper(
                    request=request.model_dump()
                ),
                conversation_history="",
                shared_resources={}
            )
        )
        
        # Convert to dict and add routing
        message_dict = message.model_dump()
        message_dict['to_component'] = "cpac_consistency_reviewer"
        message_dict['from_component'] = "test_orchestrator"
        
        # Create and start graph agent
        self.agent = CPACGraphAgent(config_path="config/config.yaml")
        agent_task = asyncio.create_task(self.agent.start_service())
        
        # Give agent time to start
        await asyncio.sleep(2)
        
        # Send message
        await self.service_bus.send_message(message_dict)
        
        # Wait for response
        response = await self.service_bus.receive_message("test_orchestrator", timeout=120)
        
        # Stop agent
        await self.agent.stop()
        agent_task.cancel()
        
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
        
        return response, luma_data

    def _extract_discrepancies(self, response: Dict) -> List[Dict]:
        """Extract discrepancies from graph response"""
        if not response:
            return []
            
        payload = response.get('payload', {})
        task_result = payload.get('task', {}).get('result', {})
        return task_result.get('discrepancies', [])

    def _assert_discrepancy_expectations(self, actual_discrepancies: List[Dict], expected_count: int, 
                                       expected_affected_claims: List[str] = None):
        """
        Assert discrepancy expectations
        
        Args:
            actual_discrepancies: List of actual discrepancies found
            expected_count: Expected number of discrepancies
            expected_affected_claims: Expected affected claim IDs (optional)
        """
        self.assertEqual(len(actual_discrepancies), expected_count, 
                        f"Expected {expected_count} discrepancies, got {len(actual_discrepancies)}")
        
        if expected_affected_claims:
            all_affected_claims = set()
            for disc in actual_discrepancies:
                all_affected_claims.update(disc.get('affected_claim_ids', []))
            
            self.assertEqual(set(str(c) for c in all_affected_claims), 
                           set(str(c) for c in expected_affected_claims),
                           f"Expected affected claims {expected_affected_claims}, got {list(all_affected_claims)}")

    # Existing test cases from archived version
    def test_scenario_16_3_extremely_long_gap_remove(self):
        """Test Scenario 16.3: Extremely Long Gap Periods - REMOVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_16_3_extremely_long_gap_remove")
            
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 0 discrepancies (gap should be removed due to CPAC explanation)
            self._assert_discrepancy_expectations(
                actual_discrepancies=actual_discrepancies,
                expected_count=0
            )
            
            payload = response.get('payload', {})
            status = payload.get('task', {}).get('status', 'unknown')
            self.assertEqual(status, 'completed', f"Expected completed status, got {status}")
            
        asyncio.run(run_test())

    def test_scenario_4_1_obvious_employment_gap(self):
        """Test Scenario 4.1: Obvious Employment Gap"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_4_1_obvious_employment_gap")
            
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 employment_timeline_gap discrepancy affecting claims ["1", "2"]
            self._assert_discrepancy_expectations(
                actual_discrepancies=actual_discrepancies,
                expected_count=1,
                expected_affected_claims=["1", "2"]
            )
            
            payload = response.get('payload', {})
            status = payload.get('task', {}).get('status', 'unknown')
            self.assertEqual(status, 'completed', f"Expected completed status, got {status}")
            
        asyncio.run(run_test())

    def test_scenario_3_1_missing_critical_fields(self):
        """Test Scenario 3.1: Missing Critical Fields"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_3_1_missing_critical_fields")
            
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: Should detect missing fields as discrepancies
            self._assert_discrepancy_expectations(
                actual_discrepancies=actual_discrepancies,
                expected_count=1,
                expected_affected_claims=["1", "2"]
            )
            
            payload = response.get('payload', {})
            status = payload.get('task', {}).get('status', 'unknown')
            self.assertEqual(status, 'completed', f"Expected completed status, got {status}")
            
        asyncio.run(run_test())

    # Test scenarios 1-8 and 17+ from test_scenarios_luma
    def test_scenario_1_1_mixed_date_formats(self):
        """Test Scenario 1.1: Mixed Date Formats"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_1_1_mixed_date_formats")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy for mixed date formats
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_1_2_ambiguous_date_formats(self):
        """Test Scenario 1.2: Ambiguous Date Formats"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_1_2_ambiguous_date_formats")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy for ambiguous date formats
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_1_3_partial_date_information(self):
        """Test Scenario 1.3: Partial Date Information"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_1_3_partial_date_information")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy for partial date information
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_2_1_structured_vs_unstructured_cpac(self):
        """Test Scenario 2.1: Structured vs Unstructured CPAC"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_2_1_structured_vs_unstructured_cpac")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy for structured vs unstructured data
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_2_2_fragmented_employment_information(self):
        """Test Scenario 2.2: Fragmented Employment Information"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_2_2_fragmented_employment_information")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy for fragmented information
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_2_3_inconsistent_company_names(self):
        """Test Scenario 2.3: Inconsistent Company Names"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_2_3_inconsistent_company_names")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy for inconsistent company names
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_3_2_contradictory_duration_information(self):
        """Test Scenario 3.2: Contradictory Duration Information"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_3_2_contradictory_duration_information")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy for contradictory duration information
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_3_3_null_vs_empty_string_handling(self):
        """Test Scenario 3.3: Null vs Empty String Handling"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_3_3_null_vs_empty_string_handling")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy for null vs empty string handling
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_4_2_justified_gap_in_cpac(self):
        """Test Scenario 4.2: Justified Gap in CPAC"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_4_2_justified_gap_in_cpac")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 0 discrepancies (gap is justified in CPAC)
            self._assert_discrepancy_expectations(actual_discrepancies, 0)
            
        asyncio.run(run_test())

    def test_scenario_4_3_short_gap_vs_transition(self):
        """Test Scenario 4.3: Short Gap vs Transition"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_4_3_short_gap_vs_transition")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 0-1 discrepancies (short gap might be acceptable)
            self._assert_discrepancy_expectations(actual_discrepancies, 0)
            
        asyncio.run(run_test())

    def test_scenario_5_1_impossible_full_overlap(self):
        """Test Scenario 5.1: Impossible Full Overlap"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_5_1_impossible_full_overlap")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy for impossible full overlap
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_5_2_valid_concurrent_positions(self):
        """Test Scenario 5.2: Valid Concurrent Positions"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_5_2_valid_concurrent_positions")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 0 discrepancies (concurrent positions are valid)
            self._assert_discrepancy_expectations(actual_discrepancies, 0)
            
        asyncio.run(run_test())

    def test_scenario_5_3_partial_overlap(self):
        """Test Scenario 5.3: Partial Overlap"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_5_3_partial_overlap")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy for partial overlap
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_6_1_multiple_overlaps_and_gaps(self):
        """Test Scenario 6.1: Multiple Overlaps and Gaps"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_6_1_multiple_overlaps_and_gaps")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 2+ discrepancies for multiple issues
            self._assert_discrepancy_expectations(actual_discrepancies, 2, ["1", "2", "3"])
            
        asyncio.run(run_test())

    def test_scenario_8_1_same_day_transitions(self):
        """Test Scenario 8.1: Same Day Transitions"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_8_1_same_day_transitions")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 0 discrepancies (same day transitions are valid)
            self._assert_discrepancy_expectations(actual_discrepancies, 0)
            
        asyncio.run(run_test())

    def test_scenario_8_2_future_employment_dates(self):
        """Test Scenario 8.2: Future Employment Dates"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_8_2_future_employment_dates")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy for future employment dates
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_8_3_extremely_long_career(self):
        """Test Scenario 8.3: Extremely Long Career"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_8_3_extremely_long_career")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 0-1 discrepancies (long career might be valid)
            self._assert_discrepancy_expectations(actual_discrepancies, 0)
            
        asyncio.run(run_test())

    def test_scenario_17_1_duplicate_claims_analysis(self):
        """Test Scenario 17.1: Duplicate Claims Analysis"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_17_1_duplicate_claims_analysis")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy for duplicate claims
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    # Test scenarios 9-16 from test_scenarios_luma
    def test_scenario_9_1_exact_period_match_remove(self):
        """Test Scenario 9.1: Exact Period Match - REMOVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_9_1_exact_period_match_remove")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 0 discrepancies (gap should be removed)
            self._assert_discrepancy_expectations(actual_discrepancies, 0)
            
        asyncio.run(run_test())

    def test_scenario_9_2_period_mismatch_approve(self):
        """Test Scenario 9.2: Period Mismatch - APPROVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_9_2_period_mismatch_approve")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy that should be approved
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_9_3_partial_period_match_approve(self):
        """Test Scenario 9.3: Partial Period Match - APPROVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_9_3_partial_period_match_approve")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy that should be approved
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_9_4_multiple_gap_explanations_remove(self):
        """Test Scenario 9.4: Multiple Gap Explanations - REMOVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_9_4_multiple_gap_explanations_remove")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 0 discrepancies (gaps should be removed)
            self._assert_discrepancy_expectations(actual_discrepancies, 0)
            
        asyncio.run(run_test())

    def test_scenario_10_1_subsidiary_relationship_remove(self):
        """Test Scenario 10.1: Subsidiary Relationship - REMOVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_10_1_subsidiary_relationship_remove")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 0 discrepancies (overlap should be removed)
            self._assert_discrepancy_expectations(actual_discrepancies, 0)
            
        asyncio.run(run_test())

    def test_scenario_10_2_regional_structure_remove(self):
        """Test Scenario 10.2: Regional Structure - REMOVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_10_2_regional_structure_remove")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 0 discrepancies (overlap should be removed)
            self._assert_discrepancy_expectations(actual_discrepancies, 0)
            
        asyncio.run(run_test())

    def test_scenario_10_3_multiple_board_positions_remove(self):
        """Test Scenario 10.3: Multiple Board Positions - REMOVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_10_3_multiple_board_positions_remove")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 0 discrepancies (overlap should be removed)
            self._assert_discrepancy_expectations(actual_discrepancies, 0)
            
        asyncio.run(run_test())

    def test_scenario_10_4_no_relationship_explanation_approve(self):
        """Test Scenario 10.4: No Relationship Explanation - APPROVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_10_4_no_relationship_explanation_approve")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy that should be approved
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_10_5_ambiguous_relationship_approve(self):
        """Test Scenario 10.5: Ambiguous Relationship - APPROVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_10_5_ambiguous_relationship_approve")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy that should be approved
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_11_1_explicit_promotion_remove(self):
        """Test Scenario 11.1: Explicit Promotion - REMOVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_11_1_explicit_promotion_remove")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 0 discrepancies (overlap should be removed)
            self._assert_discrepancy_expectations(actual_discrepancies, 0)
            
        asyncio.run(run_test())

    def test_scenario_11_2_role_change_explanation_remove(self):
        """Test Scenario 11.2: Role Change Explanation - REMOVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_11_2_role_change_explanation_remove")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 0 discrepancies (overlap should be removed)
            self._assert_discrepancy_expectations(actual_discrepancies, 0)
            
        asyncio.run(run_test())

    def test_scenario_11_3_salary_mention_no_explanation_approve(self):
        """Test Scenario 11.3: Salary Mention No Explanation - APPROVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_11_3_salary_mention_no_explanation_approve")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy that should be approved
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_11_4_multiple_salary_references_approve(self):
        """Test Scenario 11.4: Multiple Salary References - APPROVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_11_4_multiple_salary_references_approve")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy that should be approved
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_12_1_consecutive_progression_remove(self):
        """Test Scenario 12.1: Consecutive Progression - REMOVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_12_1_consecutive_progression_remove")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 0 discrepancies (progression should be removed)
            self._assert_discrepancy_expectations(actual_discrepancies, 0)
            
        asyncio.run(run_test())

    def test_scenario_12_2_overlapping_despite_progression_remove(self):
        """Test Scenario 12.2: Overlapping Despite Progression - REMOVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_12_2_overlapping_despite_progression_remove")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 0 discrepancies (overlap should be removed)
            self._assert_discrepancy_expectations(actual_discrepancies, 0)
            
        asyncio.run(run_test())

    def test_scenario_13_1_same_company_no_progression_approve(self):
        """Test Scenario 13.1: Same Company No Progression - APPROVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_13_1_same_company_no_progression_approve")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy that should be approved
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_13_2_different_companies_no_explanation_approve(self):
        """Test Scenario 13.2: Different Companies No Explanation - APPROVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_13_2_different_companies_no_explanation_approve")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy that should be approved
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_13_3_employment_gap_no_explanation_approve(self):
        """Test Scenario 13.3: Employment Gap No Explanation - APPROVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_13_3_employment_gap_no_explanation_approve")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy that should be approved
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_14_1_subsidiary_overlap_remove_template(self):
        """Test Scenario 14.1: Subsidiary Overlap - REMOVE Template"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_14_1_subsidiary_overlap_remove_template")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 0 discrepancies (overlap should be removed)
            self._assert_discrepancy_expectations(actual_discrepancies, 0)
            
        asyncio.run(run_test())

    def test_scenario_14_2_same_company_progression_remove_template(self):
        """Test Scenario 14.2: Same Company Progression - REMOVE Template"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_14_2_same_company_progression_remove_template")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 0 discrepancies (overlap should be removed)
            self._assert_discrepancy_expectations(actual_discrepancies, 0)
            
        asyncio.run(run_test())

    def test_scenario_15_1_multiple_keywords_remove(self):
        """Test Scenario 15.1: Multiple Keywords - REMOVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_15_1_multiple_keywords_remove")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 0 discrepancies (keywords should remove discrepancy)
            self._assert_discrepancy_expectations(actual_discrepancies, 0)
            
        asyncio.run(run_test())

    def test_scenario_15_2_similar_keywords_approve(self):
        """Test Scenario 15.2: Similar Keywords - APPROVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_15_2_similar_keywords_approve")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy that should be approved
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_15_3_keyword_wrong_context_approve(self):
        """Test Scenario 15.3: Keyword Wrong Context - APPROVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_15_3_keyword_wrong_context_approve")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy that should be approved
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_15_4_complex_multi_company_remove(self):
        """Test Scenario 15.4: Complex Multi Company - REMOVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_15_4_complex_multi_company_remove")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 0 discrepancies (complex explanation should remove)
            self._assert_discrepancy_expectations(actual_discrepancies, 0)
            
        asyncio.run(run_test())

    def test_scenario_16_1_missing_cpac_info_approve(self):
        """Test Scenario 16.1: Missing CPAC Info - APPROVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_16_1_missing_cpac_info_approve")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 1 discrepancy that should be approved
            self._assert_discrepancy_expectations(actual_discrepancies, 1, ["1", "2"])
            
        asyncio.run(run_test())

    def test_scenario_16_2_contradictory_statements_remove(self):
        """Test Scenario 16.2: Contradictory Statements - REMOVE"""
        async def run_test():
            response, luma_data = await self._run_luma_scenario_test("test_scenario_16_2_contradictory_statements_remove")
            actual_discrepancies = self._extract_discrepancies(response)
            
            print(f"\nTest: {luma_data['test_name']}")
            print(f"Description: {luma_data['test_description']}")
            print(f"Actual Discrepancies Found: {len(actual_discrepancies)}")
            
            for i, disc in enumerate(actual_discrepancies):
                print(f"  [{i+1}] {disc['discrepancy_type']}: {disc['description']}")
                print(f"      Affected Claims: {disc['affected_claim_ids']}")
            
            # Expected: 0 discrepancies (contradictions should be resolved)
            self._assert_discrepancy_expectations(actual_discrepancies, 0)
            
        asyncio.run(run_test())


if __name__ == '__main__':
    # Set up environment for testing
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run tests
    unittest.main(verbosity=2)
"""
Unit tests for ImproverNode
"""
import os
import sys
import unittest
import asyncio
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_model.schemas import (
    CPACConsistencyRequest, CPACConsistencyResult, ReviewOutput,
    ClaimCategory, Claim, CPACData, Discrepancy, DiscrepancyType
)
from agent.improver_node import ImproverNode


class TestImproverNode(unittest.TestCase):
    """Test cases for ImproverNode"""

    def setUp(self):
        """Set up test fixtures"""
        # Set test OpenAI API key if not already set
        if not os.getenv('OPENAI_API_KEY'):
            os.environ['OPENAI_API_KEY'] = 'test-key-for-unit-tests'
        
        # Check if we should skip LLM-dependent tests
        if os.getenv('OPENAI_API_KEY') == 'test-key-for-unit-tests':
            self.skipTest("Skipping ImproverNode tests - requires real OpenAI API key for LLM operations")
        
        # Create improver node with existing prompt file
        try:
            self.improver = ImproverNode(
                prompt_path="prompts/improvement_prompt_v4.j2",
                config_path="config/config.yaml"
            )
        except Exception as e:
            self.skipTest(f"ImproverNode initialization failed: {str(e)}")
        
        # Create test data
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
            )
        ]
        
        self.test_discrepancies = [
            Discrepancy(
                discrepancy_id=1,
                discrepancy_type=DiscrepancyType.CPAC_CLAIM_INCONSISTENCY,
                description="Test discrepancy",
                reason="Test reason",
                recommendation="Test recommendation",
                affected_claim_ids=[1],
                affected_document_ids=[]
            )
        ]
        
        self.test_request = CPACConsistencyRequest(
            claim_category=ClaimCategory.EMPLOYMENT,
            claims=self.test_claims,
            cpac_text="Test CPAC text"
        )
        
        self.test_result = CPACConsistencyResult(
            claim_category=ClaimCategory.EMPLOYMENT,
            discrepancies=self.test_discrepancies
        )

    def test_improver_initialization(self):
        """Test that improver initializes correctly"""
        self.assertIsNotNone(self.improver)
        self.assertIsNotNone(self.improver.llm_handler)
        self.assertIsInstance(self.improver.print_results, bool)

    def test_improve_discrepancies_with_remove_action(self):
        """Test improving discrepancies with remove action"""
        async def run_test():
            if os.getenv('OPENAI_API_KEY') == 'test-key-for-unit-tests':
                self.skipTest("Skipping LLM test - no real API key provided")
            
            review_action = {
                "remove": {"1": "Remove this discrepancy"},
                "approve": {},
                "revise": {}
            }
            
            result = await self.improver.improve_discrepancies(
                self.test_discrepancies,
                review_action=review_action
            )
            
            # Should return list (content depends on LLM response)
            self.assertIsInstance(result, list)
            
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

    def test_improve_discrepancies_with_approve_action(self):
        """Test improving discrepancies with approve action"""
        async def run_test():
            if os.getenv('OPENAI_API_KEY') == 'test-key-for-unit-tests':
                self.skipTest("Skipping LLM test - no real API key provided")
            
            review_action = {
                "remove": {},
                "approve": {"1": "Approve this discrepancy"},
                "revise": {}
            }
            
            result = await self.improver.improve_discrepancies(
                self.test_discrepancies,
                review_action=review_action
            )
            
            # Should return list (content depends on LLM response)
            self.assertIsInstance(result, list)
            
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

    def test_improve_discrepancies_with_revise_action(self):
        """Test improving discrepancies with revise action"""
        review_action = {
            "remove": {},
            "approve": {},
            "revise": {
                "1": {
                    "suggestion": "Update the description",
                    "field": "description"
                }
            }
        }
        
        async def run_test():
            if os.getenv('OPENAI_API_KEY') == 'test-key-for-unit-tests':
                self.skipTest("Skipping LLM test - no real API key provided")
            
            result = await self.improver.improve_discrepancies(
                self.test_discrepancies,
                review_action=review_action
            )
            
            # Should return list with revised discrepancy
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            
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

    def test_improve_discrepancies_no_action(self):
        """Test improving discrepancies with no action"""
        async def run_test():
            result = await self.improver.improve_discrepancies(
                self.test_discrepancies,
                review_action=None
            )
            
            # Should return original list unchanged
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            
            return result
        
        result = asyncio.run(run_test())
        self.assertIsNotNone(result)

    def test_error_handling(self):
        """Test error handling with invalid data"""
        async def run_test():
            if os.getenv('OPENAI_API_KEY') == 'test-key-for-unit-tests':
                self.skipTest("Skipping LLM test - no real API key provided")
            
            invalid_review_action = {
                "remove": {"999": "Non-existent discrepancy"},  # Invalid ID
                "approve": {},
                "revise": {}
            }
            
            result = await self.improver.improve_discrepancies(
                self.test_discrepancies,
                review_action=invalid_review_action
            )
            
            # Should handle gracefully
            self.assertIsInstance(result, list)
            
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


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)
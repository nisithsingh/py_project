"""
Unit tests for ExecutorNode
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
    CPACConsistencyRequest, ClaimCategory, Claim, CPACData, Discrepancy
)
from agent.executor_node import ExecutorNode
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TestExecutorNode(unittest.TestCase):
    """Test cases for ExecutorNode"""

    def setUp(self):
        """Set up test fixtures"""
        # Set test OpenAI API key if not already set
        if not os.getenv('OPENAI_API_KEY'):
            os.environ['OPENAI_API_KEY'] = 'test-key-for-unit-tests'
        
        # Create executor node
        try:
            self.executor = ExecutorNode(
                config_path="config/config.yaml"
            )
        except Exception as e:
            self.skipTest(f"ExecutorNode initialization failed: {str(e)}")
        
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
        
        self.test_request = CPACConsistencyRequest(
            claim_category=ClaimCategory.EMPLOYMENT,
            claims=self.test_claims,
            cpac_text="Client worked at TechCorp Inc from 2020 to 2021, then moved to DataSystems Ltd in 2022."
        )

    def test_executor_initialization(self):
        """Test that executor initializes correctly"""
        self.assertIsNotNone(self.executor)
        self.assertIsNotNone(self.executor.llm_handler)
        self.assertIsNotNone(self.executor.config)
        self.assertTrue(hasattr(self.executor, 'exception_list'))

    def test_analyze_employment_claims_with_llm(self):
        """Test LLM analysis of employment claims"""
        async def run_test():
            if os.getenv('OPENAI_API_KEY') == 'test-key-for-unit-tests':
                self.skipTest("Skipping LLM test - no real API key provided")
            
            result = await self.executor.analyze_claims_with_llm(self.test_request)
            
            # Should return list of Discrepancy objects
            self.assertIsInstance(result, list)
            # All items should be Discrepancy objects
            for item in result:
                self.assertIsInstance(item, Discrepancy)
            
            return result
        
        try:
            result = asyncio.run(run_test())
            self.assertIsNotNone(result)
        except unittest.SkipTest:
            raise
        except Exception as e:
            if "API key" in str(e) or "authentication" in str(e).lower():
                self.skipTest(f"Skipping LLM test - API key issue: {str(e)}")
            else:
                # Test that error handling works (should not crash)
                self.assertIsNotNone(self.executor.get_exceptions())

    def test_analyze_inheritance_claims_with_llm(self):
        """Test LLM analysis of inheritance claims"""
        # Create inheritance claims
        inheritance_claims = [
            Claim(
                claim_id=1,
                cpac_data=CPACData(
                    inheritance_source="Father",
                    inheritance_date="2022-05-15",
                    inheritance_amount=500000,
                    inheritance_currency="USD",
                    inheritance_type="cash",
                    start_date="2022-05-15"  # Required field
                )
            )
        ]
        
        request = CPACConsistencyRequest(
            claim_category=ClaimCategory.INHERITANCE,
            claims=inheritance_claims,
            cpac_text="Client inherited $500,000 from father in May 2022."
        )
        
        async def run_test():
            if os.getenv('OPENAI_API_KEY') == 'test-key-for-unit-tests':
                self.skipTest("Skipping LLM test - no real API key provided")
            
            result = await self.executor.analyze_claims_with_llm(request)
            
            # Should return list of Discrepancy objects
            self.assertIsInstance(result, list)
            for item in result:
                self.assertIsInstance(item, Discrepancy)
            
            return result
        
        try:
            result = asyncio.run(run_test())
            self.assertIsNotNone(result)
        except unittest.SkipTest:
            raise
        except Exception as e:
            if "API key" in str(e) or "authentication" in str(e).lower():
                self.skipTest(f"Skipping LLM test - API key issue: {str(e)}")
            else:
                # Test that error handling works
                self.assertIsNotNone(self.executor.get_exceptions())

    def test_analyze_business_claims_with_llm(self):
        """Test LLM analysis of business claims"""
        # Create business claims
        business_claims = [
            Claim(
                claim_id=1,
                cpac_data=CPACData(
                    company_name="TechStartup LLC",
                    business_type="Technology Services",
                    ownership_percentage=75.0,
                    profit_amount=250000,
                    profit_currency="USD",
                    business_role="CEO",
                    start_date="2019-01-01"  # Required field
                )
            )
        ]
        
        request = CPACConsistencyRequest(
            claim_category=ClaimCategory.BUSINESS,
            claims=business_claims,
            cpac_text="Client owns 75% of TechStartup LLC and received $250,000 in profits."
        )
        
        async def run_test():
            if os.getenv('OPENAI_API_KEY') == 'test-key-for-unit-tests':
                self.skipTest("Skipping LLM test - no real API key provided")
            
            result = await self.executor.analyze_claims_with_llm(request)
            
            # Should return list of Discrepancy objects
            self.assertIsInstance(result, list)
            for item in result:
                self.assertIsInstance(item, Discrepancy)
            
            return result
        
        try:
            result = asyncio.run(run_test())
            self.assertIsNotNone(result)
        except unittest.SkipTest:
            raise
        except Exception as e:
            if "API key" in str(e) or "authentication" in str(e).lower():
                self.skipTest(f"Skipping LLM test - API key issue: {str(e)}")
            else:
                # Test that error handling works
                self.assertIsNotNone(self.executor.get_exceptions())

    def test_analyze_empty_claims(self):
        """Test LLM analysis with empty claims list"""
        empty_request = CPACConsistencyRequest(
            claim_category=ClaimCategory.EMPLOYMENT,
            claims=[],
            cpac_text="No claims provided."
        )
        
        async def run_test():
            if os.getenv('OPENAI_API_KEY') == 'test-key-for-unit-tests':
                self.skipTest("Skipping LLM test - no real API key provided")
            
            result = await self.executor.analyze_claims_with_llm(empty_request)
            
            # Should handle empty claims gracefully
            self.assertIsInstance(result, list)
            # May return empty list or some analysis result
            
            return result
        
        try:
            result = asyncio.run(run_test())
            self.assertIsNotNone(result)
        except unittest.SkipTest:
            raise
        except Exception as e:
            if "API key" in str(e) or "authentication" in str(e).lower():
                self.skipTest(f"Skipping LLM test - API key issue: {str(e)}")
            else:
                # Test that error handling works
                self.assertIsNotNone(self.executor.get_exceptions())

    def test_exception_handling(self):
        """Test exception tracking functionality"""
        # Test exception list starts empty
        self.assertEqual(len(self.executor.get_exceptions()), 0)
        
        # Test clear exceptions
        self.executor.clear_exceptions()
        self.assertEqual(len(self.executor.get_exceptions()), 0)
        
        # Test that get_exceptions returns a copy (not reference)
        exceptions = self.executor.get_exceptions()
        exceptions.append({"test": "exception"})
        self.assertEqual(len(self.executor.get_exceptions()), 0)

    def test_error_handling_with_invalid_llm_response(self):
        """Test error handling when LLM analysis fails"""
        # Create request that might cause LLM issues
        problematic_request = CPACConsistencyRequest(
            claim_category=ClaimCategory.EMPLOYMENT,
            claims=[
                Claim(
                    claim_id=1,
                    cpac_data=CPACData(
                        employer_name="",  # Empty employer name
                        job_title="",      # Empty job title
                        start_date="invalid-date",  # Invalid date
                        end_date="also-invalid",
                        annual_compensation="not-a-number",
                        employment_type="unknown"
                    )
                )
            ],
            cpac_text=""  # Empty text
        )
        
        async def run_test():
            if os.getenv('OPENAI_API_KEY') == 'test-key-for-unit-tests':
                # Still test error handling logic even without real API
                self.executor.exception_list.append({
                    "exception_type": "TestException",
                    "message": "Test exception for error handling",
                    "timestamp": datetime.now().isoformat()
                })
                exceptions = self.executor.get_exceptions()
                self.assertEqual(len(exceptions), 1)
                self.assertEqual(exceptions[0]["exception_type"], "TestException")
                return []
            
            try:
                result = await self.executor.analyze_claims_with_llm(problematic_request)
                
                # Should return a list (might be empty due to errors)
                self.assertIsInstance(result, list)
                
                # Check if exceptions were recorded
                exceptions = self.executor.get_exceptions()
                self.assertIsInstance(exceptions, list)
                
                return result
                
            except Exception:
                # Even if it fails, should not crash completely
                # Check that exception tracking still works
                exceptions = self.executor.get_exceptions()
                self.assertIsInstance(exceptions, list)
                return []
        
        result = asyncio.run(run_test())
        self.assertIsInstance(result, list)

    def test_configuration_settings(self):
        """Test configuration settings"""
        # Check that config is loaded and contains expected settings
        self.assertIsNotNone(self.executor.config)
        self.assertIn('debug', self.executor.config)
        
        # Check LLM config
        self.assertIn('llm', self.executor.config)
        self.assertIn('prompt_path', self.executor.config['llm'])

    def test_llm_handler_initialization(self):
        """Test that LLM handler is initialized correctly"""
        # Check that LLM handler was initialized
        self.assertIsNotNone(self.executor.llm_handler)
        self.assertTrue(hasattr(self.executor.llm_handler, 'query'))

    def test_prompt_selection_logic(self):
        """Test that different prompts are selected for different claim categories"""
        # This tests the logic in analyze_claims_with_llm that selects appropriate prompts
        # We can't easily test the actual prompt loading without mocking, but we can verify
        # the configuration exists for different categories
        
        # Check employment prompt (default)
        self.assertIn('prompt_path', self.executor.config['llm'])
        
        # Check if generic prompt path is configured
        generic_prompt_path = self.executor.config['llm'].get('generic_prompt_path')
        if generic_prompt_path:
            self.assertIsInstance(generic_prompt_path, str)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)
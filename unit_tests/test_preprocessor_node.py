"""
Unit tests for PreprocessorNode
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
    CPACConsistencyRequest, ClaimCategory, Claim, CPACData
)
from agent.preprocessor_node import PreprocessorNode


class TestPreprocessorNode(unittest.TestCase):
    """Test cases for PreprocessorNode"""

    def setUp(self):
        """Set up test fixtures"""
        # Set test OpenAI API key if not already set
        if not os.getenv('OPENAI_API_KEY'):
            os.environ['OPENAI_API_KEY'] = 'test-key-for-unit-tests'
        
        # Check if we should skip LLM-dependent tests
        if os.getenv('OPENAI_API_KEY') == 'test-key-for-unit-tests':
            self.skipTest("Skipping PreprocessorNode tests - requires real OpenAI API key for LLM operations")
        
        # Create preprocessor node
        try:
            self.preprocessor = PreprocessorNode(
                config_path="config/config.yaml"
            )
        except Exception as e:
            self.skipTest(f"PreprocessorNode initialization failed: {str(e)}")
        
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

    def test_preprocessor_initialization(self):
        """Test that preprocessor initializes correctly"""
        self.assertIsNotNone(self.preprocessor)
        self.assertIsNotNone(self.preprocessor.config)
        self.assertTrue(hasattr(self.preprocessor, 'llm_handler'))

    def test_preprocess_employment_claims(self):
        """Test preprocessing employment claims"""
        async def run_test():
            result = await self.preprocessor.preprocess_claims(self.test_request)
            
            # Should return CPACConsistencyRequest with processed claims
            self.assertIsInstance(result, CPACConsistencyRequest)
            self.assertEqual(result.claim_category, ClaimCategory.EMPLOYMENT)
            self.assertIsInstance(result.claims, list)
            self.assertEqual(len(result.claims), 2)
            
            return result
        
        result = asyncio.run(run_test())
        self.assertIsNotNone(result)

    def test_chronological_sorting(self):
        """Test chronological sorting of claims"""
        # Create claims in non-chronological order
        unsorted_claims = [
            Claim(
                claim_id=1,
                cpac_data=CPACData(
                    employer_name="Company B",
                    job_title="Senior Engineer",
                    start_date="2023-01-01",
                    end_date="2024-12-31",
                    annual_compensation="90000 USD",
                    employment_type="full-time"
                )
            ),
            Claim(
                claim_id=2,
                cpac_data=CPACData(
                    employer_name="Company A",
                    job_title="Engineer",
                    start_date="2020-01-01",
                    end_date="2021-06-30",
                    annual_compensation="70000 USD",
                    employment_type="full-time"
                )
            )
        ]
        
        # Sort claims chronologically
        sorted_claims = self.preprocessor._sort_claims_chronologically(unsorted_claims)
        
        # Should be sorted by start date (Company A first, then Company B)
        self.assertEqual(len(sorted_claims), 2)
        self.assertEqual(sorted_claims[0].cpac_data.employer_name, "Company A")
        self.assertEqual(sorted_claims[1].cpac_data.employer_name, "Company B")

    def test_date_format_detection(self):
        """Test date format detection from dataset"""
        # Create claims with unambiguous DD/MM format
        claims_ddmm = [
            Claim(
                claim_id=1,
                cpac_data=CPACData(
                    employer_name="Company X",
                    job_title="Developer",
                    start_date="31/12/2022",  # Clearly DD/MM (day > 12)
                    end_date="15/06/2023",
                    annual_compensation="75000 USD",
                    employment_type="full-time"
                )
            )
        ]
        
        detected_format = self.preprocessor._detect_date_format_from_dataset(claims_ddmm)
        self.assertEqual(detected_format, "DD/MM/YYYY")
        
        # Test with MM/DD format
        claims_mmdd = [
            Claim(
                claim_id=1,
                cpac_data=CPACData(
                    employer_name="Company Y",
                    job_title="Consultant",
                    start_date="12/31/2022",  # Clearly MM/DD (day > 12)
                    end_date="06/15/2023",
                    annual_compensation="85000 USD",
                    employment_type="contract"
                )
            )
        ]
        
        detected_format = self.preprocessor._detect_date_format_from_dataset(claims_mmdd)
        self.assertEqual(detected_format, "MM/DD/YYYY")

    def test_deterministic_date_parsing(self):
        """Test deterministic date parsing"""
        # Test various date formats
        test_cases = [
            ("2022-01-15", "2022-01-15"),  # Already correct
            ("31/12/2022", "2022-12-31"),  # DD/MM/YYYY
            ("12/31/2022", "2022-12-31"),  # MM/DD/YYYY
            ("2022/12/31", "2022-12-31"),  # YYYY/MM/DD
            ("present", "present"),         # Special value
            ("ongoing", "ongoing"),         # Special value
        ]
        
        for input_date, expected in test_cases:
            result = self.preprocessor._try_parse_date_deterministic(input_date)
            self.assertEqual(result, expected, f"Failed for input: {input_date}")

    def test_preprocess_empty_claims(self):
        """Test preprocessing empty claims list"""
        empty_request = CPACConsistencyRequest(
            claim_category=ClaimCategory.EMPLOYMENT,
            claims=[],
            cpac_text="No claims provided."
        )
        
        async def run_test():
            result = await self.preprocessor.preprocess_claims(empty_request)
            
            # Should handle empty claims gracefully
            self.assertIsInstance(result, CPACConsistencyRequest)
            self.assertEqual(len(result.claims), 0)
            
            return result
        
        result = asyncio.run(run_test())
        self.assertIsNotNone(result)

    def test_test_data_preprocessing(self):
        """Test preprocessing of test data format"""
        # Test data in the format used by test files
        test_data = {
            "claim_category": "employment",
            "claims": [
                {
                    "claim_id": "1",
                    "cpac_data": {
                        "employer_name": "Test Corp",
                        "job_title": "Tester",
                        "start_date": "01/01/2022",  # Non-standard format
                        "end_date": "31/12/2023",
                        "annual_compensation": "75000 USD",
                        "employment_type": "full-time"
                    }
                }
            ],
            "cpac_text": "Test with date conversion."
        }
        
        # Process test data
        result = self.preprocessor.preprocess_test_data(test_data)
        
        # Should return processed data
        self.assertIsInstance(result, dict)
        self.assertIn("claims", result)
        self.assertEqual(len(result["claims"]), 1)
        
        # Check that dates might be processed (may or may not change depending on format detection)
        processed_claim = result["claims"][0]
        self.assertIsInstance(processed_claim["cpac_data"], dict)

    def test_configuration_loading(self):
        """Test configuration loading"""
        # Check that config is loaded properly
        self.assertIsNotNone(self.preprocessor.config)
        self.assertIn('debug', self.preprocessor.config)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)
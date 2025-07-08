"""
Unit tests for Pydantic model validation scenarios
Tests how the system handles invalid Pydantic format in request claims
Uses minimal mocking to test real Pydantic validation behavior
"""
import os
import sys
import unittest
from typing import Dict, Any
from pydantic import ValidationError

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_model.schemas import (
    CPACConsistencyRequest, ClaimCategory, Claim, CPACData, 
    DiscrepancyType, Discrepancy, CPACConsistencyResult
)


class TestPydanticModelValidation(unittest.TestCase):
    """Test Pydantic model validation with various invalid input scenarios"""

    def setUp(self):
        """Set up valid test data for comparison"""
        # Valid claim structure for reference
        self.valid_claim_data = {
            "claim_id": 1,
            "cpac_data": {
                "employer_name": "TechCorp Inc",
                "job_title": "Software Engineer",
                "start_date": "2020-01-01",
                "end_date": "2021-12-31",
                "annual_compensation": "80000 USD",
                "employment_type": "full-time"
            }
        }
        
        # Valid request for reference
        self.valid_request_data = {
            "claim_category": "employment",
            "claims": [self.valid_claim_data],
            "cpac_text": "Client worked at TechCorp Inc from 2020 to 2021."
        }

    def test_valid_request_baseline(self):
        """Test that valid data works correctly (baseline test)"""
        # This should work without any errors
        request = CPACConsistencyRequest(**self.valid_request_data)
        
        # Verify the request was created correctly
        self.assertEqual(request.claim_category, ClaimCategory.EMPLOYMENT)
        self.assertEqual(len(request.claims), 1)
        self.assertEqual(request.claims[0].claim_id, 1)
        self.assertEqual(request.claims[0].cpac_data.employer_name, "TechCorp Inc")

    def test_scenario_1_invalid_claim_structure(self):
        """
        Scenario 1: Invalid Claim Structure
        Test claims with missing required fields (claim_id, cpac_data)
        """
        invalid_request_data = {
            "claim_category": "employment",
            "claims": [
                {
                    "invalid_field": "value",  # Missing claim_id, cpac_data
                    "wrong_structure": True
                }
            ],
            "cpac_text": "Invalid claim structure test"
        }
        
        # Should raise ValidationError
        with self.assertRaises(ValidationError) as context:
            CPACConsistencyRequest(**invalid_request_data)
        
        # Verify the error contains information about missing fields
        error_str = str(context.exception)
        self.assertIn("claim_id", error_str)  # Should mention missing claim_id
        self.assertIn("cpac_data", error_str)  # Should mention missing cpac_data

    def test_scenario_1_multiple_invalid_claims(self):
        """Test multiple claims with invalid structure"""
        invalid_request_data = {
            "claim_category": "employment",
            "claims": [
                {
                    "random_field": "value1"  # Missing required fields
                },
                {
                    "another_field": "value2"  # Missing required fields
                },
                {
                    "claim_id": 3,  # Has claim_id but missing cpac_data
                    "extra_field": "value3"
                }
            ],
            "cpac_text": "Multiple invalid claims test"
        }
        
        with self.assertRaises(ValidationError) as context:
            CPACConsistencyRequest(**invalid_request_data)
        
        # Should capture errors for multiple claims
        error_str = str(context.exception)
        self.assertIn("claim_id", error_str)
        self.assertIn("cpac_data", error_str)

    def test_scenario_2_invalid_cpac_data_fields(self):
        """
        Scenario 2: Invalid CPACData Fields
        Test claims with wrong data types and missing required fields
        """
        invalid_request_data = {
            "claim_category": "employment",
            "claims": [
                {
                    "claim_id": "not_an_integer",  # Should be int
                    "cpac_data": {
                        "invalid_date": "not-a-date",  # Missing required start_date
                        "unknown_field": "value",
                        "employer_name": 123,  # Should be string if provided
                    }
                }
            ],
            "cpac_text": "Invalid CPACData fields test"
        }
        
        with self.assertRaises(ValidationError) as context:
            CPACConsistencyRequest(**invalid_request_data)
        
        error_str = str(context.exception)
        # Should mention issues with claim_id type
        self.assertTrue(
            "claim_id" in error_str or "int" in error_str.lower(),
            f"Expected claim_id validation error in: {error_str}"
        )

    def test_scenario_2_missing_required_start_date(self):
        """Test CPACData missing required start_date field"""
        invalid_request_data = {
            "claim_category": "employment", 
            "claims": [
                {
                    "claim_id": 1,
                    "cpac_data": {
                        "employer_name": "TechCorp",
                        "job_title": "Engineer",
                        # Missing required start_date
                        "end_date": "2021-12-31",
                        "annual_compensation": "80000 USD"
                    }
                }
            ],
            "cpac_text": "Missing start_date test"
        }
        
        with self.assertRaises(ValidationError) as context:
            CPACConsistencyRequest(**invalid_request_data)
        
        error_str = str(context.exception)
        self.assertIn("start_date", error_str)  # Should mention missing start_date

    def test_scenario_3_type_mismatches(self):
        """
        Scenario 3: Type Mismatches
        Test completely wrong types for main fields
        """
        invalid_request_data = {
            "claim_category": 123,  # Should be string
            "claims": "not_a_list",  # Should be array
            "cpac_text": ["should", "be", "string"]  # Should be string
        }
        
        with self.assertRaises(ValidationError) as context:
            CPACConsistencyRequest(**invalid_request_data)
        
        error_str = str(context.exception)
        # Should mention issues with claims type (expecting list)
        self.assertTrue(
            "claims" in error_str or "list" in error_str.lower(),
            f"Expected claims validation error in: {error_str}"
        )

    def test_scenario_3_claims_as_dict(self):
        """Test claims field as dict instead of list"""
        invalid_request_data = {
            "claim_category": "employment",
            "claims": {  # Should be list, not dict
                "claim_1": {
                    "claim_id": 1,
                    "cpac_data": {"start_date": "2020-01-01"}
                }
            },
            "cpac_text": "Claims as dict test"
        }
        
        with self.assertRaises(ValidationError) as context:
            CPACConsistencyRequest(**invalid_request_data)
        
        error_str = str(context.exception)
        self.assertIn("claims", error_str.lower())

    def test_scenario_3_invalid_claim_category(self):
        """Test invalid claim category enum value"""
        invalid_request_data = {
            "claim_category": "invalid_category",  # Not in ClaimCategory enum
            "claims": [self.valid_claim_data],
            "cpac_text": "Invalid category test"
        }
        
        with self.assertRaises(ValidationError) as context:
            CPACConsistencyRequest(**invalid_request_data)
        
        error_str = str(context.exception)
        # Should mention the invalid enum value
        self.assertTrue(
            "claim_category" in error_str or "invalid_category" in error_str,
            f"Expected claim_category validation error in: {error_str}"
        )

    def test_mixed_valid_invalid_claims(self):
        """Test mix of valid and invalid claims in same request"""
        mixed_request_data = {
            "claim_category": "employment",
            "claims": [
                # Valid claim
                {
                    "claim_id": 1,
                    "cpac_data": {
                        "employer_name": "Valid Corp",
                        "start_date": "2020-01-01",
                        "end_date": "2021-12-31"
                    }
                },
                # Invalid claim (missing cpac_data)
                {
                    "claim_id": 2,
                    "invalid_field": "value"
                }
            ],
            "cpac_text": "Mixed valid/invalid claims test"
        }
        
        with self.assertRaises(ValidationError) as context:
            CPACConsistencyRequest(**invalid_request_data)
        
        # Should fail on the invalid claim even if first claim is valid
        error_str = str(context.exception)
        self.assertIn("cpac_data", error_str)

    def test_inheritance_claim_validation(self):
        """Test validation for inheritance category claims"""
        # Valid inheritance claim
        valid_inheritance_data = {
            "claim_category": "inheritance",
            "claims": [
                {
                    "claim_id": 1,
                    "cpac_data": {
                        "inheritance_source": "Father",
                        "inheritance_date": "2022-05-15",
                        "inheritance_amount": 500000,
                        "inheritance_currency": "USD",
                        "inheritance_type": "cash",
                        "start_date": "2022-05-15"  # Required field
                    }
                }
            ],
            "cpac_text": "Valid inheritance test"
        }
        
        # Should work without errors
        request = CPACConsistencyRequest(**valid_inheritance_data)
        self.assertEqual(request.claim_category, ClaimCategory.INHERITANCE)
        self.assertEqual(request.claims[0].cpac_data.inheritance_source, "Father")
        
        # Test invalid inheritance claim (missing start_date)
        invalid_inheritance_data = {
            "claim_category": "inheritance",
            "claims": [
                {
                    "claim_id": 1,
                    "cpac_data": {
                        "inheritance_source": "Father",
                        "inheritance_amount": 500000
                        # Missing required start_date
                    }
                }
            ],
            "cpac_text": "Invalid inheritance test"
        }
        
        with self.assertRaises(ValidationError):
            CPACConsistencyRequest(**invalid_inheritance_data)

    def test_business_claim_validation(self):
        """Test validation for business category claims"""
        # Valid business claim
        valid_business_data = {
            "claim_category": "business",
            "claims": [
                {
                    "claim_id": 1,
                    "cpac_data": {
                        "company_name": "TechStartup LLC",
                        "business_type": "Technology Services",
                        "ownership_percentage": 75.0,
                        "profit_amount": 250000,
                        "profit_currency": "USD",
                        "business_role": "CEO",
                        "start_date": "2019-01-01"  # Required field
                    }
                }
            ],
            "cpac_text": "Valid business test"
        }
        
        # Should work without errors
        request = CPACConsistencyRequest(**valid_business_data)
        self.assertEqual(request.claim_category, ClaimCategory.BUSINESS)
        self.assertEqual(request.claims[0].cpac_data.company_name, "TechStartup LLC")

    def test_discrepancy_model_validation(self):
        """Test Discrepancy model validation"""
        # Valid discrepancy
        valid_discrepancy_data = {
            "discrepancy_id": 1,
            "discrepancy_type": "employment_timeline_gap",
            "description": "Gap between employments",
            "reason": "6-month gap detected",
            "recommendation": "Investigate employment gap",
            "affected_claim_ids": [1, 2],
            "affected_document_ids": []
        }
        
        discrepancy = Discrepancy(**valid_discrepancy_data)
        self.assertEqual(discrepancy.discrepancy_type, DiscrepancyType.EMPLOYMENT_TIMELINE_GAP)
        
        # Invalid discrepancy (wrong enum value)
        invalid_discrepancy_data = {
            "discrepancy_id": 1,
            "discrepancy_type": "invalid_discrepancy_type",  # Invalid enum
            "description": "Test discrepancy",
            "reason": "",
            "recommendation": "",
            "affected_claim_ids": [1],
            "affected_document_ids": []
        }
        
        with self.assertRaises(ValidationError):
            Discrepancy(**invalid_discrepancy_data)

    def test_cpac_consistency_result_validation(self):
        """Test CPACConsistencyResult model validation"""
        # Valid result
        valid_result_data = {
            "claim_category": "employment",
            "discrepancies": []
        }
        
        result = CPACConsistencyResult(**valid_result_data)
        self.assertEqual(result.claim_category, ClaimCategory.EMPLOYMENT)
        self.assertEqual(len(result.discrepancies), 0)
        
        # Invalid result (wrong category type)
        invalid_result_data = {
            "claim_category": 123,  # Should be string
            "discrepancies": "not_a_list"  # Should be list
        }
        
        with self.assertRaises(ValidationError):
            CPACConsistencyResult(**invalid_result_data)

    def test_empty_and_none_values(self):
        """Test handling of empty and None values"""
        # Test empty claims list (should be valid)
        empty_claims_data = {
            "claim_category": "employment",
            "claims": [],  # Empty list should be valid
            "cpac_text": "No claims"
        }
        
        request = CPACConsistencyRequest(**empty_claims_data)
        self.assertEqual(len(request.claims), 0)
        
        # Test None values where not allowed
        none_category_data = {
            "claim_category": None,  # Should not be None
            "claims": [],
            "cpac_text": "None category test"
        }
        
        with self.assertRaises(ValidationError):
            CPACConsistencyRequest(**none_category_data)

    def test_extra_fields_handling(self):
        """Test how extra/unknown fields are handled"""
        # Pydantic by default allows extra fields in BaseModel
        # Test that extra fields don't break validation
        extra_fields_data = {
            "claim_category": "employment",
            "claims": [
                {
                    "claim_id": 1,
                    "cpac_data": {
                        "start_date": "2020-01-01",
                        "employer_name": "TechCorp",
                        "extra_unknown_field": "should_be_ignored"  # Extra field
                    },
                    "extra_claim_field": "also_ignored"  # Extra field
                }
            ],
            "cpac_text": "Extra fields test",
            "extra_request_field": "ignored"  # Extra field
        }
        
        # Should work - Pydantic allows extra fields by default
        request = CPACConsistencyRequest(**extra_fields_data)
        self.assertEqual(request.claim_category, ClaimCategory.EMPLOYMENT)
        self.assertEqual(len(request.claims), 1)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)
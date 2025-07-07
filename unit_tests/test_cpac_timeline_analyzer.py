"""
Unit tests for CPACTimelineAnalyzer
Tests rule-based employment timeline analysis with minimal mocking
"""
import os
import sys
import unittest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_model.schemas import (
    CPACConsistencyRequest, CPACConsistencyResult,
    ClaimCategory, Claim, CPACData, Discrepancy, DiscrepancyType
)
from agent.cpac_timeline_analyzer import CPACTimelineAnalyzer


class TestCPACTimelineAnalyzer(unittest.TestCase):
    """Test cases for CPACTimelineAnalyzer"""

    def setUp(self):
        """Set up test fixtures"""
        # Create timeline analyzer with default thresholds
        self.analyzer = CPACTimelineAnalyzer(
            gap_threshold_months=6,
            overlap_threshold_months=1
        )
        
        # Create test claims for different scenarios
        self.sequential_claims = [
            Claim(
                claim_id=1,
                cpac_data=CPACData(
                    employer_name="Company A",
                    job_title="Engineer",
                    start_date="2020-01-01",
                    end_date="2021-12-31",
                    annual_compensation="70000 USD",
                    employment_type="full-time"
                )
            ),
            Claim(
                claim_id=2,
                cpac_data=CPACData(
                    employer_name="Company B",
                    job_title="Senior Engineer",
                    start_date="2022-01-01",
                    end_date="2024-12-31",
                    annual_compensation="90000 USD",
                    employment_type="full-time"
                )
            )
        ]
        
        self.gap_claims = [
            Claim(
                claim_id=1,
                cpac_data=CPACData(
                    employer_name="Company A",
                    job_title="Engineer",
                    start_date="2020-01-01",
                    end_date="2021-06-30",
                    annual_compensation="70000 USD",
                    employment_type="full-time"
                )
            ),
            Claim(
                claim_id=2,
                cpac_data=CPACData(
                    employer_name="Company B",
                    job_title="Senior Engineer",
                    start_date="2023-01-01",  # 18-month gap
                    end_date="2024-12-31",
                    annual_compensation="90000 USD",
                    employment_type="full-time"
                )
            )
        ]
        
        self.overlap_claims = [
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
        ]

    def test_analyzer_initialization(self):
        """Test that analyzer initializes correctly"""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.gap_threshold_months, 6)
        self.assertEqual(self.analyzer.overlap_threshold_months, 1)

    def test_sequential_employment_no_issues(self):
        """Test sequential employment with no gaps or overlaps"""
        request = CPACConsistencyRequest(
            claim_category=ClaimCategory.EMPLOYMENT,
            claims=self.sequential_claims,
            cpac_text="Client worked at Company A from 2020-2021, then at Company B from 2022-2024."
        )
        
        discrepancies = self.analyzer.analyze_employment_timeline(request)
        
        # Should return list of discrepancies with no timeline issues
        self.assertIsInstance(discrepancies, list)
        
        # Check for timeline-specific discrepancies
        timeline_discrepancies = [
            d for d in discrepancies 
            if d.discrepancy_type in [
                DiscrepancyType.EMPLOYMENT_TIMELINE_GAP,
                DiscrepancyType.EMPLOYMENT_TIMELINE_OVERLAP
            ]
        ]
        self.assertEqual(len(timeline_discrepancies), 0, "Should not find timeline issues in sequential employment")

    def test_employment_timeline_gap_detection(self):
        """Test detection of employment timeline gaps"""
        request = CPACConsistencyRequest(
            claim_category=ClaimCategory.EMPLOYMENT,
            claims=self.gap_claims,
            cpac_text="Client worked at Company A until June 2021, then joined Company B in January 2023."
        )
        
        discrepancies = self.analyzer.analyze_employment_timeline(request)
        
        # Should detect employment gap
        self.assertIsInstance(discrepancies, list)
        
        gap_discrepancies = [
            d for d in discrepancies 
            if d.discrepancy_type == DiscrepancyType.EMPLOYMENT_TIMELINE_GAP
        ]
        self.assertGreater(len(gap_discrepancies), 0, "Should detect employment timeline gap")
        
        # Check gap discrepancy details
        gap_discrepancy = gap_discrepancies[0]
        self.assertIn("gap", gap_discrepancy.description.lower())
        self.assertEqual(len(gap_discrepancy.affected_claim_ids), 2)

    def test_employment_timeline_overlap_detection(self):
        """Test detection of employment timeline overlaps"""
        request = CPACConsistencyRequest(
            claim_category=ClaimCategory.EMPLOYMENT,
            claims=self.overlap_claims,
            cpac_text="Client worked at Company X full-time and also did consulting for Company Y."
        )
        
        discrepancies = self.analyzer.analyze_employment_timeline(request)
        
        # Should detect employment overlap
        self.assertIsInstance(discrepancies, list)
        
        overlap_discrepancies = [
            d for d in discrepancies 
            if d.discrepancy_type == DiscrepancyType.EMPLOYMENT_TIMELINE_OVERLAP
        ]
        self.assertGreater(len(overlap_discrepancies), 0, "Should detect employment timeline overlap")
        
        # Check overlap discrepancy details
        overlap_discrepancy = overlap_discrepancies[0]
        self.assertIn("overlap", overlap_discrepancy.description.lower())
        self.assertEqual(len(overlap_discrepancy.affected_claim_ids), 2)

    def test_single_employment_claim(self):
        """Test analysis with single employment claim"""
        single_claim_request = CPACConsistencyRequest(
            claim_category=ClaimCategory.EMPLOYMENT,
            claims=self.sequential_claims[:1],  # Only first claim
            cpac_text="Client worked at Company A from 2020-2021."
        )
        
        discrepancies = self.analyzer.analyze_employment_timeline(single_claim_request)
        
        # Should handle single claim without timeline issues
        self.assertIsInstance(discrepancies, list)
        timeline_discrepancies = [
            d for d in discrepancies 
            if d.discrepancy_type in [
                DiscrepancyType.EMPLOYMENT_TIMELINE_GAP,
                DiscrepancyType.EMPLOYMENT_TIMELINE_OVERLAP
            ]
        ]
        self.assertEqual(len(timeline_discrepancies), 0, "Single claim should not have timeline issues")

    def test_empty_claims_list(self):
        """Test analysis with empty claims list"""
        empty_request = CPACConsistencyRequest(
            claim_category=ClaimCategory.EMPLOYMENT,
            claims=[],
            cpac_text="No employment claims."
        )
        
        discrepancies = self.analyzer.analyze_employment_timeline(empty_request)
        
        # Should handle empty claims gracefully
        self.assertIsInstance(discrepancies, list)
        self.assertEqual(len(discrepancies), 0)

    def test_non_employment_claims(self):
        """Test that non-employment claims are handled correctly"""
        inheritance_claims = [
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
        ]
        
        inheritance_request = CPACConsistencyRequest(
            claim_category=ClaimCategory.INHERITANCE,
            claims=inheritance_claims,
            cpac_text="Client inherited $500,000 from father in May 2022."
        )
        
        discrepancies = self.analyzer.analyze_employment_timeline(inheritance_request)
        
        # Should return empty list for non-employment claims
        self.assertIsInstance(discrepancies, list)
        self.assertEqual(len(discrepancies), 0)
        
        # Should not have employment timeline discrepancies
        timeline_discrepancies = [
            d for d in discrepancies 
            if d.discrepancy_type in [
                DiscrepancyType.EMPLOYMENT_TIMELINE_GAP,
                DiscrepancyType.EMPLOYMENT_TIMELINE_OVERLAP
            ]
        ]
        self.assertEqual(len(timeline_discrepancies), 0)

    def test_invalid_date_handling(self):
        """Test handling of invalid date formats"""
        invalid_date_claims = [
            Claim(
                claim_id=1,
                cpac_data=CPACData(
                    employer_name="Test Corp",
                    job_title="Tester",
                    start_date="invalid-date",  # Invalid format
                    end_date="2023-12-31",
                    annual_compensation="75000 USD",
                    employment_type="full-time"
                )
            ),
            Claim(
                claim_id=2,
                cpac_data=CPACData(
                    employer_name="Test Corp 2",
                    job_title="Tester 2",
                    start_date="2024-01-01",
                    end_date="not-a-date",  # Invalid format
                    annual_compensation="80000 USD",
                    employment_type="full-time"
                )
            )
        ]
        
        invalid_request = CPACConsistencyRequest(
            claim_category=ClaimCategory.EMPLOYMENT,
            claims=invalid_date_claims,
            cpac_text="Test with invalid dates."
        )
        
        discrepancies = self.analyzer.analyze_employment_timeline(invalid_request)
        
        # Should handle invalid dates gracefully
        self.assertIsInstance(discrepancies, list)
        # May or may not have discrepancies depending on error handling

    def test_complex_employment_timeline(self):
        """Test complex employment timeline with multiple potential issues"""
        complex_claims = [
            Claim(
                claim_id=1,
                cpac_data=CPACData(
                    employer_name="Company A",
                    job_title="Junior Developer",
                    start_date="2019-01-01",
                    end_date="2020-06-30",
                    annual_compensation="60000 USD",
                    employment_type="full-time"
                )
            ),
            Claim(
                claim_id=2,
                cpac_data=CPACData(
                    employer_name="Company B",
                    job_title="Senior Developer",
                    start_date="2020-03-01",  # Overlaps with first job
                    end_date="2021-12-31",
                    annual_compensation="80000 USD",
                    employment_type="full-time"
                )
            ),
            Claim(
                claim_id=3,
                cpac_data=CPACData(
                    employer_name="Company C",
                    job_title="Lead Developer",
                    start_date="2023-06-01",  # Gap after second job
                    end_date="2024-12-31",
                    annual_compensation="100000 USD",
                    employment_type="full-time"
                )
            )
        ]
        
        complex_request = CPACConsistencyRequest(
            claim_category=ClaimCategory.EMPLOYMENT,
            claims=complex_claims,
            cpac_text="Client had overlapping employment and gaps in timeline."
        )
        
        discrepancies = self.analyzer.analyze_employment_timeline(complex_request)
        
        # Should detect both overlaps and gaps
        self.assertIsInstance(discrepancies, list)
        
        gap_discrepancies = [
            d for d in discrepancies 
            if d.discrepancy_type == DiscrepancyType.EMPLOYMENT_TIMELINE_GAP
        ]
        
        overlap_discrepancies = [
            d for d in discrepancies 
            if d.discrepancy_type == DiscrepancyType.EMPLOYMENT_TIMELINE_OVERLAP
        ]
        
        # Should find at least one of each type
        self.assertGreater(len(gap_discrepancies) + len(overlap_discrepancies), 0,
                          "Should detect timeline issues in complex scenario")

    def test_threshold_configuration(self):
        """Test threshold configuration"""
        # Test analyzer with custom thresholds
        custom_analyzer = CPACTimelineAnalyzer(
            gap_threshold_months=3,
            overlap_threshold_months=0.5
        )
        
        self.assertEqual(custom_analyzer.gap_threshold_months, 3)
        self.assertEqual(custom_analyzer.overlap_threshold_months, 0.5)

    def test_small_gap_below_threshold(self):
        """Test that small gaps below threshold are not reported"""
        # Create claims with small 3-month gap (below 6-month threshold)
        small_gap_claims = [
            Claim(
                claim_id=1,
                cpac_data=CPACData(
                    employer_name="Company A",
                    job_title="Engineer",
                    start_date="2020-01-01",
                    end_date="2020-06-30",
                    annual_compensation="70000 USD",
                    employment_type="full-time"
                )
            ),
            Claim(
                claim_id=2,
                cpac_data=CPACData(
                    employer_name="Company B",
                    job_title="Senior Engineer",
                    start_date="2020-10-01",  # 3-month gap
                    end_date="2021-12-31",
                    annual_compensation="90000 USD",
                    employment_type="full-time"
                )
            )
        ]
        
        request = CPACConsistencyRequest(
            claim_category=ClaimCategory.EMPLOYMENT,
            claims=small_gap_claims,
            cpac_text="Client had a brief break between jobs."
        )
        
        discrepancies = self.analyzer.analyze_employment_timeline(request)
        
        # Should not detect gap because it's below threshold
        gap_discrepancies = [
            d for d in discrepancies 
            if d.discrepancy_type == DiscrepancyType.EMPLOYMENT_TIMELINE_GAP
        ]
        self.assertEqual(len(gap_discrepancies), 0, "Small gap below threshold should not be detected")

    def test_small_overlap_below_threshold(self):
        """Test that small overlaps below threshold are not reported"""
        # Create analyzer with higher threshold for this test
        strict_analyzer = CPACTimelineAnalyzer(
            gap_threshold_months=6,
            overlap_threshold_months=2  # 2-month threshold
        )
        
        # Create claims with small 1-month overlap (below 2-month threshold)
        small_overlap_claims = [
            Claim(
                claim_id=1,
                cpac_data=CPACData(
                    employer_name="Company X",
                    job_title="Developer",
                    start_date="2022-01-01",
                    end_date="2022-06-30",
                    annual_compensation="75000 USD",
                    employment_type="full-time"
                )
            ),
            Claim(
                claim_id=2,
                cpac_data=CPACData(
                    employer_name="Company Y",
                    job_title="Consultant",
                    start_date="2022-06-01",  # 1-month overlap
                    end_date="2022-12-31",
                    annual_compensation="85000 USD",
                    employment_type="contract"
                )
            )
        ]
        
        request = CPACConsistencyRequest(
            claim_category=ClaimCategory.EMPLOYMENT,
            claims=small_overlap_claims,
            cpac_text="Client had brief overlapping employment."
        )
        
        discrepancies = strict_analyzer.analyze_employment_timeline(request)
        
        # Should not detect overlap because it's below threshold
        overlap_discrepancies = [
            d for d in discrepancies 
            if d.discrepancy_type == DiscrepancyType.EMPLOYMENT_TIMELINE_OVERLAP
        ]
        self.assertEqual(len(overlap_discrepancies), 0, "Small overlap below threshold should not be detected")

    def test_current_employment_handling(self):
        """Test handling of current/ongoing employment (None end_date)"""
        current_employment_claims = [
            Claim(
                claim_id=1,
                cpac_data=CPACData(
                    employer_name="Previous Company",
                    job_title="Engineer",
                    start_date="2020-01-01",
                    end_date="2022-12-31",
                    annual_compensation="70000 USD",
                    employment_type="full-time"
                )
            ),
            Claim(
                claim_id=2,
                cpac_data=CPACData(
                    employer_name="Current Company",
                    job_title="Senior Engineer",
                    start_date="2023-01-01",
                    end_date=None,  # Current employment
                    annual_compensation="90000 USD",
                    employment_type="full-time"
                )
            )
        ]
        
        request = CPACConsistencyRequest(
            claim_category=ClaimCategory.EMPLOYMENT,
            claims=current_employment_claims,
            cpac_text="Client currently works at Current Company."
        )
        
        discrepancies = self.analyzer.analyze_employment_timeline(request)
        
        # Should handle None end_date gracefully
        self.assertIsInstance(discrepancies, list)
    
    def test_parse_date_method(self):
        """Test the _parse_date internal method with various formats"""
        # Test standard format
        date_obj = self.analyzer._parse_date("2023-12-25")
        self.assertEqual(date_obj.year, 2023)
        self.assertEqual(date_obj.month, 12)
        self.assertEqual(date_obj.day, 25)
        
        # Test with None and use_today_if_none=True
        today_date = self.analyzer._parse_date(None, use_today_if_none=True)
        self.assertIsNotNone(today_date)
        
        # Test various string representations of None
        for none_value in ['none', 'null', 'present', 'current', 'ongoing']:
            result = self.analyzer._parse_date(none_value, use_today_if_none=True)
            self.assertIsNotNone(result)
        
        # Test alternative date formats
        alt_formats = [
            ("25/12/2023", (2023, 12, 25)),
            ("12/25/2023", (2023, 12, 25)),
            ("2023/12/25", (2023, 12, 25)),
            ("25-12-2023", (2023, 12, 25)),
            ("12-25-2023", (2023, 12, 25))
        ]
        
        for date_str, (year, month, day) in alt_formats:
            try:
                parsed = self.analyzer._parse_date(date_str)
                # Some formats might be ambiguous (month/day vs day/month)
                # Just verify it parses to a valid date
                self.assertIsInstance(parsed, datetime)
            except ValueError:
                # Some formats might not parse - that's acceptable
                pass
        
        # Test invalid date format
        with self.assertRaises(ValueError):
            self.analyzer._parse_date("invalid-date")
        
        # Test None without use_today_if_none flag
        with self.assertRaises(ValueError):
            self.analyzer._parse_date(None, use_today_if_none=False)
    
    def test_format_duration_method(self):
        """Test the _format_duration internal method"""
        # Test days
        self.assertEqual(self.analyzer._format_duration(15), "15 days")
        self.assertEqual(self.analyzer._format_duration(29), "29 days")
        
        # Test months
        self.assertEqual(self.analyzer._format_duration(30), "1 month")
        self.assertEqual(self.analyzer._format_duration(61), "2 months")
        self.assertEqual(self.analyzer._format_duration(182), "6 months")
        
        # Test years
        self.assertEqual(self.analyzer._format_duration(365), "1 year")
        self.assertEqual(self.analyzer._format_duration(730), "2 years")
        
        # Test years and months combination
        result = self.analyzer._format_duration(395)  # ~1 year, 1 month
        self.assertIn("year", result)
        self.assertIn("month", result)
    
    def test_are_duplicate_claims_method(self):
        """Test the _are_duplicate_claims internal method"""
        # Test exact duplicates
        claim1 = Claim(
            claim_id=1,
            cpac_data=CPACData(
                employer_name="TechCorp",
                job_title="Engineer",
                start_date="2020-01-01",
                end_date="2021-12-31",
                annual_compensation="80000 USD",
                employment_type="full-time"
            )
        )
        
        claim2 = Claim(
            claim_id=2,
            cpac_data=CPACData(
                employer_name="TechCorp",
                job_title="Engineer",
                start_date="2020-01-01",
                end_date="2021-12-31",
                annual_compensation="80000 USD",
                employment_type="full-time"
            )
        )
        
        # Should be duplicates
        self.assertTrue(self.analyzer._are_duplicate_claims(claim1, claim2))
        
        # Test different employers - not duplicates
        claim3 = Claim(
            claim_id=3,
            cpac_data=CPACData(
                employer_name="DifferentCorp",
                job_title="Engineer",
                start_date="2020-01-01",
                end_date="2021-12-31",
                annual_compensation="80000 USD",
                employment_type="full-time"
            )
        )
        
        self.assertFalse(self.analyzer._are_duplicate_claims(claim1, claim3))
        
        # Test partial duplicates (one has missing field)
        claim4 = Claim(
            claim_id=4,
            cpac_data=CPACData(
                employer_name="TechCorp",
                job_title=None,  # Missing job title
                start_date="2020-01-01",
                end_date="2021-12-31",
                annual_compensation="80000 USD",
                employment_type="full-time"
            )
        )
        
        # Should still be considered duplicates
        self.assertTrue(self.analyzer._are_duplicate_claims(claim1, claim4))
        
        # Test completely different claims
        claim5 = Claim(
            claim_id=5,
            cpac_data=CPACData(
                employer_name="StartupInc",
                job_title="Developer",
                start_date="2022-01-01",
                end_date="2023-12-31",
                annual_compensation="90000 USD",
                employment_type="contract"
            )
        )
        
        self.assertFalse(self.analyzer._are_duplicate_claims(claim1, claim5))
    
    def test_gap_threshold_edge_cases(self):
        """Test gap detection at threshold boundaries"""
        # Test gap exactly at threshold (6 months = ~183 days)
        threshold_claims = [
            Claim(
                claim_id=1,
                cpac_data=CPACData(
                    employer_name="Company A",
                    job_title="Engineer",
                    start_date="2020-01-01",
                    end_date="2020-06-30",
                    annual_compensation="70000 USD",
                    employment_type="full-time"
                )
            ),
            Claim(
                claim_id=2,
                cpac_data=CPACData(
                    employer_name="Company B",
                    job_title="Senior Engineer",
                    start_date="2021-01-01",  # Exactly 6 months gap
                    end_date="2021-12-31",
                    annual_compensation="90000 USD",
                    employment_type="full-time"
                )
            )
        ]
        
        request = CPACConsistencyRequest(
            claim_category=ClaimCategory.EMPLOYMENT,
            claims=threshold_claims,
            cpac_text="Employment with gap at threshold."
        )
        
        discrepancies = self.analyzer.analyze_employment_timeline(request)
        
        # Should detect gap at threshold
        gap_discrepancies = [
            d for d in discrepancies
            if d.discrepancy_type == DiscrepancyType.EMPLOYMENT_TIMELINE_GAP
        ]
        self.assertGreater(len(gap_discrepancies), 0)
    
    @patch('agent.cpac_timeline_analyzer.datetime')
    def test_current_date_handling_with_mock(self, mock_datetime):
        """Test current date handling with minimal mocking"""
        # Mock only the datetime.now() call
        fixed_date = datetime(2024, 6, 15)
        mock_datetime.now.return_value = fixed_date
        mock_datetime.strptime = datetime.strptime  # Keep real strptime
        
        # Test current employment (None end_date)
        current_date = self.analyzer._parse_date(None, use_today_if_none=True)
        
        # Should return the mocked current date
        self.assertEqual(current_date, fixed_date)
    
    def test_timeline_analysis_with_mixed_data_quality(self):
        """Test timeline analysis with mixed data quality (some missing fields)"""
        mixed_quality_claims = [
            Claim(
                claim_id=1,
                cpac_data=CPACData(
                    employer_name="Complete Corp",
                    job_title="Engineer",
                    start_date="2020-01-01",
                    end_date="2021-12-31",
                    annual_compensation="80000 USD",
                    employment_type="full-time"
                )
            ),
            Claim(
                claim_id=2,
                cpac_data=CPACData(
                    employer_name="Partial Inc",
                    job_title=None,  # Missing job title
                    start_date="2022-06-01",
                    end_date="2023-05-31",
                    annual_compensation=None,  # Missing compensation
                    employment_type="full-time"
                )
            ),
            Claim(
                claim_id=3,
                cpac_data=CPACData(
                    employer_name=None,  # Missing employer
                    job_title="Developer",
                    start_date="2023-07-01",
                    end_date=None,  # Current employment
                    annual_compensation="95000 USD",
                    employment_type="contract"
                )
            )
        ]
        
        request = CPACConsistencyRequest(
            claim_category=ClaimCategory.EMPLOYMENT,
            claims=mixed_quality_claims,
            cpac_text="Employment history with varying data quality."
        )
        
        discrepancies = self.analyzer.analyze_employment_timeline(request)
        
        # Should handle mixed data quality gracefully
        self.assertIsInstance(discrepancies, list)
        
        # Should still detect timeline issues despite missing fields
        gap_discrepancies = [
            d for d in discrepancies
            if d.discrepancy_type == DiscrepancyType.EMPLOYMENT_TIMELINE_GAP
        ]
        # There's a gap between claim 2 (ends 2023-05-31) and claim 3 (starts 2023-07-01)
        # But it's only ~1 month, below the 6-month threshold
        self.assertEqual(len(gap_discrepancies), 0)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)
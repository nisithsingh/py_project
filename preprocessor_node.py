"""
Preprocessing Agent for CPAC Consistency Reviewer
Handles data cleaning and formatting before analysis:
- Converts various date formats to YYYY-MM-DD
- Sorts claims chronologically
"""
import os
import sys
import logging
import yaml
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
import re

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_model.schemas import (
    CPACConsistencyRequest, Claim, CPACData, ClaimCategory
)
from utils.llm_handler import LLMHandler
from jinja2 import Environment, FileSystemLoader


class PreprocessorNode:
    """
    Preprocessing Node for cleaning and standardizing input data
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the Preprocessing Node
        
        Args:
            config_path: Path to configuration file
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        config_full_path = Path(__file__).parent.parent / config_path
        with open(config_full_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Get preprocessing config
        self.preprocessing_config = self.config.get('preprocessing', {})
        prompt_path = self.preprocessing_config.get('prompt_path', 'prompts/preprocessing_prompt_v1.j2')
        
        # Get full prompt path
        prompt_full_path = Path(__file__).parent.parent / prompt_path
        
        # Initialize LLM handler with prompt path and config path
        self.llm_handler = LLMHandler(
            prompt_path=str(prompt_full_path),
            config_path=str(config_full_path),
            logger=self.logger
        )
        
        # Setup Jinja2 template for manual rendering if needed
        template_dir = Path(__file__).parent.parent / "prompts"
        self.env = Environment(loader=FileSystemLoader(template_dir))
        # Extract just the filename for Jinja2
        self.prompt_filename = Path(prompt_path).name
        
        # Debug settings
        self.print_prompts = self.config.get('debug', {}).get('print_prompts', True)
        self.print_results = self.config.get('debug', {}).get('print_results', True)
    
    async def preprocess_claims(self, request: CPACConsistencyRequest) -> CPACConsistencyRequest:
        """
        Preprocess claims data to standardize format and sort chronologically
        
        Args:
            request: Original CPAC consistency request
            
        Returns:
            Preprocessed request with cleaned and sorted data
        """
        # Step 1: Use LLM to standardize date formats
        cleaned_claims = await self._standardize_claims_with_llm(request)
        
        # Step 2: Sort claims chronologically using Python
        sorted_claims = self._sort_claims_chronologically(cleaned_claims)
        
        # Create new request with preprocessed data
        preprocessed_request = CPACConsistencyRequest(
            claim_category=request.claim_category,
            claims=sorted_claims,
            cpac_text=request.cpac_text
        )
        
        return preprocessed_request
    
    def _try_parse_date_deterministic(self, date_str: str, detected_format: Optional[str] = None) -> Optional[str]:
        """
        Try to parse date using deterministic rules
        
        Args:
            date_str: Date string in various formats
            detected_format: Optional format detected from dataset analysis
            
        Returns:
            Date in YYYY-MM-DD format if successful, None otherwise
        """
        if not date_str or date_str.lower() in ['none', 'null', 'present', 'current', 'ongoing']:
            self.logger.debug(f"Date '{date_str}' is a special value, keeping as-is")
            return date_str  # Return as-is for special values
            
        # Already in correct format?
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            self.logger.debug(f"Date '{date_str}' is already in YYYY-MM-DD format")
            return date_str
            
        # Try common unambiguous formats first
        formats_to_try = [
            # Already correct formats
            ('%Y-%m-%d', 'YYYY-MM-DD'),
            ('%Y/%m/%d', 'YYYY/MM/DD'),
            
            # Unambiguous with month names
            ('%d-%b-%Y', 'DD-Mon-YYYY'),      # e.g., 31-Dec-2011
            ('%d/%b/%Y', 'DD/Mon/YYYY'),      # e.g., 31/Dec/2011
            ('%d.%b.%Y', 'DD.Mon.YYYY'),      # e.g., 31.Dec.2011
            ('%d %b %Y', 'DD Mon YYYY'),      # e.g., 31 Dec 2011
            ('%d-%B-%Y', 'DD-Month-YYYY'),    # e.g., 31-December-2011
            ('%d/%B/%Y', 'DD/Month/YYYY'),    # e.g., 31/December/2011
            ('%d %B %Y', 'DD Month YYYY'),    # e.g., 31 December 2011
            ('%b %d, %Y', 'Mon DD, YYYY'),    # e.g., Dec 31, 2011
            ('%B %d, %Y', 'Month DD, YYYY'),  # e.g., December 31, 2011
            ('%b. %d, %Y', 'Mon. DD, YYYY'), # e.g., Dec. 31, 2011
            ('%d %b, %Y', 'DD Mon, YYYY'),    # e.g., 31 Dec, 2011
            ('%d %B, %Y', 'DD Month, YYYY'),  # e.g., 31 December, 2011
            
            # ISO-like formats
            ('%Y%m%d', 'YYYYMMDD'),           # e.g., 20111231
            ('%Y.%m.%d', 'YYYY.MM.DD'),       # e.g., 2011.12.31
            
            # Common European formats (try these after checking for day > 12)
            ('%d/%m/%Y', 'DD/MM/YYYY'),       # e.g., 31/12/2011
            ('%d-%m-%Y', 'DD-MM-YYYY'),       # e.g., 31-12-2011
            ('%d.%m.%Y', 'DD.MM.YYYY'),       # e.g., 31.12.2011
            ('%d %m %Y', 'DD MM YYYY'),       # e.g., 31 12 2011
            
            # Common US formats (try these after checking for month > 12)
            ('%m/%d/%Y', 'MM/DD/YYYY'),       # e.g., 12/31/2011
            ('%m-%d-%Y', 'MM-DD-YYYY'),       # e.g., 12-31-2011
            ('%m.%d.%Y', 'MM.DD.YYYY'),       # e.g., 12.31.2011
            
            # Two-digit year formats (assume 20xx for years 00-30, 19xx for 31-99)
            ('%d/%m/%y', 'DD/MM/YY'),         # e.g., 31/12/11
            ('%m/%d/%y', 'MM/DD/YY'),         # e.g., 12/31/11
            ('%d-%m-%y', 'DD-MM-YY'),         # e.g., 31-12-11
            ('%m-%d-%y', 'MM-DD-YY'),         # e.g., 12-31-11
            ('%y-%m-%d', 'YY-MM-DD'),         # e.g., 11-12-31
            ('%y/%m/%d', 'YY/MM/DD'),         # e.g., 11/12/31
            
            # Other variations
            ('%d of %B, %Y', 'DD of Month, YYYY'),  # e.g., 31st of December, 2011
            ('%dth %B %Y', 'DDth Month YYYY'),      # e.g., 31st December 2011
            ('%dst %B %Y', 'DDst Month YYYY'),      # e.g., 1st December 2011
            ('%dnd %B %Y', 'DDnd Month YYYY'),      # e.g., 2nd December 2011
            ('%drd %B %Y', 'DDrd Month YYYY'),      # e.g., 3rd December 2011
        ]
        
        # Filter out None values
        formats_to_try = [(fmt, desc) for fmt, desc in formats_to_try if fmt is not None]
        
        for fmt, desc in formats_to_try:
            try:
                parsed = datetime.strptime(date_str, fmt)
                result = parsed.strftime('%Y-%m-%d')
                self.logger.debug(f"Successfully parsed '{date_str}' as {desc} format -> '{result}'")
                return result
            except ValueError:
                continue
                
        # For ambiguous formats, check if we can determine based on values
        # If day > 12, we know it must be DD not MM
        if '/' in date_str or '-' in date_str:
            parts = re.split(r'[/-]', date_str)
            if len(parts) == 3:
                # Try to determine format based on values
                if all(p.isdigit() for p in parts):
                    p0, p1, p2 = int(parts[0]), int(parts[1]), int(parts[2])
                    
                    # Year is 4 digits
                    if len(parts[0]) == 4:  # YYYY-MM-DD or YYYY-DD-MM
                        if p1 <= 12 and p2 <= 31:
                            result = f"{parts[0]}-{parts[1]:02d}-{parts[2]:02d}"
                            self.logger.debug(f"Parsed '{date_str}' as YYYY-MM-DD based on numeric constraints -> '{result}'")
                            return result
                    elif len(parts[2]) == 4:  # DD-MM-YYYY or MM-DD-YYYY
                        if p0 > 12 and p1 <= 12:  # Must be DD-MM-YYYY
                            result = f"{parts[2]}-{p1:02d}-{p0:02d}"
                            self.logger.debug(f"Parsed '{date_str}' as DD-MM-YYYY (day > 12) -> '{result}'")
                            return result
                        elif p1 > 12 and p0 <= 12:  # Must be MM-DD-YYYY
                            result = f"{parts[2]}-{p0:02d}-{p1:02d}"
                            self.logger.debug(f"Parsed '{date_str}' as MM-DD-YYYY (day > 12) -> '{result}'")
                            return result
                        elif p0 <= 12 and p1 <= 12:
                            # Both could be month - check if they give same result
                            dd_mm_result = f"{parts[2]}-{p1:02d}-{p0:02d}"
                            mm_dd_result = f"{parts[2]}-{p0:02d}-{p1:02d}"
                            
                            if dd_mm_result == mm_dd_result:
                                # Same result either way (e.g., 1/1/2005)
                                self.logger.debug(f"Date '{date_str}' gives same result for both formats -> '{dd_mm_result}'")
                                return dd_mm_result
                            elif detected_format == 'MIXED':
                                # Mixed formats in dataset - can't reliably parse
                                self.logger.debug(f"Date '{date_str}' is ambiguous in mixed-format dataset (DD/MM: {dd_mm_result}, MM/DD: {mm_dd_result})")
                                return None  # Send to LLM
                            elif detected_format:
                                # Use detected format
                                if detected_format == 'DD/MM/YYYY':
                                    self.logger.debug(f"Using detected DD/MM format for ambiguous date '{date_str}' -> '{dd_mm_result}'")
                                    return dd_mm_result
                                else:
                                    self.logger.debug(f"Using detected MM/DD format for ambiguous date '{date_str}' -> '{mm_dd_result}'")
                                    return mm_dd_result
                            else:
                                # Truly ambiguous - different results and no format detected
                                self.logger.debug(f"Date '{date_str}' is truly ambiguous (DD/MM: {dd_mm_result}, MM/DD: {mm_dd_result})")
                                return None
        
        self.logger.debug(f"Could not parse '{date_str}' deterministically")
        return None  # Couldn't parse deterministically
    
    def _detect_date_format_from_dataset(self, claims: List[Claim]) -> Optional[str]:
        """
        Detect the predominant date format by analyzing unambiguous dates in the dataset
        
        Returns:
            'DD/MM/YYYY', 'MM/DD/YYYY', or None if no format can be determined
        """
        dd_mm_evidence = 0
        mm_dd_evidence = 0
        
        # Check all date fields in all claims
        for claim in claims:
            date_fields = [
                ('start_date', claim.cpac_data.start_date),
                ('end_date', claim.cpac_data.end_date),
                ('inheritance_date', getattr(claim.cpac_data, 'inheritance_date', None))
            ]
            
            for field_name, date_str in date_fields:
                if not date_str or date_str.lower() in ['none', 'null', 'present', 'current', 'ongoing']:
                    continue
                    
                # Skip if already in YYYY-MM-DD format
                if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                    continue
                
                # Check numeric date formats with / or -
                if '/' in date_str or '-' in date_str:
                    parts = re.split(r'[/-]', date_str)
                    if len(parts) == 3 and all(p.isdigit() for p in parts):
                        # Determine position of year (assume 4-digit year)
                        if len(parts[0]) == 4:  # YYYY/?/?
                            continue  # Can't determine DD/MM vs MM/DD from this
                        elif len(parts[2]) == 4:  # ?/?/YYYY
                            p0, p1 = int(parts[0]), int(parts[1])
                            
                            # Check for unambiguous indicators
                            if p0 > 12 and p1 <= 12:  # Must be DD/MM/YYYY
                                dd_mm_evidence += 1
                                self.logger.debug(f"Found DD/MM evidence: '{date_str}' (day {p0} > 12)")
                            elif p1 > 12 and p0 <= 12:  # Must be MM/DD/YYYY
                                mm_dd_evidence += 1
                                self.logger.debug(f"Found MM/DD evidence: '{date_str}' (day {p1} > 12)")
                            elif p0 > 31:  # Invalid day, must be year in wrong position
                                continue
                            # If both <= 12, it's ambiguous - don't count as evidence
        
        # Determine format based on evidence
        self.logger.debug(f"Date format detection: DD/MM evidence={dd_mm_evidence}, MM/DD evidence={mm_dd_evidence}")
        
        if dd_mm_evidence > 0 and mm_dd_evidence == 0:
            self.logger.debug("Detected consistent DD/MM/YYYY format in dataset")
            return 'DD/MM/YYYY'
        elif mm_dd_evidence > 0 and dd_mm_evidence == 0:
            self.logger.debug("Detected consistent MM/DD/YYYY format in dataset")
            return 'MM/DD/YYYY'
        elif dd_mm_evidence > 0 and mm_dd_evidence > 0:
            self.logger.warning(f"Found MIXED date formats in dataset (DD/MM: {dd_mm_evidence}, MM/DD: {mm_dd_evidence})")
            self.logger.warning("This suggests inconsistent data entry - will need LLM for all ambiguous dates")
            return 'MIXED'  # Special indicator for mixed formats
        else:
            self.logger.debug("No unambiguous date format indicators found")
            return None
    
    def _parse_date_with_detected_format(self, date_str: str, detected_format: str) -> Optional[str]:
        """
        Parse date using the detected format preference
        
        Args:
            date_str: Date string to parse
            detected_format: 'DD/MM/YYYY' or 'MM/DD/YYYY'
            
        Returns:
            Parsed date in YYYY-MM-DD format or None
        """
        if detected_format == 'DD/MM/YYYY':
            formats_to_try = [
                '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y', '%d %m %Y',
                '%d/%m/%y', '%d-%m-%y', '%d.%m.%y'
            ]
        else:  # MM/DD/YYYY
            formats_to_try = [
                '%m/%d/%Y', '%m-%d-%Y', '%m.%d.%Y', '%m %d %Y',
                '%m/%d/%y', '%m-%d-%y', '%m.%d.%y'
            ]
            
        for fmt in formats_to_try:
            try:
                parsed = datetime.strptime(date_str, fmt)
                return parsed.strftime('%Y-%m-%d')
            except ValueError:
                continue
                
        return None

    async def _standardize_claims_with_llm(self, request: CPACConsistencyRequest) -> List[Claim]:
        """
        Standardize date formats using deterministic parsing first, then LLM for ambiguous cases
        
        Args:
            request: Original request with claims
            
        Returns:
            List of claims with standardized date formats
        """
        try:
            # Step 1: Detect date format from unambiguous dates in dataset
            detected_format = self._detect_date_format_from_dataset(request.claims)
            
            # If mixed formats detected, we should use LLM for all dates to ensure consistency
            if detected_format == 'MIXED':
                self.logger.warning("Mixed date formats detected - using LLM for all date standardization")
                # Skip deterministic parsing and go straight to LLM
                needs_llm = True
                processed_claims = []
                for claim in request.claims:
                    processed_claims.append({
                        "claim_id": claim.claim_id,
                        "cpac_data": claim.cpac_data.model_dump()
                    })
            else:
                # Step 2: Try deterministic parsing with detected format
                needs_llm = False
                processed_claims = []
                ambiguous_dates = []
                
                for claim in request.claims:
                    claim_dict = {
                        "claim_id": claim.claim_id,
                        "cpac_data": claim.cpac_data.model_dump()
                    }
                    
                    # Try to parse dates deterministically
                    cpac_data = claim_dict["cpac_data"]
                    date_fields = ['start_date', 'end_date', 'inheritance_date']
                    
                    for field in date_fields:
                        if field in cpac_data and cpac_data[field]:
                            original_date = cpac_data[field]
                            parsed = self._try_parse_date_deterministic(original_date, detected_format)
                            if parsed:
                                cpac_data[field] = parsed
                                if parsed != original_date:
                                    self.logger.debug(f"Claim {claim.claim_id} - {field}: '{original_date}' -> '{parsed}'")
                            else:
                                needs_llm = True  # Found unparseable date
                                ambiguous_dates.append(f"Claim {claim.claim_id} - {field}: '{original_date}'")
                                self.logger.debug(f"Claim {claim.claim_id} - {field}: '{original_date}' needs LLM parsing")
                                
                    processed_claims.append(claim_dict)
            
            # If all dates were parsed successfully, skip LLM
            if not needs_llm:
                self.logger.debug("=" * 60)
                self.logger.debug("DATE PARSING SUMMARY: All dates parsed deterministically")
                self.logger.debug("No LLM needed - using deterministic results")
                self.logger.debug("=" * 60)
                cleaned_claims = []
                for claim_data in processed_claims:
                    cpac_data = CPACData(**claim_data["cpac_data"])
                    claim = Claim(
                        claim_id=int(claim_data["claim_id"]),
                        cpac_data=cpac_data
                    )
                    cleaned_claims.append(claim)
                return cleaned_claims
                
            # Otherwise, use LLM for ambiguous cases
            self.logger.debug("=" * 60)
            self.logger.debug("DATE PARSING SUMMARY: Found unparseable dates")
            self.logger.debug(f"Detected format in dataset: {detected_format or 'None'}")
            self.logger.debug(f"Unparseable dates requiring LLM: {len(ambiguous_dates)}")
            for date_info in ambiguous_dates[:5]:  # Show first 5
                self.logger.debug(f"  - {date_info}")
            if len(ambiguous_dates) > 5:
                self.logger.debug(f"  ... and {len(ambiguous_dates) - 5} more")
            self.logger.debug("Using LLM for date standardization")
            self.logger.debug("=" * 60)
            # Prepare input data for LLM handler
            claims_data = []
            for claim in request.claims:
                # Use model_dump to get all fields including None values
                claim_dict = {
                    "claim_id": claim.claim_id,
                    "cpac_data": claim.cpac_data.model_dump()
                }
                claims_data.append(claim_dict)
            
            # Prepare input data for query
            input_data = {
                'claim_category': request.claim_category,
                'claims': json.dumps(claims_data, indent=2)
            }
            
            # Use LLM handler's query_generic method with custom system message
            system_message = "You are a data preprocessing specialist. Convert dates to YYYY-MM-DD format and preserve all other data exactly as provided."
            
            result = self.llm_handler.query_generic(input_data, system_message=system_message)
            
            # Extract response
            if result.get('response_data'):
                response_data = result['response_data']
            else:
                self.logger.error("Failed to get response from LLM")
                return request.claims
            
            # Log response if enabled
            if self.print_results:
                self.logger.debug("=" * 80)
                self.logger.debug("LLM RESPONSE:")
                self.logger.debug("=" * 80)
                self.logger.debug(json.dumps(response_data, indent=2))
                self.logger.debug("=" * 80)
            
            # Convert back to Claim objects using Pydantic models
            cleaned_claims = []
            self.logger.debug("=" * 60)
            self.logger.debug("LLM DATE CONVERSION RESULTS:")
            for claim_data in response_data.get("claims", []):
                try:
                    # Ensure claim_id is integer
                    claim_id = int(claim_data["claim_id"])
                    
                    # Log date conversions from LLM
                    cpac_data_dict = claim_data["cpac_data"]
                    for field in ['start_date', 'end_date', 'inheritance_date']:
                        if field in cpac_data_dict and cpac_data_dict[field]:
                            # Find original value
                            original_claim = next(
                                (c for c in request.claims if c.claim_id == claim_id),
                                None
                            )
                            if original_claim:
                                original_value = getattr(original_claim.cpac_data, field, None)
                                new_value = cpac_data_dict[field]
                                if original_value and original_value != new_value:
                                    self.logger.debug(f"Claim {claim_id} - {field}: '{original_value}' -> '{new_value}' (LLM)")
                    
                    # Create CPACData object using Pydantic model
                    # This will validate all fields according to the schema
                    cpac_data = CPACData(**cpac_data_dict)
                    
                    # Create Claim object
                    claim = Claim(
                        claim_id=claim_id,
                        cpac_data=cpac_data
                    )
                    cleaned_claims.append(claim)
                    
                except Exception as e:
                    self.logger.error(f"Error parsing claim {claim_data.get('claim_id', 'unknown')}: {str(e)}")
                    # If parsing fails for a claim, use the original
                    # Find the original claim with matching ID
                    original_claim = next(
                        (c for c in request.claims if c.claim_id == claim_data.get("claim_id", -1)),
                        None
                    )
                    if original_claim:
                        cleaned_claims.append(original_claim)
            
            self.logger.debug("=" * 60)
            return cleaned_claims
            
        except Exception as e:
            self.logger.error(f"Error in LLM standardization: {str(e)}")
            # Return original claims if LLM processing fails
            return request.claims
    
    def _sort_claims_chronologically(self, claims: List[Claim]) -> List[Claim]:
        """
        Sort claims chronologically by start_date (employment) or other date fields
        Only sorts employment claims; other categories maintain original order
        
        Args:
            claims: List of claims to sort
            
        Returns:
            Sorted list of claims (employment) or original order (other categories)
        """
        try:
            # Check if any claim has employment-related fields
            if not claims:
                return claims
                
            # Check if any claims have start_date field (could be employment, business_profit, etc.)
            has_timeline = any(hasattr(claim.cpac_data, 'start_date') and claim.cpac_data.start_date for claim in claims)
            
            if not has_timeline:
                self.logger.debug("No timeline data found in claims, skipping chronological sorting")
                return claims
                
            self.logger.debug("Timeline data detected, performing chronological sorting")
            # Define a function to parse date for sorting
            def get_start_date(claim):
                start_date_str = claim.cpac_data.start_date
                if not start_date_str:
                    return datetime.min
                
                try:
                    # Try parsing YYYY-MM-DD format
                    return datetime.strptime(start_date_str, "%Y-%m-%d")
                except:
                    # Return min date if parsing fails
                    return datetime.min
            
            # Define a function to get entity name for secondary sorting
            def get_entity_name(claim):
                # Get employer name for employment, company name for business, or empty string
                if hasattr(claim.cpac_data, 'employer_name') and claim.cpac_data.employer_name:
                    return claim.cpac_data.employer_name.lower()
                elif hasattr(claim.cpac_data, 'company_name') and claim.cpac_data.company_name:
                    return claim.cpac_data.company_name.lower()
                else:
                    return ''
            
            # Sort claims first by start date, then by entity name
            # This groups same-employer/company claims together when they have the same start date
            sorted_claims = sorted(claims, key=lambda c: (get_start_date(c), get_entity_name(c)))
            
            # Log sorting results
            self.logger.debug("=" * 60)
            self.logger.debug("CHRONOLOGICAL SORTING RESULTS:")
            for i, claim in enumerate(sorted_claims, 1):
                date_str = getattr(claim.cpac_data, 'start_date', 'No start date') or "No start date"
                # Get appropriate entity name
                if hasattr(claim.cpac_data, 'employer_name') and claim.cpac_data.employer_name:
                    entity = f"Employer: {claim.cpac_data.employer_name}"
                elif hasattr(claim.cpac_data, 'company_name') and claim.cpac_data.company_name:
                    entity = f"Company: {claim.cpac_data.company_name}"
                else:
                    entity = "Unknown entity"
                self.logger.debug(f"Position {i}: Claim {claim.claim_id} -> {date_str} | {entity}")
            
            # Reassign claim IDs to maintain sequential order
            # for i, claim in enumerate(sorted_claims, 1):
            #     if claim.claim_id != i:
            #         self.logger.debug(f"Reassigning claim ID: {claim.claim_id} -> {i}")
            #     claim.claim_id = i
            # self.logger.debug("=" * 60)
            
            return sorted_claims
            
        except Exception as e:
            self.logger.error(f"Error in chronological sorting: {str(e)}")
            # Return original order if sorting fails
            return claims
    
    def preprocess_test_data(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess test data format (for compatibility with test runners)
        
        Args:
            test_data: Raw test data dictionary
            
        Returns:
            Preprocessed test data
        """
        try:
            # Create request from test data
            claims = []
            for claim_data in test_data.get("claims", []):
                cpac_data_dict = claim_data.get("cpac_data", {})
                
                cpac_data = CPACData(**cpac_data_dict)
                claim = Claim(
                    claim_id=int(claim_data.get("claim_id", 1)),
                    cpac_data=cpac_data
                )
                claims.append(claim)
            
            request = CPACConsistencyRequest(
                claim_category=ClaimCategory(test_data.get("claim_category", "employment")),
                claims=claims,
                cpac_text=test_data.get("cpac_text", "")
            )
            
            # Run preprocessing synchronously for test data
            import asyncio
            preprocessed_request = asyncio.run(self.preprocess_claims(request))
            
            # Convert back to test data format
            preprocessed_data = test_data.copy()
            preprocessed_data["claims"] = []
            
            for claim in preprocessed_request.claims:
                # Keep claim_id as string if original was string, otherwise as int
                original_claim_data = next(
                    (c for c in test_data.get("claims", []) 
                     if str(c.get("claim_id", "")) == str(claim.claim_id)),
                    None
                )
                
                if original_claim_data:
                    # Preserve original claim_id type (string or int)
                    claim_id = original_claim_data.get("claim_id", claim.claim_id)
                else:
                    claim_id = claim.claim_id
                
                claim_dict = {
                    "claim_id": claim_id,
                    "cpac_data": claim.cpac_data.model_dump()  # Include all fields, even None
                }
                
                # Add back any additional fields from original that aren't in our schema
                if original_claim_data:
                    for key in ["claim_sow_category", "claim_type"]:
                        if key in original_claim_data:
                            claim_dict[key] = original_claim_data[key]
                
                preprocessed_data["claims"].append(claim_dict)
            
            return preprocessed_data
            
        except Exception as e:
            self.logger.error(f"Error in test data preprocessing: {str(e)}")
            return test_data


# Test function for standalone execution
if __name__ == "__main__":
    import asyncio
    
    async def test_preprocessing():
        # Initialize agent
        agent = PreprocessingAgent()
        
        # Load test data
        test_file = Path(__file__).parent.parent / "test_data" / "employment_testcase_2_orig.json"
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # Preprocess
        preprocessed = agent.preprocess_test_data(test_data)
        
        # Print results
        print("\n" + "="*80)
        print("PREPROCESSED DATA:")
        print("="*80)
        print(json.dumps(preprocessed, indent=2))
        print("="*80)
    
    # Run test
    asyncio.run(test_preprocessing())
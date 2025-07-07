"""
Executor Node - Pure Business Logic for CPAC Consistency Analysis

This module contains the core business logic extracted from cpac_consistency_reviewer.py
specifically for use as the executor_node in the LangGraph workflow.

The executor node analyzes CPAC claims for discrepancies using:
- LLM-based analysis for all claim categories
- Rule-based fallback for employment timeline analysis (handled separately)
"""
import os
import sys
import logging
from typing import List, Optional
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_model.schemas import (
    CPACConsistencyRequest, Discrepancy, ClaimCategory
)
from pydantic import ValidationError
from utils.llm_handler import LLMHandler


class ExecutorNode:
    """
    Pure business logic for CPAC consistency analysis
    Extracted from CPACConsistencyReviewerAgent for use in LangGraph workflow
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the executor node
        
        Args:
            config_path: Path to configuration file
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize exception tracking list
        self.exception_list = []
        
        # Load configuration
        import yaml
        config_full_path = Path(__file__).parent.parent / config_path
        with open(config_full_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize LLM handler for employment-specific analysis
        employment_prompt_path = self.config['llm']['prompt_path']
        full_prompt_path = Path(__file__).parent.parent / employment_prompt_path
        
        self.llm_handler = LLMHandler(
            prompt_path=str(full_prompt_path),
            config_path=str(config_full_path),
            logger=self.logger
        )
        
    async def analyze_claims_with_llm(self, request: CPACConsistencyRequest) -> List[Discrepancy]:
        """
        Analyze claims for discrepancies using LLM with exception tracking
        Uses category-appropriate prompts
        
        Args:
            request: CPAC consistency request
            
        Returns:
            List of discrepancies found
        """
        # Reset exception list for this analysis
        self.exception_list = []
        
        # Select appropriate prompt based on category
        if request.claim_category != ClaimCategory.EMPLOYMENT:
            # Use generic prompt for non-employment categories from config
            generic_prompt_path = self.config['llm'].get('generic_prompt_path', 'prompts/cpac_analysis_prompt_generic_v8.j2')
            full_generic_path = Path(__file__).parent.parent / generic_prompt_path
            
            if full_generic_path.exists():
                self.logger.info(f"Using generic prompt for {request.claim_category.value} category: {generic_prompt_path}")
                # Create temporary LLM handler with generic prompt
                llm_handler = LLMHandler(
                    prompt_path=str(full_generic_path),
                    config_path=str(Path(__file__).parent.parent / "config" / "config.yaml"),
                    logger=self.logger
                )
            else:
                self.logger.warning(f"Generic prompt not found at {full_generic_path}, using default employment prompt for {request.claim_category.value}")
                llm_handler = self.llm_handler
        else:
            # Use default employment prompt
            self.logger.info("Using employment-specific prompt for employment category")
            llm_handler = self.llm_handler
            
        # Get threshold values from config (in months) and convert to days for backward compatibility
        gap_threshold_months = self.config.get('analysis', {}).get('timeline_gap_threshold_months', 2)
        overlap_threshold_months = self.config.get('analysis', {}).get('timeline_overlap_threshold_months', 1)
        
        # Convert months to approximate days (30 days per month) for internal processing if needed
        gap_threshold_days = gap_threshold_months * 30
        overlap_threshold_days = overlap_threshold_months * 30
        
        # Prepare data for LLM - pass months to the prompt
        input_data = {
            'claim_category': request.claim_category.value,
            'claims': [claim.model_dump() for claim in request.claims],
            'cpac_text': request.cpac_text,
            'gap_threshold_months': gap_threshold_months,
            'overlap_threshold_months': overlap_threshold_months
        }
        
        # Use LLM to analyze claims
        # Call LLM handler (using the selected handler)
        llm_result = llm_handler.query(input_data)
        
        # Merge LLM exceptions with executor exceptions
        if 'exceptions' in llm_result and llm_result['exceptions']:
            self.exception_list.extend(llm_result['exceptions'])
            self.logger.warning(f"LLM handler reported {len(llm_result['exceptions'])} exceptions")
        
        # Extract discrepancies from LLM result
        if isinstance(llm_result, dict) and 'discrepancies' in llm_result:
            discrepancies = llm_result['discrepancies']
        else:
            error_msg = "LLM result does not contain discrepancies"
            self.logger.warning(error_msg)
            self.exception_list.append({
                "exception_type": "LLMResponseFormatError",
                "message": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            discrepancies = []
            
        # Convert to Discrepancy objects if needed
        discrepancy_objects = []
        for idx, disc in enumerate(discrepancies):
            if isinstance(disc, dict):
                discrepancy_objects.append(Discrepancy(**disc))
            else:
                discrepancy_objects.append(disc)
                
        return discrepancy_objects
    
    def get_exceptions(self) -> List[dict]:
        """
        Get list of exceptions that occurred during analysis
        
        Returns:
            List of exception dictionaries
        """
        return self.exception_list.copy()
    
    def clear_exceptions(self):
        """Clear the exception list"""
        self.exception_list = []
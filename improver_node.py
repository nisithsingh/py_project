"""
Improvement Agent - Processes reviewer feedback to improve discrepancy analysis
This agent receives feedback from the Review Agent and applies suggested improvements
"""
import os
import sys
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_model.schemas import Discrepancy, BinaryClassification, BinaryReviewOutput
from utils.llm_handler import LLMHandler


class ImproverNode:
    """
    Improvement Node that processes reviewer feedback and improves discrepancy analysis
    
    This agent:
    1. Receives feedback from the reviewer including:
       - action (what to do: remove or revise discrepancies)
       - analysis/reason (detailed explanation)
       - decision (pass/fail)
    2. Processes the original discrepancy list
    3. Applies improvements based on reviewer feedback
    4. Returns improved discrepancy list
    """
    
    def __init__(self, 
                 prompt_path: str = None,
                 config_path: str = "config/config.yaml"):
        """Initialize the improvement agent"""
        self.logger = logging.getLogger(__name__)
        
        # Set default prompt path if not provided
        if prompt_path is None:
            prompt_path = str(Path(__file__).parent.parent / "prompts" / "improvement_prompt.j2")
        
        # Initialize LLMHandler with the improvement prompt
        self.llm_handler = LLMHandler(
            prompt_path=prompt_path,
            config_path=config_path,
            logger=self.logger
        )
        
        # Load config for debug settings
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        self.print_results = config.get('debug', {}).get('print_results', True)
        
        self.logger.info("Improvement Agent initialized")
    
    async def improve_discrepancies_from_binary(
        self,
        discrepancies: List[Discrepancy],
        binary_output: BinaryReviewOutput
    ) -> List[Discrepancy]:
        """
        Improve discrepancies based on binary classification output
        
        Args:
            discrepancies: Current list of discrepancies
            binary_output: Binary classification output from reviewer
            
        Returns:
            List[Discrepancy]: Improved list of discrepancies
        """
        self.logger.info(f"Processing {len(discrepancies)} discrepancies from binary classification")
        self.logger.info(f"Binary decisions: {binary_output.total_remove} to remove, {binary_output.total_approve} to approve")
        
        # Convert binary output to standard review action format
        review_action = {
            "remove": {},
            "approve": {},
            "revise": {}
        }
        
        # Process each classification
        for disc_id, classification in binary_output.classifications.items():
            # Clean the discrepancy ID (remove brackets if present)
            clean_id = disc_id.strip('[]')
            
            if classification.decision == "REMOVE":
                review_action["remove"][clean_id] = classification.justification
            else:  # APPROVE
                review_action["approve"][clean_id] = classification.justification
        
        # Create review reasons from binary output
        review_reasons = []
        for disc_id, classification in binary_output.classifications.items():
            clean_id = disc_id.strip('[]')
            if classification.decision == "REMOVE":
                review_reasons.append(f"Discrepancy {clean_id} marked for removal")
            else:
                review_reasons.append(f"Discrepancy {clean_id} approved for further review")
        
        # Use existing improvement logic
        return await self.improve_discrepancies(
            discrepancies=discrepancies,
            review_action=review_action,
            review_analysis=None,
            review_reasons=review_reasons
        )
    
    async def improve_discrepancies(
        self,
        discrepancies: List[Discrepancy],
        review_action: Optional[Dict[str, Any]] = None,
        review_analysis: Optional[str] = None,
        review_reasons: Optional[List[str]] = None
    ) -> List[Discrepancy]:
        """
        Improve discrepancies based on reviewer feedback
        
        Args:
            discrepancies: Current list of discrepancies
            review_action: Action details from reviewer (e.g., which discrepancy to remove/revise)
            review_analysis: Detailed analysis from reviewer
            review_reasons: List of reasons from reviewer
            
        Returns:
            List[Discrepancy]: Improved list of discrepancies
        """
        self.logger.info(f"Processing {len(discrepancies)} discrepancies for improvement")
        
        # If no discrepancies or no actions to take, return original list
        if not discrepancies or not review_action:
            self.logger.info("No discrepancies or no review actions provided")
            return discrepancies
        
        # Debug logging
        self.logger.info(f"Improvement agent received review_action: {review_action}")
        if review_action:
            self.logger.info(f"  - remove keys: {list(review_action.get('remove', {}).keys())}")
            self.logger.info(f"  - keep keys: {list(review_action.get('approve', {}).keys())}")
            self.logger.info(f"  - revise keys: {list(review_action.get('revise', {}).keys())}")
        
        # Prepare review feedback structure for the prompt
        # Note: review_reasons contains the actual analysis list from reviewer
        # review_analysis parameter is deprecated and usually None/empty
        review_feedback = {
            "action": review_action,
            "analysis": review_reasons if review_reasons else []  # Use review_reasons which has the actual data
        }
        
        # Prepare input variables for the prompt
        input_variables = {
            "discrepancy_list": [d.model_dump() for d in discrepancies],
            "review_feedback": review_feedback
        }
        
        # Use LLMHandler's generic query method
        response = self.llm_handler.query_generic(input_variables)
        
        # Check if there were exceptions
        if response.get('exceptions'):
            self.logger.warning(f"LLM handler reported {len(response['exceptions'])} exceptions")
        
        # Get the response data
        response_data = response.get('response_data', {})
        
        if not response_data:
            self.logger.error("Empty response from LLM")
            return discrepancies
        
        # Extract improved discrepancies from response
        improved_discrepancy_list = response_data.get('discrepancies', [])
        
        # Convert back to Discrepancy objects
        improved_discrepancies = []
        for disc_data in improved_discrepancy_list:
            # Ensure affected_document_ids is empty (CPAC-only)
            if 'affected_document_ids' not in disc_data:
                disc_data['affected_document_ids'] = []
            
            # Create Discrepancy object
            discrepancy = Discrepancy(**disc_data)
            improved_discrepancies.append(discrepancy)
        
        # Print improvement results header if enabled
        if hasattr(self, 'print_results') and self.print_results:
            print("\n" + "="*60)
            print(f"IMPROVEMENT AGENT RESULTS - {len(discrepancies)} -> {len(improved_discrepancies)} discrepancies")
            print("="*60 + "\n")
        
        self.logger.info(f"Improvement completed: {len(discrepancies)} -> {len(improved_discrepancies)} discrepancies")
        
        # Log details of improvements
        if review_action.get('remove'):
            self.logger.info(f"Removed {len(review_action['remove'])} discrepancies")
        if review_action.get('revise'):
            self.logger.info(f"Revised {len(review_action['revise'])} discrepancies")
        if review_action.get('approve'):
            self.logger.info(f"Approved {len(review_action['approve'])} discrepancies")
        
        return improved_discrepancies
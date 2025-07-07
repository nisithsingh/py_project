import os, sys, asyncio, logging, yaml, math
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_model.schemas import (
    CPACConsistencyRequest, CPACConsistencyResult, 
    ReviewOutput, ClaimCategory, BinaryClassification, BinaryReviewOutput
)
from utils.llm_handler import LLMHandler


class ReviewerNode:
    def __init__(self, 
                 prompt_path: str,
                 config_path: str = "config/config.yaml"):
        
        self.logger = logging.getLogger(__name__)
        
        # Store paths for prompt switching
        self.default_prompt_path = prompt_path
        self.config_path = config_path
        
        # Initialize LLMHandler with the review prompt
        self.llm_handler = LLMHandler(
            prompt_path=prompt_path,
            config_path=config_path,
            logger=self.logger
        )
        
        # Load config for debug settings
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.print_prompts = self.config.get('debug', {}).get('print_prompts', True)
        self.print_results = self.config.get('debug', {}).get('print_results', True)
    
    # async def binary_classification_review(self, discrepancy_objects: CPACConsistencyResult, cpac_object: CPACConsistencyRequest) -> BinaryReviewOutput:
    #     """
    #     Process review request using binary classification prompt with logprobs.
    #     Returns BinaryReviewOutput instead of ReviewOutput.
    #     """
    #     if not discrepancy_objects.discrepancies:
    #         return BinaryReviewOutput(
    #             classifications={},
    #             review_decision=False,
    #             error="No Discrepancy Found",
    #             total_remove=0,
    #             total_approve=0
    #         )
            
    #     try:
    #         # Get binary prompt path from config
    #         binary_prompt_filename = self.config.get('reviewer', {}).get('binary_prompt_path', 'prompts/reviewer_prompt_binary_v15.j2')
    #         binary_prompt_path = Path(__file__).parent.parent / binary_prompt_filename
            
    #         if not binary_prompt_path.exists():
    #             self.logger.error(f"Binary classification prompt not found at {binary_prompt_path}")
    #             raise FileNotFoundError(f"Binary classification prompt not found: {binary_prompt_filename}")
                
    #         # Create LLM handler with binary prompt
    #         binary_llm_handler = LLMHandler(
    #             prompt_path=str(binary_prompt_path),
    #             config_path=self.config_path,
    #             logger=self.logger
    #         )
            
    #         # Prepare input variables
    #         input_variables = { 
    #             'claim_category': cpac_object.claim_category.value,
    #             "discrepancy_list" : [d.model_dump() for d in discrepancy_objects.discrepancies],
    #             "claims_list" : [c.model_dump() for c in cpac_object.claims],
    #             "cpac_text" : cpac_object.cpac_text
    #         }
            
    #         # Call LLM with logprobs enabled
    #         response = binary_llm_handler.query_generic_with_logprobs(input_variables)
            
    #         # Extract classification data
    #         classification_data = response.get('response_data', {})
    #         logprobs_data = response.get('logprobs', {})
            
    #         # Parse classifications
    #         classifications = {}
    #         total_remove = 0
    #         total_approve = 0
            
    #         raw_classifications = classification_data.get('classification', {})
    #         for disc_id, class_data in raw_classifications.items():
    #             # Create BinaryClassification object
    #             classification = BinaryClassification(
    #                 decision=class_data.get('decision', 'APPROVE'),
    #                 justification=class_data.get('justification', '')
    #             )
    #             classifications[disc_id] = classification
                
    #             if classification.decision == 'REMOVE':
    #                 total_remove += 1
    #             else:
    #                 total_approve += 1
            
    #         # Display results if enabled
    #         if self.print_results and classifications:
    #             print("\n" + "="*60)
    #             print("BINARY CLASSIFICATION RESULTS")
    #             print("="*60)
    #             print(f"\nTotal Discrepancies: {len(classifications)}")
    #             print(f"Approved: {total_approve}")
    #             print(f"Removed: {total_remove}")
                
    #             print("\nClassifications:")
    #             for disc_id, classification in classifications.items():
    #                 print(f"\n  Discrepancy {disc_id}:")
    #                 print(f"    Decision: {classification.decision}")
    #                 print(f"    Justification: {classification.justification}")
                
    #             # Analyze token probabilities if available
    #             # if logprobs_data:
    #             #     print("\n" + "-"*60)
    #             #     print("LOGPROBS ANALYSIS")
    #             #     print("-"*60)
    #             #     self._analyze_classification_confidence(logprobs_data)
                
    #             # print("="*60 + "\n")
            
    #         return BinaryReviewOutput(
    #             classifications=classifications,
    #             review_decision=False,  # Always False to trigger improvement
    #             error="",
    #             total_remove=total_remove,
    #             total_approve=total_approve
    #         )
            
    #     except Exception as e:
    #         self.logger.error(f"Error in binary classification review: {e}")
    #         return BinaryReviewOutput(
    #             classifications={},
    #             review_decision=False,
    #             error=f"Binary Classification Failed: {str(e)}",
    #             total_remove=0,
    #             total_approve=0
    #         )

    async def process_request_with_binary_classification(self, discrepancy_objects: CPACConsistencyResult, cpac_object: CPACConsistencyRequest):
        """
        Process review request using binary classification prompt with logprobs for confidence assessment.
        This method is designed for use with OpenAI's logprobs feature.
        """
        if not discrepancy_objects.discrepancies:
            return ReviewOutput(
                review_decision=True, 
                error="No Discrepancy Found", 
                analysis=[],
                action=None
            )
            
        # Get binary prompt path from config
        binary_prompt_filename = self.config.get('reviewer', {}).get('binary_prompt_path', 'prompts/reviewer_prompt_binary_v13.j2')
        binary_prompt_path = Path(__file__).parent.parent / binary_prompt_filename
        
        if not binary_prompt_path.exists():
            self.logger.error(f"Binary classification prompt not found at {binary_prompt_path}")
            raise FileNotFoundError(f"Binary classification prompt not found: {binary_prompt_filename}")
            
        # Create LLM handler with binary prompt
        binary_llm_handler = LLMHandler(
            prompt_path=str(binary_prompt_path),
            config_path=self.config_path,
            logger=self.logger
        )
        
        # Prepare input variables
        input_variables = { 
            'claim_category': cpac_object.claim_category.value,
            "discrepancy_list" : [d.model_dump() for d in discrepancy_objects.discrepancies],
            "claims_list" : [c.model_dump() for c in cpac_object.claims],
            "cpac_text" : cpac_object.cpac_text
        }
        
        # Call LLM with logprobs enabled
        response = binary_llm_handler.query_generic_with_logprobs(input_variables)
        
        # Extract classification data
        classification_data = response.get('response_data', {})
        logprobs_data = response.get('logprobs', {})
        
        if self.print_results and classification_data:
            print("\n" + "="*60)
            print("BINARY CLASSIFICATION RESULTS")
            print("="*60)
            
            classifications = classification_data.get('classification', {})
            
            # Calculate counts from classifications
            approved_count = sum(1 for c in classifications.values() if c.get('decision') == 'APPROVE')
            removed_count = sum(1 for c in classifications.values() if c.get('decision') == 'REMOVE')
            total_count = len(classifications)
            
            print(f"\nTotal Discrepancies: {total_count}")
            print(f"Kept: {approved_count}")
            print(f"Removed: {removed_count}")
            
            print("\nClassifications:")
            for disc_id, classification in classifications.items():
                decision = classification.get('decision', 'UNKNOWN')
                if decision == 'APPROVED':
                    decision = 'KEPT' # using a synonym for clarity
                justification = classification.get('justification', '')
                print(f"\n  Discrepancy {disc_id}:")
                print(f"    Decision: {decision}")
                print(f"    Justification: {justification}")
            
            # # Analyze token probabilities if available
            # if logprobs_data:
            #     print("\n" + "-"*60)
            #     print("LOGPROBS ANALYSIS")
            #     print("-"*60)
            #     # This is a simplified analysis - you can expand based on needs
            #     self._analyze_classification_confidence(logprobs_data)
            
            print("="*60 + "\n")
        
        # Convert classification to standard ReviewOutput format
        result = self._convert_classification_to_review_output(classification_data, discrepancy_objects)
        
        # Debug logging
        self.logger.info(f"Binary classification ReviewOutput created:")
        self.logger.info(f"  - review_decision: {result.review_decision}")
        self.logger.info(f"  - action remove keys: {list(result.action.get('remove', {}).keys())}")
        self.logger.info(f"  - action keep keys: {list(result.action.get('approve', {}).keys())}")
        
        return result
    
    def _analyze_classification_confidence(self, logprobs_data: Dict[str, Any]):
        """Analyze logprobs to assess classification confidence"""
        if not logprobs_data or 'content' not in logprobs_data:
            print("No logprobs data available")
            return
            
        # Look for decision tokens (APPROVE/REMOVE) and their probabilities
        content_tokens = logprobs_data.get('content', [])
        
        decision_tokens = []
        
        # Track if we've seen "decision" recently
        decision_window = 10  # Look for decision tokens within 10 tokens of "decision"
        
        for i, token_data in enumerate(content_tokens):
            token = token_data.get('token', '')
            logprob = token_data.get('logprob', 0)
            prob = math.exp(logprob)  # Convert log probability to probability (e^logprob)
            
            # Check if this is a decision token or part of one
            # APPROVE can be split as: APP+ROVE, APPRO+VE, etc.
            # REMOVE can be split as: REM+OVE, etc.
            decision_related = token.upper() in ['APPROVE', 'REMOVE', 'APP', 'REM', 'APPRO', 'VE', 'ROVE', 'OVE']
            
            if decision_related:
                # Look back to see if "decision" appears nearby
                found_decision = False
                for j in range(max(0, i-decision_window), i):
                    if 'decision' in content_tokens[j].get('token', '').lower():
                        found_decision = True
                        break
                
                # Only include if we found "decision" nearby
                if found_decision:
                    # Get top alternatives
                    alternatives = []
                    for alt in token_data.get('top_logprobs', []):
                        alt_token = alt.get('token', '')
                        alt_logprob = alt.get('logprob', 0)
                        alt_prob = math.exp(alt_logprob)
                        alternatives.append(f"{alt_token} ({alt_prob:.2%})")
                    
                    # Determine the full decision word
                    full_decision = "UNKNOWN"
                    if token.upper() in ['APPROVE', 'REMOVE']:
                        full_decision = token.upper()
                    elif token.upper() in ['APP', 'APPRO']:
                        full_decision = "APPROVE"
                    elif token.upper() in ['REM']:
                        full_decision = "REMOVE"
                    elif token.upper() in ['VE', 'ROVE'] and i > 0:
                        # Check if previous token was APP/APPRO
                        prev_token = content_tokens[i-1].get('token', '').upper()
                        if prev_token in ['APP', 'APPRO']:
                            full_decision = "APPROVE"
                    elif token.upper() in ['OVE'] and i > 0:
                        # Check if previous token was REM
                        prev_token = content_tokens[i-1].get('token', '').upper()
                        if prev_token == 'REM':
                            full_decision = "REMOVE"
                    
                    decision_tokens.append({
                        'token': token,
                        'full_decision': full_decision,
                        'probability': prob,
                        'alternatives': alternatives,
                        'position': i
                    })
            
         
        # Display analysis
        if decision_tokens:
            print("\nDecision Token Analysis:")
            
            # Group by decision for better readability
            # Use smaller grouping to separate different discrepancies
            grouped_decisions = {}
            decision_counter = 0
            last_position = -50  # Initialize to ensure first token creates new group
            
            for dt in decision_tokens:
                # Create new group if position jumped significantly (more than 30 tokens)
                if dt['position'] - last_position > 30:
                    decision_counter += 1
                
                key = f"Discrepancy {decision_counter}"
                if key not in grouped_decisions:
                    grouped_decisions[key] = []
                grouped_decisions[key].append(dt)
                last_position = dt['position']
            
            for group_key, tokens in grouped_decisions.items():
                # Determine the full decision from the tokens
                full_decision = "UNKNOWN"
                for token in tokens:
                    if token['full_decision'] != "UNKNOWN":
                        full_decision = token['full_decision']
                        break
                
                print(f"\n  {group_key}: {full_decision}")
                for dt in tokens:
                    print(f"    Token '{dt['token']}' - Probability: {dt['probability']:.2%}")
                    if dt['alternatives']:
                        print(f"      Top alternatives: {', '.join(dt['alternatives'][:3])}")
        
        if not decision_tokens:
            print("No explicit APPROVE/REMOVE tokens found in response")
        
        # Calculate average confidence for decision tokens
        if decision_tokens:
            avg_confidence = sum(dt['probability'] for dt in decision_tokens) / len(decision_tokens)
            print(f"\nAverage decision token confidence: {avg_confidence:.2%}")
    
    def _convert_classification_to_review_output(self, classification_data: Dict[str, Any], 
                                                discrepancy_objects: CPACConsistencyResult) -> ReviewOutput:
        """Convert binary classification format to standard ReviewOutput format"""
        classifications = classification_data.get('classification', {})
        
        # Debug logging
        self.logger.info(f"Converting binary classification to ReviewOutput")
        self.logger.info(f"Classifications received: {classifications}")
        
        # Build action dictionary
        action = {
            "remove": {},
            "approve": {}
        }
        
        analysis = []
        
        for disc in discrepancy_objects.discrepancies:
            disc_id = str(disc.discrepancy_id)
            
            # Try to find classification with different key formats
            classification = None
            for key in [disc_id, f"[{disc_id}]", f"{{{disc_id}}}", f'"{disc_id}"']:
                if key in classifications:
                    classification = classifications[key]
                    self.logger.debug(f"Found classification for discrepancy {disc_id} with key '{key}'")
                    break
            
            if not classification:
                classification = {}
                self.logger.warning(f"No classification found for discrepancy {disc_id}")
            
            decision = classification.get('decision', 'APPROVE')  # Default to APPROVE
            justification = classification.get('justification', 'No justification provided')
            
            self.logger.debug(f"Discrepancy {disc_id}: decision={decision}, justification={justification[:50]}...")
            
            if decision == 'REMOVE':
                action['remove'][disc_id] = justification
                analysis.append(f"Discrepancy {disc_id} marked for removal")
            else:  # APPROVE
                action['approve'][disc_id] = justification
                analysis.append(f"Discrepancy {disc_id} approved for further review")
        
        # Log final action dictionary
        self.logger.info(f"Final action dictionary: remove={list(action['remove'].keys())}, approve={list(action['approve'].keys())}")
        
        # Always set review_decision to False for binary classification
        # This ensures the improvement node is triggered in the workflow
        review_decision = False
        
        return ReviewOutput(
            review_decision=review_decision,
            error="",
            analysis=analysis,
            action=action
        )

    async def process_request(self, discrepancy_objects: CPACConsistencyResult, cpac_object: CPACConsistencyRequest):
        review_reasons_list = []        
        if len(discrepancy_objects.discrepancies):
            # Select appropriate prompt based on category
            if cpac_object.claim_category != ClaimCategory.EMPLOYMENT:
                # Use generic prompt for non-employment categories
                generic_prompt_path = Path(__file__).parent.parent / "prompts" / "reviewer_prompt_generic_v12.j2"
                
                if generic_prompt_path.exists():
                    self.logger.info(f"Using generic reviewer prompt v12 for {cpac_object.claim_category.value} category")
                    # Create temporary LLM handler with generic prompt
                    llm_handler = LLMHandler(
                        prompt_path=str(generic_prompt_path),
                        config_path=self.config_path,
                        logger=self.logger
                    )
                else:
                    self.logger.warning(f"Generic reviewer prompt v12 not found, using default prompt")
                    llm_handler = self.llm_handler
            else:
                # Use default employment prompt
                self.logger.info("Using employment-specific reviewer prompt v11")
                llm_handler = self.llm_handler
            
            # Prepare input variables
            input_variables = { 
                'claim_category': cpac_object.claim_category.value,  # Get from request object
                "discrepancy_list" : [d.model_dump() for d in discrepancy_objects.discrepancies],
                "claims_list" : [c.model_dump() for c in cpac_object.claims],
                "cpac_text" : cpac_object.cpac_text
            }
            
            # Use LLMHandler's generic query method with selected handler
            # System message is already included in the prompt template
            response = llm_handler.query_generic(input_variables)
            
            # Check if there were exceptions
            if response.get('exceptions'):
                self.logger.warning(f"LLM handler reported {len(response['exceptions'])} exceptions")
            
            # Get the response data
            response_data = response.get('response_data', {})
            raw_response = response.get('raw_response', '')
            
            # Format and display reviewer response if enabled
            if self.print_results and response_data:
                print("\n" + "="*60)
                print("REVIEWER AGENT RESPONSE")
                print("="*60)
                
                # Display Reviewer General Decision (renamed from review_decision)
                decision = response_data.get("review_decision", False)
                print(f"\nReviewer General Decision: {'PASS' if decision else 'FAIL'}")
                
                # Display Analysis as bullet points (renamed from reason)
                analysis_items = response_data.get("analysis", [])
                if analysis_items:
                    print("\nAnalysis:")
                    for item in analysis_items:
                        print(f"  • {item}")
                
                # Display Action section nicely
                action = response_data.get("action", None)
                if action:
                    print("\nAction Required:")
                    if isinstance(action, dict):
                        # Handle remove section
                        if 'remove' in action and action['remove']:
                            print("  Remove:")
                            if isinstance(action['remove'], dict):
                                for disc_id, reason in action['remove'].items():
                                    print(f"    • Discrepancy {disc_id}: {reason}")
                            else:
                                print(f"    • {action['remove']}")
                        
                        # Handle revise section
                        if 'revise' in action and action['revise']:
                            print("  Revise:")
                            if isinstance(action['revise'], dict):
                                for disc_id, revision in action['revise'].items():
                                    if isinstance(revision, dict):
                                        suggestion = revision.get('suggestion', 'No suggestion provided')
                                        print(f"    • Discrepancy {disc_id}: {suggestion}")
                                    else:
                                        print(f"    • Discrepancy {disc_id}: {revision}")
                            else:
                                print(f"    • {action['revise']}")
                        
                        # Handle approve section
                        if 'approve' in action and action['approve']:
                            print("  Approve:")
                            if isinstance(action['approve'], dict):
                                for disc_id, reason in action['approve'].items():
                                    print(f"    • Discrepancy {disc_id}: {reason}")
                            else:
                                print(f"    • {action['approve']}")
                        
                        # Handle any other action items
                        for key, value in action.items():
                            if key not in ['remove', 'revise', 'approve']:
                                print(f"  {key}: {value}")
                    else:
                        print(f"  {action}")
                
                # Note: Hiding "error" field and "analysis" section as requested
                
                print("="*60 + "\n")
            
            # Check for empty response
            if not response_data:
                self.logger.error("Empty or invalid response from LLM")
                return ReviewOutput(
                    review_decision=True, 
                    error="Failed to parse LLM response", 
                    analysis=review_reasons_list,
                    action=None
                )
            
            return ReviewOutput(
                review_decision=response_data.get("review_decision", False),
                error=response_data.get("error", ""),
                analysis=response_data.get("analysis", []),
                action=response_data.get("action", None)
            )
                
        else:
            return ReviewOutput(
                review_decision=True, 
                error="No Discrepancy Found", 
                analysis=review_reasons_list,
                action=None
            )
    
        

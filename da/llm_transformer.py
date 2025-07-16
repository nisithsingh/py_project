"""
LLM-based request transformer for utility agents
Transforms employment agent requests into utility agent-specific formats using LLM
"""
import logging
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import asyncio

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class TransformationError(Exception):
    """Custom exception for transformation failures"""
    pass


class LLMTransformer:
    """
    LLM-based transformer that converts employment requests to utility agent formats
    """
    
    def __init__(self, agent_cards_dir: Optional[Path] = None):
        self.agent_cards_dir = agent_cards_dir or Path(__file__).parent.parent / "agent-cards"
        self.agent_cards = {}
        self.llm_client = None
        self._load_agent_cards()
        
    def _load_agent_cards(self):
        """Load all agent cards from the agent-cards directory"""
        if not self.agent_cards_dir.exists():
            logger.warning(f"Agent cards directory not found: {self.agent_cards_dir}")
            return
            
        for card_file in self.agent_cards_dir.glob("*.json"):
            try:
                with open(card_file, 'r') as f:
                    card_data = json.load(f)
                
                agent_name = card_data.get("name")
                if agent_name:
                    self.agent_cards[agent_name] = card_data
                    logger.info(f"Loaded agent card for: {agent_name}")
                    
            except Exception as e:
                logger.error(f"Failed to load agent card {card_file}: {e}")
                
        logger.info(f"Loaded {len(self.agent_cards)} agent cards")
    
    def _get_llm_client(self):
        """Get or create LLM client"""
        if self.llm_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise TransformationError("OPENAI_API_KEY environment variable not set")
                
            model = os.getenv("OPENAI_MODEL", "gpt-4o")
            self.llm_client = ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=0.1  # _create_transformation_promptLow temperature for consistent transformations
            )
            
        return self.llm_client
    
    def _create_transformation_prompt(self, employment_request: Dict, target_agent_card: Dict) -> str:
        """Create detailed prompt for LLM transformation"""
        
        agent_name = target_agent_card["name"]
        target_schema = target_agent_card["requestSchema"]
        agent_description = target_agent_card["description"]
        agent_responsibilities = target_agent_card["scope"]["responsibilities"]
        
        prompt = f"""You are a request transformer that converts employment agent requests into specific utility agent requests.

        TARGET AGENT: {agent_name}
        AGENT DESCRIPTION: {agent_description}
        AGENT RESPONSIBILITIES: {" ".join(agent_responsibilities)}

        TARGET REQUEST SCHEMA:
        {json.dumps(target_schema, indent=2)}

        ORIGINAL EMPLOYMENT REQUEST:
        {json.dumps(employment_request, indent=2)}

        TRANSFORMATION TASK:
        1. Analyze the original employment request structure and data
        2. Transform it to match the target agent's exact schema requirements
        3. Preserve all relevant data that the target agent needs
        4. Add any required fields based on the target schema
        5. Remove or modify fields that don't match the target schema



        OUTPUT FORMAT:
        Return ONLY a valid JSON object that matches the target schema exactly.
        Do not include any explanations or additional text.
        Ensure the JSON is properly formatted and valid.

        TRANSFORMED REQUEST:"""
        
        return prompt
    
    def _get_transformation_examples(self, agent_name: str) -> str:
        """Get agent-specific transformation examples"""
        examples = {
            "cpac_consistency_agent": """
            Example: Remove supporting_documents array, keep cpac_data and claims structure
            Original: {{"claims": [{{"supporting_documents": [...]}}]}}
            Transformed: {{"claims": [{{"claim_id": 1, "cpac_data": {{...}}}}]}}
                        """,
                        
                        "document_consistency_agent": """
            Example: Ensure client_id is present, keep supporting_documents
            Original: {{"name": "John Doe", "claims": [...]}}
            Transformed: {{"client_id": "uuid", "name": "John Doe", "claims": [...]}}
            """
        }
        return examples.get(agent_name, "")
    
    def _parse_llm_response(self, llm_response: str, target_agent_card: Dict) -> Dict:
        """Parse LLM response and validate against target schema"""
        
        try:
            # Extract JSON from LLM response
            response_text = llm_response.strip()
            
            # Try to find JSON content in the response
            if response_text.startswith("```json"):
                # Remove markdown code blocks
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                # Remove generic code blocks
                response_text = response_text.replace("```", "").strip()
                
            transformed_request = json.loads(response_text)
            
            # Basic validation - check if required fields are present
            target_schema = target_agent_card["requestSchema"]
            required_fields = target_schema.get("required", [])
            
            for field in required_fields:
                if field not in transformed_request:
                    raise TransformationError(f"Missing required field: {field}")
            
            logger.info(f"Successfully transformed request for {target_agent_card['name']}")
            return transformed_request
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from LLM: {e}")
            logger.error(f"LLM Response: {llm_response}")
            raise TransformationError(f"Invalid JSON from LLM: {e}")
        except Exception as e:
            logger.error(f"Transformation validation failed: {e}")
            raise TransformationError(f"Schema validation failed: {e}")
    
    async def transform_request_for_agent(self, employment_request: Dict, agent_name: str) -> Dict:
        """
        Transform employment request for a specific agent using LLM
        
        Args:
            employment_request: Original request from employment agent
            agent_name: Name of the target utility agent
            
        Returns:
            Transformed request matching the target agent's schema
            
        Raises:
            TransformationError: If transformation fails
        """
        logger.info(f"Transforming request for agent: {agent_name}")
        
        # Get agent card
        if agent_name not in self.agent_cards:
            raise TransformationError(f"No agent card found for: {agent_name}")
            
        target_agent_card = self.agent_cards[agent_name]
        
        # Get LLM client
        llm_client = self._get_llm_client()
        
        #max_retries = 3
        
        #for attempt in range(max_retries):
        try:
            # Create transformation prompt
            prompt = self._create_transformation_prompt(employment_request, target_agent_card)
            
            #logger.debug(f"Sending transformation prompt to LLM (attempt {attempt + 1})")
            logger.info(f"Sending transformation prompt to LLM for "+agent_name)
            print("prompt:")
            print(prompt)

            # Call LLM
            message = HumanMessage(content=prompt)
            response = await llm_client.ainvoke([message])
            
            # Parse and validate response
            transformed = self._parse_llm_response(response.content, target_agent_card)
            print("LLM response:")
            print(transformed)
            logger.info(f"Successfully transformed request for {agent_name}")
            return transformed
            
        except Exception as e:
            #logger.warning(f"Transformation attempt {attempt + 1} failed for {agent_name}: {e}")
            
            #if attempt == max_retries - 1:
            logger.error(f"All transformation attempts failed for {agent_name}")
            raise TransformationError(f"Failed to transform request for {agent_name}: {e}")
            
            # Wait before retry
            await asyncio.sleep(1)
        
        #raise TransformationError(f"Exhausted all retry attempts for {agent_name}")
    
    def create_fallback_request(self, employment_request: Dict, agent_name: str) -> Dict:
        """
        Create a fallback request using rule-based transformation
        Used when LLM transformation fails
        """
        logger.warning(f"Creating fallback request for {agent_name}")
        
        # Basic fallback - copy common fields
        fallback_request = {
            "name": employment_request.get("name"),
            "claim_category": employment_request.get("claim_category"),
            "cpac_text": employment_request.get("cpac_text"),
            "claims": employment_request.get("claims", [])
        }
        
        # Agent-specific fallback rules
        if agent_name == "cpac_consistency_agent":
            # Remove supporting_documents from claims
            claims = fallback_request.get("claims", [])
            for claim in claims:
                if "supporting_documents" in claim:
                    del claim["supporting_documents"]
                    
        elif agent_name == "document_consistency_agent":
            # Ensure client_id is present
            if "client_id" not in fallback_request:
                fallback_request["client_id"] = employment_request.get("client_id", "unknown")
        
        logger.info(f"Created fallback request for {agent_name}")
        return fallback_request


# Global transformer instance
_transformer_instance = None


def get_transformer() -> LLMTransformer:
    """Get global transformer instance"""
    global _transformer_instance
    if _transformer_instance is None:
        _transformer_instance = LLMTransformer()
    return _transformer_instance


async def transform_request_for_agent(employment_request: Dict, agent_name: str, use_fallback: bool = True) -> Dict:
    """
    Convenience function to transform request for an agent
    
    Args:
        employment_request: Original employment request
        agent_name: Target agent name
        use_fallback: Whether to use fallback on LLM failure
        
    Returns:
        Transformed request
    """
    transformer = get_transformer()
    
    try:
        return await transformer.transform_request_for_agent(employment_request, agent_name)
    except TransformationError as e:
        if use_fallback:
            logger.warning(f"LLM transformation failed, using fallback for {agent_name}: {e}")
            return transformer.create_fallback_request(employment_request, agent_name)
        else:
            raise e
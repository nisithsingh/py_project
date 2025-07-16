"""
Preprocessing node for Discrepancy Analyzer Agent
Uses LLM to transform employment requests into utility agent-specific formats
"""
import logging
from typing import Dict, List
from datetime import datetime, timezone
import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.output_handler import get_output_handler
from utils.print_handler import get_print_handler
from utils.llm_transformer import transform_request_for_agent, TransformationError

logger = logging.getLogger(__name__)
output_handler = get_output_handler()
print_handler = get_print_handler()


async def preprocess_node(state: Dict) -> Dict:
    """
    Preprocessing node - uses LLM to transform requests for utility agents
    
    Args:
        state: Current graph state with request data
        
    Returns:
        Updated state with transformed utility agent requests
    """
    logger.info("--- Entering LLM-based preprocess_node ---")
    print("\nPREPROCESSOR NODE STARTED (LLM-based transformation)", flush=True)
    
    client_id = state.get("client_id")
    client_name = state.get("client_name")
    request_data = state.get("request_data", {})
    
    logger.info(f"Preprocessing request for client {client_id} ({client_name})")
    
    if state.get("debug", {}).get("print_results"):
        logger.info(f"Original request data: {request_data}")
    
    # Prepare the base employment request for transformation
    employment_request = {
        "client_id": client_id,
        "name": client_name,
        "claim_category": request_data.get("claim_category"),
        "cpac_text": request_data.get("cpac_text"),
        "claims": request_data.get("claims", [])
    }
    
    # Get active agents from config
    active_agents = state.get("active_agents", [])
    
    # Print dispatching status
    print_handler.print_processing_status("preprocessing_with_llm", {
        "agents": active_agents,
        "transformation_mode": "LLM-based"
    })
    
    # Transform requests for each utility agent using LLM
    utility_agent_requests = {}
    transformation_logs = {}
    
    for agent_name in active_agents:
        try:
            logger.info(f"Transforming request for agent: {agent_name}")
            print(f"Transforming request for {agent_name} using LLM...", flush=True)
            
            # Use LLM to transform the request
            transformed_request = await transform_request_for_agent(
                employment_request, 
                agent_name,
                use_fallback=True  # Enable fallback on LLM failure
            )
            
            utility_agent_requests[agent_name] = transformed_request
            transformation_logs[agent_name] = "SUCCESS"
            
            logger.info(f"Successfully transformed request for {agent_name}")
            
            if state.get("debug", {}).get("print_results"):
                logger.info(f"Transformed request for {agent_name}: {transformed_request}")
                
        except TransformationError as e:
            logger.error(f"Failed to transform request for {agent_name}: {e}")
            transformation_logs[agent_name] = f"ERROR: {str(e)}"
            
            # Create a basic fallback request
            fallback_request = {
                "client_id": client_id,
                "name": client_name,
                "claim_category": request_data.get("claim_category"),
                "cpac_text": request_data.get("cpac_text"),
                "claims": request_data.get("claims", [])
            }
            
            utility_agent_requests[agent_name] = fallback_request
            logger.warning(f"Using basic fallback request for {agent_name}")
            
        except Exception as e:
            logger.error(f"Unexpected error transforming request for {agent_name}: {e}")
            transformation_logs[agent_name] = f"UNEXPECTED_ERROR: {str(e)}"
            
            # Create a basic fallback request
            fallback_request = {
                "client_id": client_id,
                "name": client_name,
                "claim_category": request_data.get("claim_category"),
                "cpac_text": request_data.get("cpac_text"),
                "claims": request_data.get("claims", [])
            }
            
            utility_agent_requests[agent_name] = fallback_request
            logger.warning(f"Using basic fallback request for {agent_name} due to unexpected error")
    
    # Save preprocessing output with transformation logs
    thread_id = state.get("thread_id")
    output_handler.save_node_output(
        "preprocessor",
        {
            "client_id": client_id,
            "client_name": client_name,
            "active_agents": active_agents,
            "utility_agent_requests": utility_agent_requests,
            "transformation_logs": transformation_logs,
            "transformation_mode": "LLM-based"
        },
        thread_id
    )
    
    # Print transformation summary
    successful_transforms = sum(1 for log in transformation_logs.values() if log == "SUCCESS")
    print(f"Transformation complete: {successful_transforms}/{len(active_agents)} successful", flush=True)
    
    # Update state
    logger.info(f"LLM preprocessing complete. Transformed {len(utility_agent_requests)} utility agent requests")
    logger.info(f"Transformation success rate: {successful_transforms}/{len(active_agents)}")
    
    result = {
        "utility_agent_requests": utility_agent_requests,
        "transformation_logs": transformation_logs,
        "preprocessing_complete": True,
        "preprocessing_timestamp": datetime.now(timezone.utc).isoformat(),
        "transformation_mode": "LLM-based"
    }
    
    logger.info("--- Exiting LLM-based preprocess_node ---")
    return result
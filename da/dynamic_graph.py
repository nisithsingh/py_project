"""
Dynamic LangGraph implementation for Discrepancy Analyzer Agent
Creates individual nodes for each utility agent dynamically
Supports both in-memory and persistent SQLite checkpointing
"""
import logging
from typing import Dict, Optional, Callable, List
from pathlib import Path
import yaml
import os
import json
import shutil
from datetime import datetime, timezone, timedelta
import asyncio

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.runnables import RunnableConfig

# Add parent directories to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from state.graph_state import DiscrepancyAnalyzerState
from nodes.process_utility_responses_node import process_utility_responses_node
from nodes.send_final_result_node import send_final_result_node
from nodes.preprocessor import preprocess_node

logger = logging.getLogger(__name__)

# Node names as constants
PREPROCESSOR = "preprocessor"
PROCESS_UTILITY_RESPONSES = "process_utility_responses_node"
SEND_FINAL_RESULT = "send_final_result_node"


def create_utility_agent_node(agent_name: str, agent_config: Dict) -> Callable:
    """
    Factory function to create a utility agent node dynamically
    Each node sends a request to its specific utility agent
    """
    async def utility_agent_node(state: DiscrepancyAnalyzerState, config: RunnableConfig) -> Dict:
        """
        Node for sending request to a specific utility agent
        """
        logger.info(f"--- Node: {agent_name}_node ---")
        
        try:
            client_id = state["client_id"]
            thread_id = state.get("thread_id")
            
            # Get the original request data directly
            request_data = state.get("request_data", {})
            
            # Each utility agent gets the full request data
            agent_request = {
                "client_id": client_id,
                "name": state.get("client_name"),
                "claim_category": request_data.get("claim_category"),
                "cpac_text": request_data.get("cpac_text"),
                "claims": request_data.get("claims", [])
            }
            # Import here to avoid circular imports
            logger.info(f"Utility agent node {agent_name} - importing luma_helper...")
            from utils.luma_helper import get_luma_protocol, create_utility_agent_message
            logger.info(f"Utility agent node {agent_name} - imports successful")
            
            # Get LUMA protocol
            luma_protocol = await get_luma_protocol()
            
            # Ensure client_id is in request
            agent_request["client_id"] = client_id
            
            # Create LUMA message for this specific agent
            luma_message = create_utility_agent_message(
                target_agent=agent_name,
                request_data=agent_request,
                thread_id=state["thread_id"],
                client_id=client_id,
                job_id=state.get("job_id", f"da_job_{client_id}"),
                conversation_id=state.get("conversation_id", f"da_conv_{client_id}")
            )
            
            # Save the request being sent (for debugging/testing)
            from utils.output_handler import get_output_handler
            output_handler = get_output_handler()
            output_handler.save_utility_request(agent_name, agent_request, thread_id)
            
            # Check environment to determine how to send
            # Get environment from state (passed from config)
            environment = state.get("environment", "local")
            
            # Print request being sent
            from utils.print_handler import get_print_handler
            print_handler = get_print_handler()
            print_handler.print_utility_request(agent_name, agent_request)
            
            # For shared_local and azure environments, use standard LUMA protocol
            # No need for cross-namespace sender anymore
            await luma_protocol.send_request(luma_message)
            logger.info(f"Successfully sent request to {agent_name} via LUMA protocol")
            
            # Update state for this specific agent
            # Each agent updates its own field to avoid concurrent conflicts
            result = {
                f"{agent_name}_dispatched": True,
                f"{agent_name}_message_id": luma_message.payload.task_id
            }
            logger.info(f"Utility agent node {agent_name} returning: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to send to {agent_name}: {e}", exc_info=True)
            error_result = {
                f"{agent_name}_dispatched": False,
                f"{agent_name}_error": str(e)
            }
            logger.error(f"Utility agent node {agent_name} returning error: {error_result}")
            return error_result
    
    # Set function name for debugging
    utility_agent_node.__name__ = f"{agent_name}_node"
    return utility_agent_node


# Removed collect_dispatch_results_node - it was redundant


class GraphCheckpointerContext:
    """Async context manager for graph checkpointer with per-client isolation"""
    
    def __init__(self, config: Dict, client_id: str):
        self.config = config
        self.client_id = client_id
        self.checkpoint_config = config.get("checkpoint", {})
        self.use_persistent = self.checkpoint_config.get("use_persistent", False)
        
        # Per-client checkpoint path
        base_path = self.checkpoint_config.get("base_path", "checkpoints")
        self.client_dir = os.path.join(base_path, f"client_{client_id}")
        self.db_path = os.path.join(self.client_dir, "checkpoint.db")
        self.metadata_path = os.path.join(self.client_dir, "metadata.json")
        
        self.checkpointer_cm = None
        self.checkpointer_instance = None
        
    async def __aenter__(self):
        """Create and enter the checkpointer context"""
        if self.use_persistent:
            # Create client-specific directory
            if not os.path.exists(self.client_dir):
                os.makedirs(self.client_dir, exist_ok=True)
                logger.info(f"Created checkpoint directory for client {self.client_id}")
            
            # Create metadata file
            metadata = {
                "client_id": self.client_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "active",
                "checkpoint_type": "persistent_sqlite"
            }
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Use SQLite checkpointer
            logger.info(f"Using persistent SQLite checkpointer for client {self.client_id}: {self.db_path}")
            self.checkpointer_cm = AsyncSqliteSaver.from_conn_string(self.db_path)
            self.checkpointer_instance = await self.checkpointer_cm.__aenter__()
        else:
            # Use in-memory checkpointer
            logger.info(f"Using in-memory checkpointer for client {self.client_id}")
            self.checkpointer_instance = MemorySaver()
            
        return self.checkpointer_instance
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the checkpointer context and update metadata"""
        try:
            # Update metadata status
            if self.use_persistent and os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
                metadata["status"] = "completed" if exc_type is None else "failed"
                if exc_type:
                    metadata["error"] = str(exc_type.__name__)
                    
                with open(self.metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Failed to update metadata for client {self.client_id}: {e}")
        
        # Exit the checkpointer context
        if self.checkpointer_cm:
            await self.checkpointer_cm.__aexit__(exc_type, exc_val, exc_tb)
        
        # Cleanup
        self.checkpointer_instance = None
        self.checkpointer_cm = None


class DynamicGraphManager:
    """
    Manages the dynamic LangGraph workflow with per-client graph instances
    Creates individual nodes for each configured utility agent
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent.parent / "config" / "config.yaml"
        self.config = self._load_config()
        # Store graph instances per client
        self.client_graphs = {}  # client_id -> (graph, checkpointer_context)
        self.active_agents = []
        self.agent_configs = {}
        # Parse config once to determine active agents
        self._parse_active_agents()
        
    def _load_config(self) -> Dict:
        """Load configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _parse_active_agents(self):
        """Parse configuration to determine active agents"""
        utility_agents = self.config.get("orchestration", {}).get("utility_agents", [])
        self.active_agents = []
        self.agent_configs = {}
        
        for agent_config in utility_agents:
            agent_name = agent_config["name"]
            self.active_agents.append(agent_name)
            self.agent_configs[agent_name] = agent_config
            
        logger.info(f"Configured utility agents: {self.active_agents}")
    
    def get_active_agents(self) -> List[str]:
        """Get list of active utility agents"""
        return self.active_agents
    
    async def create_client_graph(self, client_id: str):
        """
        Create a graph instance with client-specific checkpointer
        
        Args:
            client_id: Unique client identifier
            
        Returns:
            Tuple of (graph, checkpointer_context)
        """
        logger.info(f"Creating graph instance for client {client_id}")
        
        # Create checkpointer context for this client
        checkpointer_context = GraphCheckpointerContext(self.config, client_id)
        checkpointer = await checkpointer_context.__aenter__()
        
        # Build graph with this checkpointer
        graph = await self._build_graph_with_checkpointer(checkpointer)
        
        # Store in cache
        self.client_graphs[client_id] = (graph, checkpointer_context)
        
        return graph, checkpointer_context
        
    async def _build_graph_with_checkpointer(self, checkpointer):
        """
        Build the LangGraph workflow with a specific checkpointer
        
        Args:
            checkpointer: The checkpointer instance to use
            
        Returns:
            Compiled graph
        """
        logger.info("Building dynamic LangGraph workflow")
        
        # Create the graph with our state schema
        workflow = StateGraph(DiscrepancyAnalyzerState)
        
        # Add fixed nodes
        workflow.add_node(PREPROCESSOR, preprocess_node)
        workflow.add_node(PROCESS_UTILITY_RESPONSES, process_utility_responses_node)
        workflow.add_node(SEND_FINAL_RESULT, send_final_result_node)
        
        # Add a simple dispatch node that triggers all utility agents and tracks dispatches
        async def dispatch_to_agents_node(state: Dict) -> Dict:
            """Simple node that just marks dispatch as started and counts agents"""
            logger.info("--- Node: dispatch_to_agents ---")
            # We'll track expected responses based on active agents
            return {
                "status": "dispatching",
                "expected_responses": len(self.active_agents),
                "utility_agents_dispatched": {},
                "awaiting_responses_from": self.active_agents.copy()
            }
        
        workflow.add_node("dispatch_to_agents", dispatch_to_agents_node)
        workflow.set_entry_point("dispatch_to_agents")
        
        # Connect dispatch to preprocessor
        workflow.add_edge("dispatch_to_agents", PREPROCESSOR)
        
        # Create nodes for each utility agent
        for agent_name in self.active_agents:
            agent_config = self.agent_configs[agent_name]
            
            # Create a dynamic node for this utility agent
            node_func = create_utility_agent_node(agent_name, agent_config)
            # Sanitize node name - replace spaces with underscores
            sanitized_name = agent_name.replace(" ", "_").replace("-", "_")
            node_name = f"{sanitized_name}_node"
            workflow.add_node(node_name, node_func)
            
            # Connect preprocessor to each utility agent (parallel execution)
            workflow.add_edge(PREPROCESSOR, node_name)
        
        logger.info(f"Created nodes for utility agents: {self.active_agents}")
        
        # Connect the rest of the graph
        # We need a node to wait after all utility agents complete
        # Since we can't have utility agents connect to nothing, we need a wait node
        async def wait_for_responses_node(state: Dict) -> Dict:
            """Node that runs after utility agents to trigger interrupt"""
            logger.info("--- Node: wait_for_responses ---")
            return {"dispatch_complete": True}
        
        workflow.add_node("wait_for_responses", wait_for_responses_node)
        
        # Connect utility agents to wait node
        for agent_name in self.active_agents:
            sanitized_name = agent_name.replace(" ", "_").replace("-", "_")
            node_name = f"{sanitized_name}_node"
            workflow.add_edge(node_name, "wait_for_responses")
        
        # Then connect to process responses
        workflow.add_edge("wait_for_responses", PROCESS_UTILITY_RESPONSES)
        workflow.add_edge(PROCESS_UTILITY_RESPONSES, SEND_FINAL_RESULT)
        workflow.add_edge(SEND_FINAL_RESULT, END)
        
        # Compile the graph with interrupt before processing responses
        compile_kwargs = {
            "interrupt_before": [PROCESS_UTILITY_RESPONSES]  # Interrupt before processing responses
        }
        
        # Add checkpointer if available
        if checkpointer:
            compile_kwargs["checkpointer"] = checkpointer
            logger.info("Graph will use checkpointing")
        else:
            logger.info("Graph will run without checkpointing")
            
        graph = workflow.compile(**compile_kwargs)
        
        logger.info("Dynamic graph compiled successfully")
        self._log_graph_structure()
        
        return graph
    def _log_graph_structure(self):
        """Log the graph structure for debugging"""
        logger.info("Graph structure:")
        logger.info("  START --> dispatch_to_agents")
        logger.info("  dispatch_to_agents --> preprocessor")
        # Sanitize agent names for node names
        node_names = [f"{agent.replace(' ', '_').replace('-', '_')}_node" for agent in self.active_agents]
        logger.info("  preprocessor --> [" + ", ".join(node_names) + "] (parallel)")
        logger.info("  [utility agent nodes] --> wait_for_responses")
        logger.info("  wait_for_responses --> [INTERRUPT] --> process_utility_responses_node")
        logger.info("  process_utility_responses_node --> send_final_result_node --> END")
    
    
    async def execute_until_interrupt(self, client_id: str, initial_state: DiscrepancyAnalyzerState, config: Dict) -> DiscrepancyAnalyzerState:
        """
        Execute the graph until it reaches the interrupt point
        
        Args:
            client_id: Client identifier
            initial_state: Initial state with request data
            config: LangGraph configuration with thread_id
            
        Returns:
            Updated state after sending to utility agents
        """
        # Get the graph for this client
        if client_id not in self.client_graphs:
            raise ValueError(f"No graph found for client {client_id}")
        
        graph, _ = self.client_graphs[client_id]
        
        # Update state with active agents and debug settings
        initial_state["active_agents"] = self.active_agents
        initial_state["agent_configs"] = self.agent_configs
        initial_state["debug"] = self.config.get("debug", {})
        initial_state["utility_agents_dispatched"] = {}  # Initialize dispatch tracking
        initial_state["environment"] = self.config.get("protocol", {}).get("environment", "local")
        
        logger.info(f"Starting graph execution for client {initial_state['client_id']}")
        
        # Run the graph until interrupt
        logger.info("About to start graph.astream...")
        print(f"\n🚀 STARTING GRAPH EXECUTION for client {client_id}", flush=True)
        print(f"   Initial state keys: {list(initial_state.keys())}", flush=True)
        print(f"   Config: {config}", flush=True)
        try:
            chunk_count = 0
            print(f"\n🔄 Starting graph.astream with {len(self.active_agents)} agents", flush=True)
            async for chunk in graph.astream(initial_state, config):
                chunk_count += 1
                print(f"\n📦 CHUNK #{chunk_count}: {list(chunk.keys()) if chunk else 'None'}", flush=True)
                logger.info(f"Received chunk from astream: {chunk}")
                if chunk:
                    logger.info(f"Workflow step: {list(chunk.keys())}")
                    
                    # Check if we hit an interrupt
                    if "__interrupt__" in chunk:
                        logger.info("Graph interrupted as expected before process_utility_responses_node")
                        break
        except Exception as e:
            logger.error(f"Error during graph.astream: {e}", exc_info=True)
            raise
        
        # Get the current state
        current_state = await graph.aget_state(config)
        
        # Log interrupt status
        if current_state.next:
            logger.info(f"Graph interrupted, next nodes: {current_state.next}")
        
        return current_state.values
    
    async def resume_with_responses(self, client_id: str, state_update: Dict, config: Dict) -> DiscrepancyAnalyzerState:
        """
        Resume graph execution after updating with utility agent responses
        
        Args:
            client_id: Client identifier
            state_update: State updates including utility agent responses
            config: LangGraph configuration with thread_id
            
        Returns:
            Final state after completion
        """
        logger.info(f"Resuming graph execution for client {state_update.get('client_id')}")
        
        # Get the graph for this client
        if client_id not in self.client_graphs:
            raise ValueError(f"No graph found for client {client_id}")
        
        graph, _ = self.client_graphs[client_id]
        
        # Update the state with utility agent responses
        await graph.aupdate_state(config, state_update)
        
        # Resume execution from where it was interrupted
        async for chunk in graph.astream(None, config):
            if chunk:
                logger.info(f"Workflow step: {list(chunk.keys())}")
                
                # Check if workflow completed
                if "__end__" in chunk:
                    logger.info("Workflow completed successfully")
                    break
        
        # Get final state
        final_state = await graph.aget_state(config)
        return final_state.values
    
    async def cleanup(self):
        """Cleanup resources including all client-specific checkpointers"""
        logger.info(f"Cleaning up resources for {len(self.client_graphs)} clients")
        
        # Clean up each client's checkpointer context
        for client_id, (graph, checkpointer_context) in list(self.client_graphs.items()):
            try:
                if checkpointer_context:
                    logger.info(f"Cleaning up checkpointer for client {client_id}")
                    await checkpointer_context.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error cleaning up checkpointer for client {client_id}: {e}")
        
        # Clear the client graphs dictionary
        self.client_graphs.clear()
        logger.info("Cleanup completed")
    
    async def cleanup_old_checkpoints(self, retention_hours: int = 24):
        """
        Clean up old checkpoint directories based on retention policy
        
        Args:
            retention_hours: Hours to retain checkpoints (default: 24)
        """
        checkpoint_config = self.config.get("checkpoint", {})
        base_path = checkpoint_config.get("base_path", "checkpoints")
        
        if not os.path.exists(base_path):
            return
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
        cleaned_count = 0
        
        logger.info(f"Starting checkpoint cleanup (retention: {retention_hours} hours)")
        
        try:
            # Iterate through client directories
            for client_dir_name in os.listdir(base_path):
                if not client_dir_name.startswith("client_"):
                    continue
                    
                client_dir = os.path.join(base_path, client_dir_name)
                if not os.path.isdir(client_dir):
                    continue
                
                # Check metadata file
                metadata_path = os.path.join(client_dir, "metadata.json")
                if not os.path.exists(metadata_path):
                    continue
                
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check if this checkpoint should be cleaned up
                    status = metadata.get("status", "unknown")
                    updated_at = metadata.get("updated_at", metadata.get("created_at"))
                    
                    if updated_at:
                        updated_time = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                        
                        # Clean up if:
                        # 1. Status is "completed" or "failed" and older than retention
                        # 2. Status is "active" but older than 2x retention (likely abandoned)
                        if (status in ["completed", "failed"] and updated_time < cutoff_time) or \
                           (status == "active" and updated_time < (datetime.now(timezone.utc) - timedelta(hours=retention_hours * 2))):
                            
                            # Extract client_id from directory name
                            client_id = client_dir_name.replace("client_", "")
                            
                            # Don't clean up if client has active graph
                            if client_id not in self.client_graphs:
                                logger.info(f"Cleaning up checkpoint for client {client_id} (status: {status})")
                                shutil.rmtree(client_dir)
                                cleaned_count += 1
                            else:
                                logger.info(f"Skipping cleanup for active client {client_id}")
                                
                except Exception as e:
                    logger.error(f"Error processing checkpoint {client_dir}: {e}")
                    
        except Exception as e:
            logger.error(f"Error during checkpoint cleanup: {e}")
        
        logger.info(f"Checkpoint cleanup completed. Cleaned {cleaned_count} directories")
        return cleaned_count
    
    def start_periodic_cleanup(self, interval_hours: int = 6, retention_hours: int = 24):
        """
        Start a background task for periodic checkpoint cleanup
        
        Args:
            interval_hours: How often to run cleanup (default: 6 hours)
            retention_hours: Hours to retain checkpoints (default: 24 hours)
        """
        async def cleanup_task():
            while True:
                try:
                    await asyncio.sleep(interval_hours * 3600)  # Convert hours to seconds
                    await self.cleanup_old_checkpoints(retention_hours)
                except asyncio.CancelledError:
                    logger.info("Checkpoint cleanup task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in periodic cleanup: {e}")
        
        # Create and return the task
        return asyncio.create_task(cleanup_task())
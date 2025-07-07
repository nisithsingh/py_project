"""
LangGraph implementation for CPAC Consistency Reviewer
Implements a workflow with executor, reviewer, and conditional routing
"""
import os
import sys
import logging
from typing import Dict, Any, List, TypedDict, Literal, Optional
from datetime import datetime, timezone
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


from data_model.schemas import (
    CPACConsistencyRequest, CPACConsistencyResult, 
    Discrepancy, LumaMessage, ClaimCategory
)

from agent.reviewer_node import ReviewerNode
from agent.improver_node import ImproverNode
from agent.preprocessor_node import PreprocessorNode
from agent.executor_node import ExecutorNode
from agent.cpac_timeline_analyzer import CPACTimelineAnalyzer

# Define the state structure
class GraphState(TypedDict):
    """State structure for the CPAC consistency graph"""
    # Input data
    luma_message: LumaMessage
    request_data: CPACConsistencyRequest
    
    # Preprocessing data
    preprocessed_request: Optional[CPACConsistencyRequest]  # Request after preprocessing
    
    # Processing data
    discrepancies: List[Discrepancy]
    analysis_result: CPACConsistencyResult
    
    # Review data
    review_decision: bool  # Changed from Literal["yes", "no", ""] to bool
    review_reason: List[str]  # Changed from str to List[str] to match ReviewOutput
    review_error: str  # Added to store review error
    review_action: Optional[Dict[str, Any]]  # Added for improvement actions
    review_analysis: Optional[str]  # Added for improvement analysis
    review_summaries: List[Dict[str, Any]]
    
    # Improvement data
    improved_discrepancies: Optional[List[Discrepancy]]  # Discrepancies after improvement
    
    # Control flow
    iteration_count: int
    max_iterations: int
    analysis_method: Optional[str]  # Track which analysis method was used
    
    # Exception tracking
    exception_list: List[Dict[str, Any]]
    
    # Final output
    final_result: Dict[str, Any]
    messages: List[Dict[str, Any]]


class CPACConsistencyGraph:
    """LangGraph implementation for CPAC Consistency Review workflow"""
    
    def __init__(self, config_path: str = "config/config.yaml", service_bus=None, agent_name: str = None):
        """Initialize the graph with configuration
        
        Args:
            config_path: Path to config file
            service_bus: Service bus instance to use for sending messages
            agent_name: Agent name for identification
        """
        # Load configuration
        config_full_path = Path(__file__).parent.parent / config_path
        with open(config_full_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Store service bus and agent name for sending responses
        self.service_bus = service_bus
        self.agent_name = agent_name or self.config['application']['service_name']
        
        # Set max iterations from config or default to 3
        self.max_iterations = self.config.get('graph', {}).get('max_review_iterations', 3)
        
        # Debug settings
        self.print_prompts = self.config.get('debug', {}).get('print_prompts', True)
        self.print_results = self.config.get('debug', {}).get('print_results', True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize the preprocessing agent
        self.preprocessor = PreprocessorNode(config_path=str(config_full_path))
        
        # Initialize the rule-based timeline analyzer
        analysis_config = self.config.get('analysis', {})
        gap_threshold = analysis_config.get('timeline_gap_threshold_months', 6)
        overlap_threshold = analysis_config.get('timeline_overlap_threshold_months', 1)
        self.timeline_analyzer = CPACTimelineAnalyzer(
            gap_threshold_months=gap_threshold,
            overlap_threshold_months=overlap_threshold
        )
        
        # Initialize the executor node for LLM-based analysis 
        self.executor = ExecutorNode(config_path=str(config_full_path))
        
        # Initialize the reviewer using prompt path from config
        reviewer_prompt_path = self.config.get('reviewer', {}).get('prompt_path', 'prompts/reviewer_prompt_list.j2')
        prompt_path = Path(__file__).parent.parent / reviewer_prompt_path
        # self.reviewer = ReviewerNode(
        #     prompt_path=str(prompt_path),
        #     config_path=str(config_full_path)
        # )

        self.reviewer = ReviewerNode(
            prompt_path=str(prompt_path),
            config_path=str(config_full_path)
        )
        
        # Initialize the improvement agent using prompt path from config
        improvement_prompt_path = self.config.get('improvement', {}).get('prompt_path', 'prompts/improvement_prompt.j2')
        prompt_path = Path(__file__).parent.parent / improvement_prompt_path
        self.improver = ImproverNode(
            prompt_path=str(prompt_path),
            config_path=str(config_full_path)
        )
        
        # Graph will be built when needed (since _build_graph is now async)
        self.graph = None
        self.saver = None
        self._saver_context = None
        
    async def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow without pre-initialized checkpointer"""
        # Create the graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("preprocessing", self.preprocessing_node)
        workflow.add_node("executor", self.executor_node)
        workflow.add_node("reviewer", self.reviewer_node)
        workflow.add_node("improver", self.improvement_node)
        workflow.add_node("send_to_bus", self.send_to_bus_node)
        
        # Add edges
        workflow.add_edge("preprocessing", "executor")
        workflow.add_edge("executor", "reviewer")
        workflow.add_edge("reviewer", "improver")
        workflow.add_edge("improver", "send_to_bus")
        # Add conditional edge from reviewer
        # workflow.add_conditional_edges(
        #     "reviewer",
        #     self.review_router,
        #     {
        #         "improve": "improvement",  # Go to improvement if review fails
        #         "finish": "send_to_bus"  # Send to bus if review passes
        #     }
        # )
        
        # # Add conditional edge from improvement
        # workflow.add_conditional_edges(
        #     "improvement",
        #     self.improvement_router,
        #     {
        #         "review": "reviewer",  # Continue to reviewer if more iterations needed
        #         "finish": "send_to_bus"  # Finish if max iterations reached
        #     }
        # )
        
        # Add edge from send_to_bus to END
        workflow.add_edge("send_to_bus", END)
        
        # Set entry point to preprocessing
        workflow.set_entry_point("preprocessing")
        
        # Return uncompiled workflow - we'll compile with fresh checkpointer each time
        return workflow
    
    async def _ensure_graph_built(self):
        """Ensure the graph workflow is built before use"""
        if self.graph is None:
            self.graph = await self._build_graph()
    
    async def _create_fresh_checkpointer(self):
        """Create a fresh AsyncSqliteSaver for each graph execution"""
        db_path = "checkpoints.sqlite"
        
        self.logger.debug(f"Creating fresh AsyncSqliteSaver for: {db_path}")
        
        # Create fresh context manager and enter it
        context_manager = AsyncSqliteSaver.from_conn_string(db_path)
        saver = await context_manager.__aenter__()
        
        return saver, context_manager
    
    async def close(self):
        """Clean up resources"""
        if self._saver_context is not None:
            await self._saver_context.__aexit__(None, None, None)
            self._saver_context = None
            self.saver = None
    
    async def preprocessing_node(self, state: GraphState) -> GraphState:
        """
        Preprocessing node: Standardizes date formats and sorts claims chronologically
        """
        self.logger.info("Preprocessing node - Standardizing data format")
        
        # Get original request data
        request_data = state['request_data']
        
        # Run preprocessing
        preprocessed_request = await self.preprocessor.preprocess_claims(request_data)
        
        # Update state
        state['preprocessed_request'] = preprocessed_request
        
        # Add message
        state['messages'].append({
            "type": "preprocessing",
            "claims_count": len(preprocessed_request.claims),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        self.logger.info(f"Preprocessing completed - {len(preprocessed_request.claims)} claims processed")
            
        # Display preprocessing results if enabled
        if self.print_results:
            print("\n" + "="*60)
            print("PREPROCESSING AGENT RESULTS")
            print("="*60)
            print(f"Total claims processed: {len(preprocessed_request.claims)}")
            
            # Display based on category and available fields
            print(f"Category: {state['request_data'].claim_category}")
            
            # Check if claims have timeline data (start_date/end_date)
            has_timeline = any(hasattr(claim.cpac_data, 'start_date') for claim in preprocessed_request.claims)
            
            if has_timeline:
                print("Claims sorted chronologically by start date")
            
            for claim in preprocessed_request.claims:
                if hasattr(claim.cpac_data, 'employer_name') and claim.cpac_data.employer_name:
                    # Employment
                    start_date = getattr(claim.cpac_data, 'start_date', 'N/A') or "N/A"
                    end_date = getattr(claim.cpac_data, 'end_date', 'Present') or "Present"
                    employer = claim.cpac_data.employer_name
                    compensation = getattr(claim.cpac_data, 'annual_compensation', 'N/A')
                    print(f"  Claim {claim.claim_id}: {start_date} to {end_date} | {employer} ({compensation})")
                elif hasattr(claim.cpac_data, 'company_name') and claim.cpac_data.company_name:
                    # Business profit
                    company = claim.cpac_data.company_name
                    ownership = getattr(claim.cpac_data, 'ownership_percentage', 'N/A')
                    start_date = getattr(claim.cpac_data, 'start_date', 'N/A') or "N/A"
                    end_date = getattr(claim.cpac_data, 'end_date', 'Present') or "Present"
                    role = getattr(claim.cpac_data, 'business_role', 'N/A')
                    print(f"  Claim {claim.claim_id}: {start_date} to {end_date} | {company} ({ownership}% ownership, {role})")
                elif hasattr(claim.cpac_data, 'inheritance_date'):
                    # Inheritance
                    date = getattr(claim.cpac_data, 'inheritance_date', 'N/A') or "N/A"
                    source = getattr(claim.cpac_data, 'inheritance_source', 'N/A')
                    amount = getattr(claim.cpac_data, 'inheritance_amount', 'N/A')
                    currency = getattr(claim.cpac_data, 'inheritance_currency', '')
                    print(f"  Claim {claim.claim_id}: {date} | {source} ({amount} {currency})")
                else:
                    # Generic fallback
                    summary = getattr(claim.cpac_data, 'summary', 'No summary')
                    print(f"  Claim {claim.claim_id}: {summary[:80]}...")
            print("="*60 + "\n")
        
        return state
    
    # Commented out old LLM-based executor node
    # async def executor_node(self, state: GraphState) -> GraphState:
    #     """
    #     Executor node: Runs the CPAC consistency reviewer with exception tracking
    #     """
    #     self.logger.info(f"Executor node - Iteration {state['iteration_count'] + 1}/{state['max_iterations']}")
    #     
    #     try:
    #         # Use preprocessed request if available, otherwise use original
    #         request_data = state.get('preprocessed_request') or state['request_data']
    #         
    #         # Reset CPAC agent's exception list for this iteration
    #         self.cpac_agent.exception_list = []
    #         
    #         # Analyze claims using the CPAC agent
    #         discrepancies = await self.cpac_agent._analyze_claims_with_llm(request_data)
    #         
    #         # Merge CPAC agent exceptions with graph exceptions
    #         if self.cpac_agent.exception_list:
    #             state['exception_list'].extend(self.cpac_agent.exception_list)
    #             self.logger.warning(f"Executor collected {len(self.cpac_agent.exception_list)} exceptions")
    #         
    #         # Create result
    #         result = CPACConsistencyResult(
    #             claim_category=request_data.claim_category,
    #             discrepancies=discrepancies
    #         )
    #         
    #         # Update state
    #         state['discrepancies'] = discrepancies
    #         state['analysis_result'] = result
    #         # Don't increment iteration count here - only in improvement node
    #         
    #         # Add message
    #         state['messages'].append({
    #             "type": "executor",
    #             "iteration": state['iteration_count'],
    #             "discrepancies_found": len(discrepancies),
    #             "timestamp": datetime.now(timezone.utc).isoformat()
    #         })
    #         
    #         self.logger.info(f"Executor found {len(discrepancies)} discrepancies")
    
    async def executor_node(self, state: GraphState) -> GraphState:
        """
        Executor node: Uses rule-based analyzer for employment, LLM for other categories
        """
        self.logger.info(f"Executor node - Iteration {state['iteration_count'] + 1}/{state['max_iterations']}")
        
        # Use preprocessed request if available, otherwise use original
        request_data = state.get('preprocessed_request') or state['request_data']
        
        # Check claim category
        if request_data.claim_category == ClaimCategory.EMPLOYMENT:
            # Use rule-based analyzer for employment timeline analysis
            self.logger.info("Using rule-based analyzer for employment claims")
            discrepancies = self.timeline_analyzer.analyze_employment_timeline(request_data)
            analysis_method = "rule_based"
        else:
            # Use LLM for other categories (inheritance, business_profit, etc.)
            self.logger.info(f"Using LLM analyzer for {request_data.claim_category} claims")
            
            # Reset executor node's exception list for this iteration
            self.executor.clear_exceptions()
            
            # Analyze claims using the executor node with category-appropriate prompt
            discrepancies = await self.executor.analyze_claims_with_llm(request_data)
            
            # Merge executor node exceptions with graph exceptions
            executor_exceptions = self.executor.get_exceptions()
            if executor_exceptions:
                state['exception_list'].extend(executor_exceptions)
                self.logger.warning(f"Executor collected {len(executor_exceptions)} exceptions")
            
            analysis_method = "llm_based"
        
        # Create result
        result = CPACConsistencyResult(
            claim_category=request_data.claim_category,
            discrepancies=discrepancies
        )
        
        # Update state
        state['discrepancies'] = discrepancies
        state['analysis_result'] = result
        state['analysis_method'] = analysis_method
        # Don't increment iteration count here - only in improvement node
        
        # Add message
        state['messages'].append({
            "type": "executor",
            "iteration": state['iteration_count'],
            "discrepancies_found": len(discrepancies),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "analysis_method": analysis_method
        })
        
        self.logger.info(f"{analysis_method} executor found {len(discrepancies)} discrepancies")
        
        # Display executor results if enabled
        if self.print_results:
            print("\n" + "="*60)
            print(f"{analysis_method.upper()} EXECUTOR RESULTS - Found {len(discrepancies)} discrepancies")
            print("="*60)
            for i, disc in enumerate(discrepancies, 1):
                print(f"\nDiscrepancy {i}:")
                print(f"  Type: {disc.discrepancy_type}")
                print(f"  Description: {disc.description}")
                print(f"  Affected Claims: {disc.affected_claim_ids}")
            print("="*60 + "\n")
        
        return state
    
    async def reviewer_node(self, state: GraphState) -> GraphState:
        """
        Reviewer node: Reviews the discrepancy results and decides pass/fail with exception tracking
        """
        self.logger.info("Reviewer node - Analyzing discrepancy results")
        
        # Get analysis result and request data
        analysis_result = state['analysis_result']
        # Use preprocessed request if available, otherwise use original
        request_data = state.get('preprocessed_request') or state['request_data']
        
        # Use the reviewer to analyze discrepancies
        # The ReviewerNode expects CPACConsistencyResult and CPACConsistencyRequest
        # Check if binary classification is enabled
        use_binary = self.config.get('graph', {}).get('use_binary_classification', False)
        
        if use_binary:
            self.logger.info("Using binary classification reviewer with logprobs")
            review_output = await self.reviewer.process_request_with_binary_classification(
                discrepancy_objects=analysis_result,
                cpac_object=request_data
            )
        else:
            review_output = await self.reviewer.process_request(
                discrepancy_objects=analysis_result,
                cpac_object=request_data
            )
        
        # Extract review decision and analysis from ReviewOutput
        state['review_decision'] = review_output.review_decision
        state['review_reason'] = review_output.analysis  # This is now a List[str] (renamed from reason)
        state['review_error'] = review_output.error
        state['review_action'] = review_output.action  # Contains remove/revise/approve actions
        
        # Add review summary to shared resources
        if 'review_summaries' not in state:
            state['review_summaries'] = []
        
        review_summary = {
            "iteration": state['iteration_count'],
            "decision": review_output.review_decision,
            "reasons": review_output.analysis,  # Renamed from reason
            "error": review_output.error,
            "action": review_output.action,
            "discrepancy_count": len(analysis_result.discrepancies) if analysis_result else 0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        state['review_summaries'].append(review_summary)
        
        # Add message
        state['messages'].append({
            "type": "reviewer",
            "iteration": state['iteration_count'],
            "decision": state['review_decision'],
            "reasons": state['review_reason'],
            "error": state['review_error'],
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        self.logger.info(f"Reviewer decision: {state['review_decision']} - Reasons: {len(state['review_reason'])}")
        if state['review_error']:
            self.logger.warning(f"Review error: {state['review_error']}")
        
        return state
    
    async def improvement_node(self, state: GraphState) -> GraphState:
        """
        Improvement node: Improves discrepancies based on reviewer feedback
        """
        self.logger.info("Improvement node - Processing reviewer feedback")
        
        # Get current discrepancies and reviewer feedback
        current_discrepancies = state['discrepancies']
        review_action = state.get('review_action', {})
        review_reasons = state['review_reason']  # This contains the analysis list from reviewer
        
        # Call improvement agent to process the feedback
        improved_result = await self.improver.improve_discrepancies(
            discrepancies=current_discrepancies,
            review_action=review_action,
            review_analysis=None,  # Deprecated - using review_reasons instead
            review_reasons=review_reasons  # Contains the actual analysis list
        )
        
        # Update state with improved discrepancies
        state['improved_discrepancies'] = improved_result
        state['discrepancies'] = improved_result
        
        # Get the request data (preprocessed or original)
        request_data = state.get('preprocessed_request') or state['request_data']
        
        # Update analysis result with improved discrepancies
        state['analysis_result'] = CPACConsistencyResult(
            claim_category=request_data.claim_category,
            discrepancies=improved_result
        )
        
        # Increment iteration count
        state['iteration_count'] += 1
        
        # Add message
        state['messages'].append({
            "type": "improvement",
            "iteration": state['iteration_count'],
            "original_discrepancy_count": len(current_discrepancies),
            "improved_discrepancy_count": len(improved_result),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        self.logger.info(f"Improvement completed: {len(current_discrepancies)} -> {len(improved_result)} discrepancies")
        
        return state
    
    # def improvement_router(self, state: GraphState) -> Literal["review", "finish"]:
    #     """
    #     Router function to decide next step after improvement
    #     """
    #     iteration = state['iteration_count']
    #     max_iter = state['max_iterations']
        
    #     self.logger.info(f"Improvement router - Iteration: {iteration}/{max_iter}")
        
    #     if iteration >= max_iter:
    #         self.logger.info(f"Routing to finish (max iterations {max_iter} reached)")
    #         return "finish"
    #     else:
    #         self.logger.info("Routing to reviewer (more iterations available)")
    #         return "review"
    
    # def review_router(self, state: GraphState) -> Literal["improve", "finish"]:
    #     """
    #     Router function to decide next step based on review decision
    #     """
    #     decision = state['review_decision']  # True / False: bool
    #     iteration = state['iteration_count']
    #     max_iter = state['max_iterations']
        
    #     self.logger.info(f"Review router - Decision: {decision}, Iteration: {iteration}/{max_iter}")
        
    #     if decision:
    #         self.logger.info("Routing to finish (review passed)")
    #         return "finish"
    #     elif iteration >= max_iter:
    #         self.logger.info(f"Routing to finish (max iterations {max_iter} reached)")
    #         return "finish"
    #     else:
    #         self.logger.info("Routing to improve (review failed, more iterations available)")
    #         return "improve"
    
    async def send_to_bus_node(self, state: GraphState) -> GraphState:
        """
        Send to bus node: Prepares the final result (actual sending handled by main.py)
        """
        self.logger.info("Send to bus node - Preparing final result")
        
        # Get the original LUMA message and result
        luma_message = state['luma_message']
        result = state['analysis_result']
        
        # Calculate total processing time from messages
        if state['messages']:
            start_time = datetime.fromisoformat(state['messages'][0]['timestamp'].replace('Z', '+00:00'))
            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()
        else:
            processing_time = 0.0
        
        # Prepare final result with review metadata
        final_result = {
            "claim_category": result.claim_category.value,
            "discrepancies": [d.model_dump() for d in result.discrepancies],
            "review_metadata": {
                "total_iterations": state['iteration_count'],
                "final_decision": state['review_decision'],
                "review_reasons": state['review_reason'],
                "review_error": state.get('review_error', ''),
                "review_summaries": state.get('review_summaries', []),
                "processing_time_seconds": processing_time,
                "analysis_method": state.get('analysis_method', 'unknown'),
                "total_claims_analyzed": len(state['request_data'].claims),
                "total_discrepancies_found": len(result.discrepancies),
                "preprocessing_applied": state.get('preprocessed_request') is not None,
                "messages": state['messages']
            }
        }
        
        # Add exception information if any
        if state['exception_list']:
            final_result['exceptions'] = {
                "count": len(state['exception_list']),
                "details": state['exception_list']
            }
        
        state['final_result'] = final_result
        
        # Add final message
        state['messages'].append({
            "type": "final_result_prepared",
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        self.logger.info("Successfully prepared final result")
        
        return state
    
    async def process_request(self, luma_message: LumaMessage, request_data: CPACConsistencyRequest, 
                            exception_list: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a CPAC consistency review request through the graph with exception tracking
        
        Args:
            luma_message: Original LUMA message
            request_data: CPAC consistency request data
            exception_list: List of exceptions from graph agent
            
        Returns:
            Final result dictionary
        """
        # Initialize state
        initial_state = GraphState(
            luma_message=luma_message,
            request_data=request_data,
            preprocessed_request=None,  # Will be populated by preprocessing node
            discrepancies=[],
            analysis_result=None,
            review_decision=False,
            review_reason=[],
            review_error="",
            review_action=None,
            review_analysis=None,
            review_summaries=[],
            improved_discrepancies=None,
            iteration_count=0,
            max_iterations=self.max_iterations,
            analysis_method=None,
            exception_list=exception_list or [],
            final_result={},
            messages=[]
        )
        
        # Ensure graph workflow is built before execution
        await self._ensure_graph_built()
        
        # Create fresh checkpointer for this execution
        self.logger.info("Creating fresh checkpointer for graph execution")
        saver, saver_context = await self._create_fresh_checkpointer()
        
        # Compile graph with fresh checkpointer
        compiled_graph = self.graph.compile(checkpointer=saver)
        
        # Create config for checkpointing with unique thread_id
        # Use conversation_id and job_id from LUMA message to create unique thread
        thread_id = f"{luma_message.payload.conversation_id}_{luma_message.payload.job_id}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # Run the graph
        self.logger.info(f"Starting graph execution with thread_id: {thread_id}")
        
        try:
            final_state = await compiled_graph.ainvoke(initial_state, config=config)
        except Exception as graph_error:
            self.logger.error(f"Graph execution failed: {graph_error}")
            self.logger.error(f"Error type: {type(graph_error)}")
            raise
        finally:
            # Clean up the checkpointer context after execution
            try:
                await saver_context.__aexit__(None, None, None)
                self.logger.debug("Checkpointer context cleaned up")
            except Exception as cleanup_error:
                self.logger.warning(f"Error cleaning up checkpointer: {cleanup_error}")
        
        self.logger.info(f"Graph execution completed after {final_state['iteration_count']} iterations")
        
        return final_state['final_result']


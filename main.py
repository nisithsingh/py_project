#!/usr/bin/env python3
"""
Main entry point for CPAC Consistency Reviewer Agent using AI Foundation
"""
import os
import sys
import asyncio
import logging
import signal
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add ai_foundation to path
ai_foundation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ai_foundation'))
sys.path.append(ai_foundation_path)

from ai_foundation.protocol.luma_protocol.luma import LumaProtocol
from ai_foundation.protocol.luma_protocol.models import (
    LumaMessage, AgentMetadata, MessageType, Status, Source, Target, Payload, Task
)
from data_model.schemas import CPACConsistencyRequest
from graph.cpac_consistency_graph import CPACConsistencyGraph


class CPACConsistencyService:
    """CPAC Consistency Reviewer using AI Foundation LUMA Protocol"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.protocol = None
        self.graph = None
        self.agent_card = None
        self.running = False
        
    async def start(self):
        """Start the CPAC Consistency service with LUMA protocol"""
        try:
            # Load agent card
            agent_card_path = Path(__file__).parent / "agent_card.json"
            with open(agent_card_path, 'r', encoding='utf-8') as f:
                agent_card_data = json.load(f)
            
            self.agent_card = AgentMetadata(**agent_card_data)
            self.agent_name = self.agent_card.name
            
            # Load configuration
            import yaml
            config_path = Path(__file__).parent / "config" / "config.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # Get environment settings
            self.environment = os.getenv('ENVIRONMENT', 'local')
            self.service_bus_namespace = os.getenv('SERVICE_BUS_NAMESPACE', 'local_namespace')
            self.result_queue = os.getenv('RESULT_QUEUE', 'orchestrator')
            self.blob_storage_account = os.getenv('BLOB_STORAGE_ACCOUNT', 'local_storage')
            self.blob_storage_container = os.getenv('BLOB_STORAGE_CONTAINER', 'sow-automation')
            
            # Create directories for local environment
            if self.environment == 'local':
                # Create service bus namespace directory
                os.makedirs(self.service_bus_namespace, exist_ok=True)
                # Create blob storage directory
                os.makedirs(self.blob_storage_container, exist_ok=True)
                # Create agent cards directory
                agent_cards_dir = os.path.join(self.blob_storage_container, 'agent-cards')
                os.makedirs(agent_cards_dir, exist_ok=True)
                self.logger.info(f"Created local directories for namespace: {self.service_bus_namespace}")
            
            # Initialize LUMA protocol
            self.protocol = LumaProtocol(
                agent_card=self.agent_card,
                blob_storage_account=self.blob_storage_account,
                blob_storage_container=self.blob_storage_container,
                service_bus_namespace=self.service_bus_namespace,
                result_queue=self.result_queue,
                environment=self.environment
            )
            
            # Initialize graph for processing
            self.graph = CPACConsistencyGraph(
                config_path="config/config.yaml",
                service_bus=None,  # Will use protocol for sending
                agent_name=self.agent_name
            )
            
            self.logger.info(f"Starting {self.agent_name} with LUMA protocol")
            self.logger.info(f"Environment: {self.environment}")
            self.logger.info(f"Service Bus Namespace: {self.service_bus_namespace}")
            
            # Print startup banner
            self._print_banner()
            
            # Register and start listening
            self.running = True
            self.protocol.register(self._handle_message)
            
        except Exception as e:
            self.logger.error(f"Failed to start service: {str(e)}", exc_info=True)
            raise
    
    async def _handle_message(self, message: LumaMessage):
        """Handle incoming LUMA messages with fail-fast exception handling"""
        task_request = None
        try:
            self.logger.info(f"Received message: {message.payload.task_id}")
            
            # Validate message type
            if message.message_type != MessageType.REQEST:
                error_msg = f"Invalid message type: {message.message_type}. Expected: {MessageType.REQEST}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Extract request data
            task_request = message.payload.task.request
            
            # Validate required fields
            if not task_request:
                raise ValueError("Missing task request data")
            
            if "claim_category" not in task_request:
                raise ValueError("Missing required field: claim_category")
            
            if "claims" not in task_request:
                raise ValueError("Missing required field: claims")
            
            # Create CPACConsistencyRequest from task data
            request_data = CPACConsistencyRequest(
                claim_category=task_request["claim_category"],
                claims=task_request["claims"],
                cpac_text=task_request.get("cpac_text", "")
            )
            
            # Process through graph - any exception here will propagate up
            result = await self.graph.process_request(
                luma_message=message,
                request_data=request_data
            )
            
            # Check if result contains errors
            if result.get("error"):
                raise RuntimeError(f"Graph processing failed: {result['error']}")
            
            # Send result back using protocol
            await self._send_result(message, result)
            
            self.logger.info(f"Completed processing: {message.payload.task_id}")
            
        except Exception as e:
            # Log the exception with full traceback
            self.logger.error(f"Error processing message {message.payload.task_id}: {str(e)}", exc_info=True)
            
            # Create LUMA error format
            error_details = {
                "type": "technical error",
                "reason": self._format_error_reason(e, message.payload.task_id)
            }
            
            # Send error result back to orchestrator
            await self._send_error_result(message, error_details)
    
    async def _send_result(self, original_message: LumaMessage, result: dict):
        """Send successful result back to orchestrator"""
        result_message = LumaMessage(
            message_type=MessageType.RESULT,
            source=Source(
                created_ts=datetime.now().isoformat() + 'Z',
                name=self.agent_name
            ),
            target=Target(
                name=original_message.source.name
            ),
            payload=Payload(
                thread_id=original_message.payload.thread_id,
                job_id=original_message.payload.job_id,
                task_id=original_message.payload.task_id,
                parent_job_id=original_message.payload.parent_job_id,
                parent_task_id=original_message.payload.parent_task_id,
                conversation_id=original_message.payload.conversation_id,
                task=Task(
                    result=result,
                    status=Status.COMPLETED
                )
            ),
            conversation_history=original_message.conversation_history,
            shared_resources=original_message.shared_resources
        )
        
        await self.protocol.send_result(result_message)
    
    async def _send_error_result(self, original_message: LumaMessage, error_details: dict):
        """Send error result back to orchestrator using proper LUMA error format"""
        result_message = LumaMessage(
            message_type=MessageType.RESULT,
            source=Source(
                created_ts=datetime.now().isoformat() + 'Z',
                name=self.agent_name
            ),
            target=Target(
                name=original_message.source.name
            ),
            payload=Payload(
                thread_id=original_message.payload.thread_id,
                job_id=original_message.payload.job_id,
                task_id=original_message.payload.task_id,
                parent_job_id=original_message.payload.parent_job_id,
                parent_task_id=original_message.payload.parent_task_id,
                conversation_id=original_message.payload.conversation_id,
                task=Task(
                    error={
                        "type": error_details["type"],
                        "reason": error_details["reason"]
                    },
                    status=Status.FAILED
                )
            ),
            conversation_history=original_message.conversation_history,
            shared_resources=original_message.shared_resources
        )
        
        await self.protocol.send_result(result_message)
    
    def _format_error_reason(self, exception: Exception, task_id: str) -> str:
        """Format error reason with exception type, location details, and full stacktrace"""
        import traceback
        
        # Get exception type
        error_type = type(exception).__name__
        
        # Get error location from traceback
        tb = traceback.extract_tb(exception.__traceback__)
        error_location = "unknown"
        
        # Find the most relevant frame (in our codebase)
        for frame in reversed(tb):
            if any(component in frame.filename for component in ['agent', 'graph', 'utils']):
                error_location = f"{frame.name}() in {frame.filename.split('/')[-1]}:{frame.lineno}"
                break
        
        # Get full stacktrace
        full_stacktrace = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        
        # Format comprehensive error reason with stacktrace
        reason = f"{error_type}: {str(exception)}"
        if error_location != "unknown":
            reason += f" | Location: {error_location}"
        
        reason += f"\n\nFull Stacktrace:\n{full_stacktrace}"
        
        return reason
    
    async def stop(self):
        """Stop the service gracefully"""
        self.logger.info("Stopping CPAC Consistency Reviewer service...")
        self.running = False
        # No need to stop protocol as it's handled by register
        self.logger.info("Service stopped")
    
    def _print_banner(self):
        """Print startup banner"""
        print("\n" + "="*80)
        print("CPAC CONSISTENCY REVIEWER - LUMA PROTOCOL")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Agent Name: {self.agent_name}")
        print(f"Environment: {self.environment}")
        print(f"Version: {self.agent_card.version}")
        print(f"Capabilities: {', '.join(self.agent_card.capabilities)}")
        print(f"Queue: {self.agent_name}")
        print("\nThe agent is now listening for LUMA messages...")
        print("Press Ctrl+C to stop the service")
        print("="*80 + "\n")


async def main():
    """Main entry point"""
    # Setup logging
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    # Configure logging format
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('cpac_consistency_service.log')
        ]
    )
    
    # Reduce noise from some libraries
    logging.getLogger('azure').setLevel(logging.WARNING)
    logging.getLogger('uamqp').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    
    # Create service instance
    service = CPACConsistencyService()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        logger.info("Shutting down...")
        # Force exit since LUMA protocol doesn't handle shutdown gracefully
        os._exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the service
        await service.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Service failed: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        await service.stop()


if __name__ == "__main__":
    # Run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nService stopped by user")
    except Exception as e:
        print(f"\nService error: {str(e)}")
        sys.exit(1)
#!/usr/bin/env python3
"""
LUMA-based Orchestrator Example
This demonstrates how an orchestrator would send requests using AI Foundation LUMA protocol
"""
import os
import sys
import asyncio
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
from utils.test_data_loader import TestDataLoader


class LumaOrchestrator:
    """Example orchestrator using LUMA protocol"""
    
    def __init__(self):
        # Load orchestrator agent card
        agent_card_path = Path(__file__).parent / "orchestrator_agent_card.json"
        with open(agent_card_path, 'r',encoding='utf-8') as f:
            agent_card_data = json.load(f)
        
        self.agent_card = AgentMetadata(**agent_card_data)
        self.agent_name = self.agent_card.name
        
        # Get environment settings
        self.environment = os.getenv('ENVIRONMENT', 'local')
        self.service_bus_namespace = os.getenv('SERVICE_BUS_NAMESPACE', 'local_namespace')
        self.blob_storage_account = os.getenv('BLOB_STORAGE_ACCOUNT', 'local_storage')
        self.blob_storage_container = os.getenv('BLOB_STORAGE_CONTAINER', 'sow-automation')
        
        # Create directories for local environment
        if self.environment == 'local':
            os.makedirs(self.service_bus_namespace, exist_ok=True)
            os.makedirs(self.blob_storage_container, exist_ok=True)
            agent_cards_dir = os.path.join(self.blob_storage_container, 'agent-cards')
            os.makedirs(agent_cards_dir, exist_ok=True)
            
            # Clean up old orchestrator response queue
            old_response = os.path.join(self.service_bus_namespace, "orchestrator.json")
            if os.path.exists(old_response):
                os.remove(old_response)
                print(f" Cleaned up old response queue")
        
        # Initialize LUMA protocol
        self.protocol = LumaProtocol(
            agent_card=self.agent_card,
            blob_storage_account=self.blob_storage_account,
            blob_storage_container=self.blob_storage_container,
            service_bus_namespace=self.service_bus_namespace,
            result_queue=self.agent_name,  # Where to receive results (matches agent name)
            environment=self.environment
        )
        
        self.received_response = None
        
    async def handle_message(self, message: LumaMessage):
        """Handle incoming LUMA messages (results from CPAC agent)"""
        print(f"\n Received response from {message.source.name}")
        self.received_response = message
        
    async def send_test_request(self, test_file: str = "employment_test_data_1"):
        """Send a test request to CPAC Consistency Reviewer"""
        
        # Load test data
        loader = TestDataLoader()
        test_data = loader.load_and_prepare_test(test_file)
        
        # Create request
        request = CPACConsistencyRequest(
            claim_category=test_data['claim_category'],
            claims=test_data['claims'],
            cpac_text=test_data['cpac_text']
        )
        
        # Generate numeric task_id
        task_id = int(datetime.now().strftime('%Y%m%d%H%M%S'))
        
        # Create LUMA message
        message = LumaMessage(
            message_type=MessageType.REQEST,
            source=Source(
                created_ts=datetime.now().isoformat() + 'Z',
                name=self.agent_name  # Use agent name from card
            ),
            target=Target(
                name="cpac_consistency_reviewer"
            ),
            payload=Payload(
                conversation_id=12345,
                job_id=67890,
                task_id=task_id,
                thread_id="test-thread-123",
                task=Task(
                    request=request.model_dump()
                )
            ),
            conversation_history="",
            shared_resources={"test_name": test_data['test_name']}
        )
        
        print(f"\n Sending request to CPAC Consistency Reviewer")
        print(f"   Test: {test_data['test_name']}")
        print(f"   Task ID: {task_id}")
        print(f"   Claims: {len(request.claims)}")
        
        # Send using LUMA protocol
        thread_id = await self.protocol.send_request(message)
        print(f"   Thread ID: {thread_id}")
        
        # Wait for response by polling the file directly
        print("\n Waiting for response...")
        timeout = 120  # seconds
        start_time = datetime.now()
        # Note: ai_foundation doesn't add "-queue" suffix for result queues
        response_file = os.path.join(self.service_bus_namespace, "orchestrator.json")
        
        while True:
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout:
                print(f"\n Timeout after {timeout} seconds")
                print(f"   No response file found at: {response_file}")
                return
            
            # Check if response file exists
            if os.path.exists(response_file):
                print(f"\n Found response file!")
                try:
                    with open(response_file, 'r', encoding='utf-8') as f:
                        response_data = json.load(f)
                    
                    # Convert to LumaMessage (already imported at top)
                    response = LumaMessage(**response_data)
                    
                    # Display response
                    self._display_response(response)
                    
                    # Clean up the response file
                    os.remove(response_file)
                    break
                    
                except Exception as e:
                    print(f"\n Error reading response: {e}")
                    break
            
            await asyncio.sleep(1.0)
    
    def _display_response(self, response: LumaMessage):
        """Display the response in a formatted way"""
        print("\n" + "="*80)
        print("RESPONSE FROM CPAC CONSISTENCY REVIEWER")
        print("="*80)
        
        if response.payload.task.status == Status.COMPLETED:
            result = response.payload.task.result
            discrepancies = result.get('discrepancies', [])
            metadata = result.get('review_metadata', {})
            
            print(f"\n Status: COMPLETED")
            print(f"  Task ID: {response.payload.task_id}")
            print(f"  Processing Time: {metadata.get('processing_time_seconds', 0):.2f} seconds")
            print(f"  Total Iterations: {metadata.get('total_iterations', 0)}")
            print(f"  Final Decision: {'PASS' if metadata.get('final_decision') else 'FAIL'}")
            print(f"\n  Total Discrepancies Found: {len(discrepancies)}")
            
            if discrepancies:
                print("\n  Discrepancies:")
                for i, disc in enumerate(discrepancies, 1):
                    print(f"\n  [{i}] Type: {disc['discrepancy_type']}")
                    print(f"      Description: {disc['description']}")
                    print(f"      Reason: {disc['reason']}")
                    print(f"      Affected Claims: {disc['affected_claim_ids']}")
                    print(f"      Recommendation: {disc['recommendation']}")
        else:
            print(f"\n Status: FAILED")
            print(f"  Error: {response.payload.task.result.get('error', 'Unknown error')}")
        
        print("="*80 + "\n")
    
    def start(self):
        """Start the orchestrator and register with LUMA protocol"""
        print("\n" + "="*80)
        print("LUMA ORCHESTRATOR EXAMPLE")
        print("="*80)
        print(f"Environment: {self.environment}")
        print(f"Agent: {self.agent_card.name}")
        print("="*80 + "\n")
        
        # Register with LUMA protocol
        self.protocol.register(self.handle_message)


async def main():
    """Main function"""
    # Get test file from command line or use default
    test_file = sys.argv[1] if len(sys.argv) > 1 else "employment_testcase_1_orig_luma"
    
    # Create orchestrator
    orchestrator = LumaOrchestrator()
    
    # DON'T call start() because register() blocks!
    # Just send the request directly
    print("\n" + "="*80)
    print("LUMA ORCHESTRATOR EXAMPLE")
    print("="*80)
    print(f"Environment: {orchestrator.environment}")
    print(f"Agent: {orchestrator.agent_card.name}")
    print("="*80 + "\n")
    
    # Send test request
    await orchestrator.send_test_request(test_file)
    
    print("\n Test completed")


if __name__ == "__main__":
    asyncio.run(main())
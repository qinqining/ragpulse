"""
Simple test script to verify the AI agent functionality.
"""
import asyncio
import os
from rag.agent.service import create_agent


def test_agent_basic():
    """Test basic agent functionality."""
    print("Testing RAGPulse AI Agent...")

    # Create an agent instance
    agent = create_agent(
        user_id="test_user",
        dept_tag="test_dept",
        kb_id="test_kb"
    )

    print("✓ Agent service created successfully")

    # Test a simple chat interaction
    try:
        result = agent.chat(
            query="你好，这是一个测试。",
            session_id="test_session_001",
            memory_id="test_memory",
            top_k=2,
            use_rag=False,  # Disable RAG for this test
            include_long_term_memory=False  # Disable long-term memory for this test
        )

        print(f"✓ Chat response received: {result['response'][:100]}...")
        print(f"✓ Session ID: {result['session_id']}")
        print(f"✓ Memory ID: {result['memory_id']}")
        print("✓ Basic agent functionality test passed!")

    except Exception as e:
        print(f"✗ Error during agent chat test: {e}")
        return False

    return True


def test_memory_management():
    """Test basic memory management functionality."""
    print("\nTesting memory management...")

    from rag.agent.memory_manager import AgentMemoryManager

    # Create a memory manager instance
    mem_manager = AgentMemoryManager(user_id="test_user")

    print("✓ Memory manager created successfully")

    # Test saving context memory
    success = mem_manager.save_context_memory(
        session_id="test_session_001",
        memory_id="test_memory",
        message_data={
            'content': 'This is a test message',
            'metadata': {'test': True}
        },
        message_type='user'
    )

    if success:
        print("✓ Context memory saved successfully")
    else:
        print("✗ Failed to save context memory")
        return False

    # Test retrieving context memory
    context = mem_manager.get_context_memory(
        session_id="test_session_001",
        memory_id="test_memory",
        limit=5
    )

    if context:
        print(f"✓ Retrieved context memory: {len(context)} messages")
    else:
        print("? No context memory found (this may be normal)")

    # Test saving long-term memory
    success = mem_manager.save_long_term_memory(
        memory_id="test_memory",
        content="This is a long-term memory test",
        importance_score=0.8,
        tags=["test", "important"]
    )

    if success:
        print("✓ Long-term memory saved successfully")
    else:
        print("✗ Failed to save long-term memory")
        return False

    print("✓ Memory management test passed!")
    return True


if __name__ == "__main__":
    print("Starting RAGPulse AI Agent tests...\n")

    success1 = test_agent_basic()
    success2 = test_memory_management()

    if success1 and success2:
        print("\n🎉 All AI agent tests passed!")
    else:
        print("\n❌ Some tests failed.")
        exit(1)
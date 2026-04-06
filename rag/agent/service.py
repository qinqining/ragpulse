"""
Core agent service for RAGPulse with memory capabilities and RAG integration.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from rag.llm.chat_model import chat_completions
from rag.retrieval.rag_retrieval import retrieve_for_query
from rag.agent.prompts import generate_agent_prompt
from memory.services.messages import MessageService
from common.constants import MemoryType


logger = logging.getLogger(__name__)


class AgentService:
    """
    Core agent service that manages conversations with context memory,
    long-term memory, and knowledge base integration.
    """

    def __init__(self, user_id: str = "default", dept_tag: str = "default", kb_id: str = "default"):
        self.user_id = user_id
        self.dept_tag = dept_tag
        self.kb_id = kb_id
        self.message_service = MessageService()

    def chat(self,
             query: str,
             session_id: str,
             memory_id: str = "default",
             top_k: int = 5,
             use_rag: bool = True,
             include_long_term_memory: bool = True) -> Dict[str, Any]:
        """
        Main chat method that handles conversation with memory and RAG integration.

        Args:
            query: User's input query
            session_id: Unique identifier for the conversation session
            memory_id: Identifier for the memory context
            top_k: Number of results to retrieve from knowledge base
            use_rag: Whether to use RAG knowledge base integration
            include_long_term_memory: Whether to include long-term memory in context

        Returns:
            Dictionary containing the agent's response and metadata
        """
        print(f"[DEBUG AgentService.chat] Starting chat. query='{query[:50]}...' session_id={session_id}")
        # Retrieve recent conversation history for context memory
        context_messages = self._get_context_memory(session_id, memory_id)
        print(f"[DEBUG AgentService.chat] context_messages count={len(context_messages)}")

        # Retrieve relevant information from knowledge base if requested
        rag_results = []
        if use_rag:
            print(f"[DEBUG AgentService.chat] Calling retrieve_for_query...")
            rag_results = retrieve_for_query(
                query=query,
                user_id=self.user_id,
                dept_tag=self.dept_tag,
                kb_id=self.kb_id,
                top_k=top_k
            )
            print(f"[DEBUG AgentService.chat] retrieve_for_query done, hits={len(rag_results)}")

        # Retrieve relevant long-term memory if requested
        long_term_memory = []
        if include_long_term_memory:
            print(f"[DEBUG AgentService.chat] Getting long-term memory...")
            long_term_memory = self._get_relevant_long_term_memory(query, memory_id)
            print(f"[DEBUG AgentService.chat] long_term_memory count={len(long_term_memory)}")

        # Generate the agent prompt with context, RAG results, and long-term memory
        print(f"[DEBUG AgentService.chat] Generating agent prompt...")
        agent_prompt = generate_agent_prompt(
            query=query,
            context_messages=context_messages,
            rag_results=rag_results,
            long_term_memory=long_term_memory
        )
        print(f"[DEBUG AgentService.chat] Prompt generated, length={len(agent_prompt)}")

        # Get response from LLM
        print(f"[DEBUG AgentService.chat] Calling chat_completions...")
        response = chat_completions(
            messages=[{"role": "user", "content": agent_prompt}],
            temperature=0.7
        )
        print(f"[DEBUG AgentService.chat] chat_completions returned, response_len={len(response)}")

        # Store the conversation in memory
        self._store_conversation(query, response, session_id, memory_id)
        print(f"[DEBUG AgentService.chat] Conversation stored, returning response")

        return {
            "response": response,
            "session_id": session_id,
            "memory_id": memory_id,
            "rag_results": rag_results,
            "context_messages": context_messages,
            "long_term_memory": long_term_memory,
            "timestamp": datetime.now().isoformat()
        }

    def _get_context_memory(self, session_id: str, memory_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent messages from the current conversation session.
        This serves as short-term/context memory.
        """
        try:
            recent_messages = self.message_service.get_recent_messages(
                uid_list=[self.user_id],
                memory_ids=[memory_id],
                agent_id="ragpulse_agent",
                session_id=session_id,
                limit=limit
            )
            return recent_messages
        except Exception as e:
            logger.warning(f"Failed to retrieve context memory: {e}")
            return []

    def _get_relevant_long_term_memory(self, query: str, memory_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories from long-term storage based on the query.
        """
        try:
            # Search for relevant memories based on the query
            condition = {"agent_id": "ragpulse_agent", "memory_id": memory_id}
            # This is a simplified search - in a real implementation,
            # we would embed the query and do semantic search in memory
            relevant_memories = self.message_service.search_message(
                memory_ids=[memory_id],
                condition_dict=condition,
                uid_list=[self.user_id],
                match_expressions=[],  # Simplified search
                top_n=top_k
            )
            return relevant_memories
        except Exception as e:
            logger.warning(f"Failed to retrieve long-term memory: {e}")
            return []

    def _store_conversation(self, user_input: str, agent_response: str, session_id: str, memory_id: str):
        """
        Store the conversation in both context and long-term memory.
        """
        try:
            # Prepare messages for storage
            messages_to_store = [
                {
                    "message_id": f"{session_id}_user_{datetime.now().timestamp()}",
                    "message_type": MemoryType.RAW.name.lower(),
                    "source_id": "",
                    "memory_id": memory_id,
                    "user_id": self.user_id,
                    "agent_id": "ragpulse_agent",
                    "session_id": session_id,
                    "content": user_input,
                    "status": True
                },
                {
                    "message_id": f"{session_id}_agent_{datetime.now().timestamp()}",
                    "message_type": MemoryType.RAW.name.lower(),
                    "source_id": "",
                    "memory_id": memory_id,
                    "user_id": self.user_id,
                    "agent_id": "ragpulse_agent",
                    "session_id": session_id,
                    "content": agent_response,
                    "status": True
                }
            ]

            # Insert messages into memory
            self.message_service.insert_message(
                messages=messages_to_store,
                uid=self.user_id,
                memory_id=memory_id
            )
        except Exception as e:
            logger.error(f"Failed to store conversation in memory: {e}")

    def clear_session_memory(self, session_id: str, memory_id: str):
        """
        Clear memory for a specific session.
        """
        try:
            condition = {
                "session_id": session_id,
                "memory_id": memory_id,
                "agent_id": "ragpulse_agent"
            }
            self.message_service.delete_message(
                condition=condition,
                uid=self.user_id,
                memory_id=memory_id
            )
        except Exception as e:
            logger.error(f"Failed to clear session memory: {e}")

    def list_sessions(self, memory_id: str) -> List[str]:
        """
        List all conversation sessions in memory.
        """
        try:
            result = self.message_service.list_message(
                uid=self.user_id,
                memory_id=memory_id,
                page=1,
                page_size=100  # Assuming we don't need pagination for sessions
            )
            sessions = set()
            for msg in result.get("message_list", []):
                if msg.get("session_id"):
                    sessions.add(msg["session_id"])
            return list(sessions)
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []


# Convenience function to initialize agent service
def create_agent(user_id: str = "default", dept_tag: str = "default", kb_id: str = "default") -> AgentService:
    """
    Factory function to create an agent service instance.
    """
    return AgentService(user_id=user_id, dept_tag=dept_tag, kb_id=kb_id)
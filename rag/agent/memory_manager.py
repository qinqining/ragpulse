"""
Memory management for RAGPulse agent with context and long-term memory capabilities.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from memory.services.messages import MessageService
from common.constants import MemoryType


logger = logging.getLogger(__name__)


class AgentMemoryManager:
    """
    Manages memory for the RAGPulse agent, including context memory (short-term)
    and long-term memory.
    """

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.message_service = MessageService()

    def save_context_memory(self,
                           session_id: str,
                           memory_id: str,
                           message_data: Dict[str, Any],
                           message_type: str = "user") -> bool:
        """
        Save a message to context memory (short-term memory for current session).
        """
        try:
            message = {
                "message_id": f"{session_id}_{message_type}_{datetime.now().timestamp()}",
                "message_type": MemoryType.RAW.name.lower(),
                "source_id": "",
                "memory_id": memory_id,
                "user_id": self.user_id,
                "agent_id": "ragpulse_agent",
                "session_id": session_id,
                "content": message_data.get('content', ''),
                "status": True
            }

            # Add any additional metadata
            if 'metadata' in message_data:
                message.update(message_data['metadata'])

            self.message_service.insert_message(
                messages=[message],
                uid=self.user_id,
                memory_id=memory_id
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save context memory: {e}")
            return False

    def get_context_memory(self, session_id: str, memory_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve context memory (short-term memory for current session).
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

    def save_long_term_memory(self,
                             memory_id: str,
                             content: str,
                             importance_score: float = 0.5,
                             tags: Optional[List[str]] = None) -> bool:
        """
        Save important information to long-term memory with an importance score.
        """
        try:
            message = {
                "message_id": f"ltm_{memory_id}_{datetime.now().timestamp()}",
                "message_type": MemoryType.RAW.name.lower(),
                "source_id": "",
                "memory_id": f"long_term_{memory_id}",
                "user_id": self.user_id,
                "agent_id": "ragpulse_agent",
                "session_id": "long_term_storage",
                "content": content,
                "status": True,
                "importance_score": importance_score,
                "tags": tags or []
            }

            self.message_service.insert_message(
                messages=[message],
                uid=self.user_id,
                memory_id=f"long_term_{memory_id}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save long-term memory: {e}")
            return False

    def search_long_term_memory(self,
                               query: str,
                               memory_id: str,
                               top_k: int = 5,
                               min_importance: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search long-term memory for relevant information based on query.
        In a production system, this would involve semantic search.
        """
        try:
            # Create a simplified search condition based on query presence in content
            condition = {
                "agent_id": "ragpulse_agent",
                "memory_id": f"long_term_{memory_id}"
            }

            # For now, we'll use a basic search approach
            # In a real implementation, this would involve embedding the query
            # and performing semantic search in the long-term memory
            results = self.message_service.search_message(
                memory_ids=[f"long_term_{memory_id}"],
                condition_dict=condition,
                uid_list=[self.user_id],
                match_expressions=[],  # In a real implementation, this would be semantic search
                top_n=top_k
            )

            # Filter by minimum importance score if specified
            if min_importance > 0:
                results = [
                    result for result in results
                    if result.get('importance_score', 0.5) >= min_importance
                ]

            return results
        except Exception as e:
            logger.warning(f"Failed to search long-term memory: {e}")
            return []

    def get_memory_stats(self, memory_id: str) -> Dict[str, Any]:
        """
        Get statistics about memory usage.
        """
        try:
            sizes = self.message_service.calculate_memory_size(
                memory_ids=[memory_id, f"long_term_{memory_id}"],
                uid_list=[self.user_id]
            )

            context_messages = self.message_service.list_message(
                uid=self.user_id,
                memory_id=memory_id,
                page=1,
                page_size=1
            )

            long_term_messages = self.message_service.list_message(
                uid=self.user_id,
                memory_id=f"long_term_{memory_id}",
                page=1,
                page_size=1
            )

            return {
                "memory_id": memory_id,
                "context_memory_size": sizes.get(memory_id, 0),
                "long_term_memory_size": sizes.get(f"long_term_{memory_id}", 0),
                "context_message_count": context_messages.get("total_count", 0),
                "long_term_message_count": long_term_messages.get("total_count", 0),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"Failed to get memory stats: {e}")
            return {}

    def clear_context_memory(self, session_id: str, memory_id: str):
        """
        Clear context memory for a specific session.
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
            logger.error(f"Failed to clear context memory: {e}")

    def clear_long_term_memory(self, memory_id: str):
        """
        Clear long-term memory.
        """
        try:
            condition = {
                "memory_id": f"long_term_{memory_id}",
                "agent_id": "ragpulse_agent"
            }
            self.message_service.delete_message(
                condition=condition,
                uid=self.user_id,
                memory_id=f"long_term_{memory_id}"
            )
        except Exception as e:
            logger.error(f"Failed to clear long-term memory: {e}")


# Singleton instance for default use
default_memory_manager = AgentMemoryManager()
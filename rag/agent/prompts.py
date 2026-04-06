"""
Prompt templates for RAGPulse agent.
"""
from __future__ import annotations

from typing import Dict, List, Any


def generate_agent_prompt(
    query: str,
    context_messages: List[Dict[str, Any]],
    rag_results: List[Dict[str, Any]],
    long_term_memory: List[Dict[str, Any]]
) -> str:
    """
    Generate the complete prompt for the agent with context, RAG results, and long-term memory.

    Args:
        query: User's current query
        context_messages: Recent conversation history (context memory)
        rag_results: Results from knowledge base retrieval
        long_term_memory: Relevant memories from long-term storage

    Returns:
        Formatted prompt string for the LLM
    """
    # Build context from recent conversation
    context_section = ""
    if context_messages:
        context_section = "### Recent Conversation Context:\n"
        for msg in context_messages[-5:]:  # Use last 5 messages
            role = "User" if "user" in msg.get("content", "").lower() or \
                   msg.get("message_id", "").endswith("_user") else "Assistant"
            content = msg.get("content", "")
            context_section += f"- {role}: {content}\n"
        context_section += "\n"

    # Build RAG results section
    rag_section = ""
    if rag_results:
        rag_section = "### Retrieved Information from Knowledge Base:\n"
        for i, result in enumerate(rag_results[:3]):  # Limit to top 3 results
            doc_content = result.get("document", "")[:500]  # Truncate long documents
            metadata = result.get("metadata", {})
            rag_section += f"Result {i+1}: {doc_content}\n"
            if metadata:
                rag_section += f"Metadata: {metadata}\n"
        rag_section += "\n"

    # Build long-term memory section
    ltm_section = ""
    if long_term_memory:
        ltm_section = "### Relevant Past Memories:\n"
        for i, memory in enumerate(long_term_memory[:3]):  # Limit to top 3 memories
            content = memory.get("content", "")
            ltm_section += f"Memory {i+1}: {content}\n"
        ltm_section += "\n"

    # Construct the final prompt
    prompt = f"""You are an AI assistant powered by RAGPulse, equipped with context memory, long-term memory, and knowledge base integration. Use the following information to respond to the user's query:

{context_section}{rag_section}{ltm_section}### Current User Query:
{query}

### Instructions:
- If the query relates to previous conversation context, acknowledge and continue from there
- Incorporate relevant information from the knowledge base if it helps answer the query
- Reference relevant past memories if they are pertinent to the current query
- If the knowledge base or memories don't provide sufficient information, acknowledge this limitation
- Maintain a helpful and conversational tone
- If asked to generate content based on the knowledge base, create well-structured and informative content

Please provide a comprehensive and helpful response based on all available information."""

    return prompt


def generate_writing_assistant_prompt(
    topic: str,
    context: str = "",
    style_guide: str = "",
    additional_requirements: str = ""
) -> str:
    """
    Generate a prompt for the writing assistant functionality.

    Args:
        topic: The main topic for the writing task
        context: Additional context or background information
        style_guide: Style requirements (tone, audience, format, etc.)
        additional_requirements: Any other specific requirements

    Returns:
        Formatted prompt string for content generation
    """
    prompt = f"""You are an AI writing assistant powered by RAGPulse. Create content on the following topic:

### Topic:
{topic}

"""

    if context:
        prompt += f"### Additional Context:\n{context}\n\n"

    if style_guide:
        prompt += f"### Style Guide:\n{style_guide}\n\n"

    if additional_requirements:
        prompt += f"### Additional Requirements:\n{additional_requirements}\n\n"

    prompt += """### Instructions:
- Create well-structured, informative content based on the provided information
- If you have access to relevant information from the knowledge base, incorporate it appropriately
- Maintain a professional and engaging tone
- Organize the content logically with appropriate headings/subheadings if needed
- Ensure the content is original and not merely a copy of source materials
- If citing specific information from the knowledge base, indicate this appropriately

Begin writing the content now:"""

    return prompt
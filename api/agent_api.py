"""
API endpoints for RAGPulse agent functionality.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rag.agent.memory_manager import AgentMemoryManager
from rag.agent.service import AgentService, create_agent

router = APIRouter()
_log = logging.getLogger(__name__)


class AgentChatRequest(BaseModel):
    query: str
    session_id: str
    memory_id: str = "default"
    top_k: int = 5
    use_rag: bool = True
    include_long_term_memory: bool = True
    user_id: str = "default"
    dept_tag: str = "default"
    kb_id: str = "default"


class AgentChatResponse(BaseModel):
    response: str
    session_id: str
    memory_id: str
    rag_results: List[Dict[str, Any]]
    context_messages: List[Dict[str, Any]]
    long_term_memory: List[Dict[str, Any]]
    timestamp: str


class MemoryClearRequest(BaseModel):
    memory_id: str
    session_id: Optional[str] = None
    user_id: str = "default"


class SessionListResponse(BaseModel):
    sessions: List[str]
    memory_id: str


@router.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest) -> AgentChatResponse:
    """
    Main chat endpoint for the RAGPulse agent.
    Supports context memory, long-term memory, and knowledge base integration.
    """
    print(f"[DEBUG] agent_chat() called with query='{request.query[:50]}...' session_id={request.session_id}")
    q = (request.query or "").strip()
    print(f"[DEBUG] agent/chat 收到请求 query_len={len(q)} session_id={request.session_id} user_id={request.user_id} dept_tag={request.dept_tag} kb_id={request.kb_id}")
    _log.info(
        "agent/chat 收到请求 query_len=%s query_preview=%r session_id=%s memory_id=%s user_id=%s "
        "dept_tag=%s kb_id=%s top_k=%s use_rag=%s include_ltm=%s",
        len(q),
        q[:300] + ("…" if len(q) > 300 else ""),
        request.session_id,
        request.memory_id,
        request.user_id,
        request.dept_tag,
        request.kb_id,
        request.top_k,
        request.use_rag,
        request.include_long_term_memory,
    )
    try:
        print(f"[DEBUG] Creating agent for user_id={request.user_id} dept_tag={request.dept_tag} kb_id={request.kb_id}")
        agent = create_agent(
            user_id=request.user_id,
            dept_tag=request.dept_tag,
            kb_id=request.kb_id
        )
        print(f"[DEBUG] Agent created, calling agent.chat()")

        result = agent.chat(
            query=request.query,
            session_id=request.session_id,
            memory_id=request.memory_id,
            top_k=request.top_k,
            use_rag=request.use_rag,
            include_long_term_memory=request.include_long_term_memory
        )
        print(f"[DEBUG] agent.chat() returned, response_len={len(result.get('response', '')) if isinstance(result, dict) else 0}")

        resp_text = (result.get("response") or "") if isinstance(result, dict) else ""
        _log.info(
            "agent/chat 完成 session_id=%s response_len=%s rag_hits=%s",
            request.session_id,
            len(resp_text),
            len(result.get("rag_results") or []) if isinstance(result, dict) else 0,
        )
        print(f"[DEBUG] Returning AgentChatResponse to client")
        return AgentChatResponse(**result)
    except Exception as e:
        print(f"[ERROR] agent_chat exception: {e}")
        import traceback
        traceback.print_exc()
        _log.exception(
            "agent/chat 失败 session_id=%s user_id=%s: %s",
            request.session_id,
            request.user_id,
            e,
        )
        raise HTTPException(status_code=500, detail=f"Agent chat error: {str(e)}")


@router.get("/agent/memory/list")
async def list_memory(memory_id: str = "default", user_id: str = "default") -> Dict[str, Any]:
    """
    List stored memories for a specific memory ID.

    使用显式 query 参数：GET 请求不能有 JSON body，若把 Pydantic 模型当「体」解析会 422。
    """
    try:
        memory_manager = AgentMemoryManager(user_id=user_id)
        stats = memory_manager.get_memory_stats(memory_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory list error: {str(e)}")


@router.post("/agent/memory/clear")
async def clear_memory(request: MemoryClearRequest) -> Dict[str, str]:
    """
    Clear specific memory contexts.
    If session_id is provided, clears context memory for that session.
    Otherwise, clears all long-term memory for the memory_id.
    """
    try:
        memory_manager = AgentMemoryManager(user_id=request.user_id)

        if request.session_id:
            # Clear context memory for specific session
            memory_manager.clear_context_memory(request.session_id, request.memory_id)
            return {"status": f"Context memory cleared for session {request.session_id}"}
        else:
            # Clear long-term memory
            memory_manager.clear_long_term_memory(request.memory_id)
            return {"status": f"Long-term memory cleared for {request.memory_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory clear error: {str(e)}")


@router.get("/agent/sessions/list", response_model=SessionListResponse)
async def list_sessions(memory_id: str = "default", user_id: str = "default") -> SessionListResponse:
    """
    List all conversation sessions in memory.

    使用显式 query 参数：GET 不能用 body；原先 SessionListRequest 易被当成 body，导致 422。
    """
    try:
        agent = create_agent(user_id=user_id)
        sessions = agent.list_sessions(memory_id)
        return SessionListResponse(sessions=sessions, memory_id=memory_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session list error: {str(e)}")
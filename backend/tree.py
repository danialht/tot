"""
Tree-of-Thoughts (ToT) implementation with Cerebras integration.
Based on arXiv:2305.10601 with fail-fast behavior and extensible design.
"""

import os
import re
import json
import logging
import asyncio
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Callable, AsyncIterator
from enum import Enum

import aiohttp
import requests

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, use environment variables directly
    pass

# Import prompt templates
try:
    from .prompts import PROMPTS, build_context_string
except ImportError:
    # Handle direct execution
    from prompts import PROMPTS, build_context_string


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Content logger for detailed thought content, prompts, and responses
content_logger = logging.getLogger('tot.content')
content_handler = logging.StreamHandler()
content_handler.setFormatter(logging.Formatter(
    '%(asctime)s - CONTENT - %(message)s',
    datefmt='%H:%M:%S'
))
content_logger.addHandler(content_handler)
content_logger.setLevel(logging.INFO)
content_logger.propagate = False  # Don't duplicate in main logger


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Thought:
    """A single thought/idea in the reasoning process."""
    text: str  # proposed action/claim
    rationale: str  # brief why-it-helps explanation
    confidence: float  # self-estimate in [0, 1]
    intent: str  # "propose" | "decompose" | "check" | "compute" | "decide"
    candidate: Optional[str] = None  # concrete suggested answer for this subproblem
    metadata: Dict[str, Any] = field(default_factory=dict)  # tool outputs, raw blocks, etc.

    def __post_init__(self):
        """Validate thought data on creation."""
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Thought text must be a non-empty string")
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ValueError("Thought rationale must be a non-empty string")
        if not (0 <= self.confidence <= 1):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        # Restrict to the new, simplified intent set
        valid_intents = ["idea", "answer"]
        if self.intent not in valid_intents:
            raise ValueError(f"Invalid intent: {self.intent}")


@dataclass
class NodeState:
    """Current state of a node in the search tree."""
    subproblem_text: str  # what this node is solving
    scratchpad: List[Thought] = field(default_factory=list)  # accepted chain so far
    objective: str = "default"  # "default" | "optimize:max" | "optimize:min"
    candidate_solution: Optional[str] = None  # best current candidate for this subproblem
    derived_facts: Dict[str, Any] = field(default_factory=dict)  # deterministic results

    def __post_init__(self):
        """Validate node state on creation."""
        if not isinstance(self.subproblem_text, str) or not self.subproblem_text.strip():
            raise ValueError("Subproblem text must be a non-empty string")
        valid_objectives = ["default", "optimize:max", "optimize:min"]
        if self.objective not in valid_objectives:
            raise ValueError(f"Objective must be one of {valid_objectives}, got {self.objective}")


@dataclass
class SubProblemNode:
    """A node in the Tree-of-Thoughts search tree."""
    state: NodeState
    depth: int
    incoming_thought: Optional[Thought] = None
    children: List['SubProblemNode'] = field(default_factory=list)
    value_estimate: Optional[float] = None
    terminal_status: Optional[str] = None  # "solved" | "dead_end"
    
    # Exposed attributes for external access
    input_context: Dict[str, Any] = field(default_factory=dict)  # What was passed into this node
    generated_thoughts: List[Thought] = field(default_factory=list)  # Ideas generated at this node
    thought_scores: Dict[str, float] = field(default_factory=dict)  # Scores for each thought
    processing_status: str = "pending"  # "pending" | "processing" | "completed" | "failed"
    
    def __post_init__(self):
        """Validate node on creation."""
        if self.depth < 0:
            raise ValueError(f"Depth must be non-negative, got {self.depth}")
        if self.terminal_status is not None and self.terminal_status not in ["solved", "dead_end"]:
            raise ValueError(f"Invalid terminal status: {self.terminal_status}")
    
    def get_chain_of_thought(self) -> List[Thought]:
        """Get the current chain of thought (scratchpad) for this node."""
        return self.state.scratchpad.copy()
    
    def get_children_with_thoughts(self) -> List[Dict[str, Any]]:
        """Get children along with the thoughts that led to them."""
        return [
            {
                "child": child,
                "incoming_thought": child.incoming_thought,
                "thought_score": child.thought_scores.get(child.incoming_thought.text if child.incoming_thought else "", 0.0)
            }
            for child in self.children
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary for external access."""
        # Short branch idea/title extraction helpers
        def _extract_branch_idea_from_subproblem(text: str) -> Optional[str]:
            try:
                m = re.search(r"Assume/Idea:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
                if m:
                    return m.group(1).strip()
            except Exception:
                pass
            return None

        incoming_idea_text = self.incoming_thought.text if self.incoming_thought else None
        branch_idea: Optional[str] = incoming_idea_text or _extract_branch_idea_from_subproblem(self.state.subproblem_text)

        # Build a concise branch title for UI consumption
        def _truncate(s: Optional[str], n: int = 60) -> Optional[str]:
            if not s:
                return None
            s = s.strip()
            return s if len(s) <= n else s[:n - 1] + "…"

        branch_title: Optional[str] = None
        if self.depth == 0:
            branch_title = _truncate(self.state.subproblem_text, 60)
        else:
            branch_title = _truncate(branch_idea or incoming_idea_text or self.state.subproblem_text, 60)

        # Parse packaged solution/reasoning if available
        solution_package = self.state.candidate_solution
        solution_text: Optional[str] = None
        reasoning_text: Optional[str] = None
        try:
            if solution_package:
                sol_m = re.search(r"^\s*Solution:\s*(.+?)\s*(?:\n|$)", solution_package, re.IGNORECASE | re.MULTILINE)
                if sol_m:
                    solution_text = sol_m.group(1).strip()
                rsn_m = re.search(r"\n\s*Reasoning:\s*(.+)\Z", solution_package, re.IGNORECASE | re.DOTALL)
                if rsn_m:
                    reasoning_text = rsn_m.group(1).strip()
        except Exception:
            pass

        # Structured chain of thought
        chain_of_thought_struct = [
            {
                "text": t.text,
                "rationale": t.rationale,
                "intent": t.intent,
                "candidate": t.candidate
            }
            for t in self.state.scratchpad
        ]

        return {
            "id": f"node_{id(self)}",
            "depth": self.depth,
            "subproblem": self.state.subproblem_text,
            "chain_of_thought_text": "\n".join(t.text for t in self.state.scratchpad),
            "generated_thoughts": [
                {
                    "text": t.text,
                    "rationale": t.rationale,
                    "score": self.thought_scores.get(t.text, 0.0)
                }
                for t in self.generated_thoughts
            ],
            "incoming_thought_text": incoming_idea_text,
            "children_count": len(self.children),
            "processing_status": self.processing_status,
            "terminal_status": self.terminal_status,
            # New additive fields for richer UI (non-breaking)
            "solution_package": solution_package,
            "solution_text": solution_text,
            "reasoning_text": reasoning_text,
            "branch_idea": branch_idea,
            "branch_title": branch_title,
            "idea_text": incoming_idea_text,
            "rationale": (self.incoming_thought.rationale if self.incoming_thought else None),
            "chain_of_thought": chain_of_thought_struct,
            "children": [child.to_dict() for child in self.children]
        }


# ============================================================================
# Abstract Interfaces
# ============================================================================

class ThoughtGenerator(ABC):
    """Abstract base class for generating thoughts."""
    
    @abstractmethod
    async def generate(self, state: NodeState, max_ideas_hint: Optional[int] = None, mode: Optional[str] = None, num_branches: Optional[int] = None) -> List[Thought]:
        """Generate thoughts for the given node state.

        mode: "idea" for idea-only branches, "solve" for leaf solving. Required.
        num_branches: required when mode == "idea"; number of idea branches to produce.
        """
        pass
    
    async def generate_streaming(self, state: NodeState, max_ideas_hint: Optional[int] = None, mode: Optional[str] = None, num_branches: Optional[int] = None) -> AsyncIterator[Thought]:
        """Generate thoughts with streaming support. Default implementation calls generate()."""
        thoughts = await self.generate(state, max_ideas_hint, mode=mode, num_branches=num_branches)
        for thought in thoughts:
            yield thought


class IdeaValidator(ABC):
    """Abstract base class for validating thoughts."""
    
    @abstractmethod
    async def evaluate(self, state: NodeState, thought: Thought) -> float:
        """Evaluate a thought, returning a score in [0, 1]."""
        pass


class SolutionValidator(ABC):
    """Abstract base class for validating solutions."""
    
    @abstractmethod
    async def is_solved(self, state: NodeState) -> bool:
        """Check if the node state represents a solved problem."""
        pass
    
    @abstractmethod
    async def quality(self, state: NodeState) -> Optional[float]:
        """Return quality score (higher is better) for optimization tasks."""
        pass


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SolverConfig:
    """Configuration for the Tree-of-Thoughts solver."""
    beam_width: int = 3
    max_depth: int = 2
    selection_top_k: int = 2
    max_ideas_hint_per_node: Optional[int] = None
    temperature: float = 0.7
    max_retries: int = 2
    cerebras_model: str = "gpt-oss-120b"
    reasoning_effort: str = "low"
    max_tokens: int = 60000
    candidate_selection_mode: str = "validation"  # "validation" | "synthesis"

    def __post_init__(self):
        """Validate configuration values."""
        if self.beam_width <= 0:
            raise ValueError(f"beam_width must be positive, got {self.beam_width}")
        if self.max_depth <= 0:
            raise ValueError(f"max_depth must be positive, got {self.max_depth}")
        if self.selection_top_k <= 0:
            raise ValueError(f"selection_top_k must be positive, got {self.selection_top_k}")
        if not (0 <= self.temperature <= 2):
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {self.max_retries}")
        if self.max_ideas_hint_per_node is not None and self.max_ideas_hint_per_node <= 0:
            raise ValueError(f"max_ideas_hint_per_node must be positive or None, got {self.max_ideas_hint_per_node}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.candidate_selection_mode not in {"validation", "synthesis"}:
            raise ValueError(f"candidate_selection_mode must be 'validation' or 'synthesis', got {self.candidate_selection_mode}")


# ============================================================================
# Utilities
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for deduplication."""
    return re.sub(r'\s+', ' ', text.strip().lower())


def deduplicate_thoughts(thoughts: List[Thought]) -> List[Thought]:
    """Remove duplicate thoughts based on normalized text."""
    seen = set()
    unique_thoughts = []
    
    for thought in thoughts:
        normalized = normalize_text(thought.text)
        if normalized not in seen and normalized:
            seen.add(normalized)
            unique_thoughts.append(thought)
    
    return unique_thoughts


# ----------------------------------------------------------------------------
# HTTP helpers with retry/backoff and concurrency limiting
# ----------------------------------------------------------------------------

# Global semaphore to limit concurrent Cerebras API requests
_DEFAULT_MAX_CONCURRENCY =  int(os.getenv("CEREBRAS_MAX_CONCURRENCY", "3"))
_CEREBRAS_SEMAPHORE = asyncio.Semaphore(_DEFAULT_MAX_CONCURRENCY)


async def _parse_retry_after_seconds(retry_after_header: Optional[str]) -> Optional[float]:
    """Parse Retry-After header to seconds, if possible."""
    if not retry_after_header:
        return None
    try:
        return float(retry_after_header.strip())
    except Exception:
        return None


async def http_post_json_with_retries(
    session: aiohttp.ClientSession,
    url: str,
    headers: Dict[str, str],
    data: Dict[str, Any],
    *,
    timeout_total: int = 30,
    max_attempts: int = 5,
    backoff_base: float = 0.5,
    backoff_factor: float = 2.0,
    backoff_max: float = 20.0,
    retry_statuses: tuple = (429, 500, 502, 503, 504),
    log_context: str = ""
) -> Dict[str, Any]:
    """POST JSON with retries, exponential backoff, and a concurrency guard.

    Retries on network errors, timeouts, and retryable HTTP statuses like 429/5xx.
    Respects Retry-After header when present (seconds).
    """
    async with _CEREBRAS_SEMAPHORE:
        last_error: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                async with session.post(
                    url,
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=timeout_total)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    # Non-200
                    error_text = await response.text()
                    status = response.status
                    retry_after_hdr = response.headers.get("Retry-After")
                    logger.warning(f"    ❌ {log_context}HTTP {status} - {error_text[:500]}")
                    if status in retry_statuses and attempt < max_attempts:
                        parsed_retry_after = await _parse_retry_after_seconds(retry_after_hdr)
                        base_sleep = min(backoff_max, backoff_base * (backoff_factor ** (attempt - 1)))
                        jitter = random.uniform(0, base_sleep * 0.25)
                        sleep_s = parsed_retry_after if parsed_retry_after is not None else base_sleep + jitter
                        logger.info(f"    ⏳ {log_context}Retrying in {sleep_s:.2f}s (attempt {attempt}/{max_attempts})")
                        await asyncio.sleep(max(0.05, sleep_s))
                        continue
                    # Non-retryable or exhausted
                    raise RuntimeError(f"HTTP {status} - {error_text}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt < max_attempts:
                    base_sleep = min(backoff_max, backoff_base * (backoff_factor ** (attempt - 1)))
                    jitter = random.uniform(0, base_sleep * 0.25)
                    sleep_s = base_sleep + jitter
                    logger.info(f"    ⏳ {log_context}Transient error: {e}. Retrying in {sleep_s:.2f}s (attempt {attempt}/{max_attempts})")
                    await asyncio.sleep(max(0.05, sleep_s))
                    continue
                raise
        # If we exit loop without returning, raise last error if set
        if last_error:
            raise last_error
        raise RuntimeError("Request failed after retries without specific error")


# ============================================================================
# Concrete Implementations
# ============================================================================

class ProposeThoughtGenerator(ThoughtGenerator):
    """Cerebras-backed thought generator with async and streaming support."""
    
    def __init__(self, config: SolverConfig):
        self.config = config
        self.api_key = os.getenv("CEREBRAS_API_KEY")
        if not self.api_key:
            raise ValueError("CEREBRAS_API_KEY environment variable is required")
        
        self.base_url = "https://api.cerebras.ai/v1"
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def generate(self, state: NodeState, max_ideas_hint: Optional[int] = None, mode: Optional[str] = None, num_branches: Optional[int] = None) -> List[Thought]:
        """Generate thoughts using Cerebras API."""
        if mode not in {"idea", "solve"}:
            raise ValueError(f"ProposeThoughtGenerator.generate requires mode in {'idea','solve'}, got {mode}")
        if mode == "idea":
            # Enforce explicit num_branches for idea mode
            if not isinstance(num_branches, int) or num_branches <= 0:
                raise ValueError(f"num_branches must be positive int for idea mode, got {num_branches}")
        
        logger.info(f"🤖 Calling Cerebras API for thought generation ({mode})")
        logger.info(f"📝 Subproblem: {state.subproblem_text}")
        logger.info(f"📊 Current scratchpad has {len(state.scratchpad)} thoughts")
        
        prompt = self._build_prompt(state, mode=mode, num_branches=num_branches)
        logger.debug(f"📤 Prompt length: {len(prompt)} chars")
        
        for attempt in range(self.config.max_retries + 1):
            try:
                logger.info(f"🔄 API attempt {attempt + 1}/{self.config.max_retries + 1}")
                response_text = await self._call_cerebras_async(prompt)
                logger.info(f"📥 Received response ({len(response_text)} chars)")
                logger.debug(f"📥 Raw response: {response_text[:200]}...")
                
                thoughts = self._parse_thoughts(response_text)
                logger.info(f"🧠 Parsed {len(thoughts)} thoughts from response")
                
                if not thoughts and attempt < self.config.max_retries:
                    logger.warning(f"No thoughts parsed on attempt {attempt + 1}, retrying...")
                    prompt += PROMPTS["retry_prompt"]
                    continue
                
                deduplicated = deduplicate_thoughts(thoughts)
                if len(deduplicated) < len(thoughts):
                    logger.info(f"🔄 Deduplicated: {len(thoughts)} -> {len(deduplicated)} thoughts")
                
                # Enforce mode-specific constraints
                if mode == "idea":
                    # Drop any items that include a direct answer candidate
                    deduplicated = [t for t in deduplicated if not t.candidate]
                else:
                    # solve mode: keep only items that include a candidate answer
                    deduplicated = [t for t in deduplicated if t.candidate]
                
                return deduplicated
                
            except Exception as e:
                if attempt == self.config.max_retries:
                    logger.error(f"Failed to generate thoughts after {self.config.max_retries + 1} attempts: {e}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
        
        return []
    
    async def generate_streaming(self, state: NodeState, max_ideas_hint: Optional[int] = None, mode: Optional[str] = None, num_branches: Optional[int] = None) -> AsyncIterator[Thought]:
        """Generate thoughts with streaming support."""
        if mode not in {"idea", "solve"}:
            raise ValueError(f"ProposeThoughtGenerator.generate_streaming requires mode in {'idea','solve'}, got {mode}")
        if mode == "idea":
            if not isinstance(num_branches, int) or num_branches <= 0:
                raise ValueError(f"num_branches must be positive int for idea mode, got {num_branches}")
        
        logger.info(f"🤖 Starting streaming thought generation ({mode})")
        logger.info(f"📝 Subproblem: {state.subproblem_text}")
        
        prompt = self._build_prompt(state, mode=mode, num_branches=num_branches)
        
        async for thought in self._call_cerebras_streaming(prompt):
            # Enforce mode-specific filtering on the fly
            if mode == "idea" and thought.candidate:
                continue
            if mode == "solve" and not thought.candidate:
                continue
            yield thought
    
    def _build_prompt(self, state: NodeState, mode: str, num_branches: Optional[int]) -> str:
        """Build the prompt for thought generation using configured templates."""
        # Build context string
        context = build_context_string(state.scratchpad, state.candidate_solution)
        
        if mode == "idea":
            template = PROMPTS["idea_generation_fixed"]
            prompt = template.format(
                problem=state.subproblem_text,
                context=context,
                num_branches=int(num_branches)
            )
        else:
            template = PROMPTS["leaf_solver"]
            prompt = template.format(
                problem=state.subproblem_text,
                context=context
            )
        
        # Log the full prompt being sent
        logger.info(f"    🤖 THOUGHT GENERATION:")
        logger.info(f"    {'-'*50}")
        for line in prompt.split('\n'):
            logger.info(f"    {line}")
        logger.info(f"    {'-'*50}")
        
        return prompt
    
    async def _call_cerebras_async(self, prompt: str) -> str:
        """Make async API call to Cerebras."""
        session = await self._get_session()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.config.cerebras_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "reasoning_effort": self.config.reasoning_effort
        }
        # Log request (without API key)
        try:
            logger.info("    🌐 POST /chat/completions (generation)")
            logger.info(f"    Request params: model={data['model']}, temperature={data['temperature']}, max_tokens={data['max_tokens']}, stream=False, reasoning_effort={self.config.reasoning_effort}")
        except Exception:
            pass
        
        result = await http_post_json_with_retries(
            session,
            f"{self.base_url}/chat/completions",
            headers,
            data,
            timeout_total=30,
            max_attempts=max(1, int(self.config.max_retries) + 1),
            log_context="generation: "
        )
        # Log raw JSON (truncated)
        try:
            raw_str = json.dumps(result)[:2000]
            logger.info("    🧾 RAW RESPONSE JSON (truncated):")
            for line in raw_str.split('\n'):
                logger.info(f"    {line}")
        except Exception:
            pass
        if "choices" not in result or not result["choices"]:
            raise RuntimeError("No choices in Cerebras response")
        
        response_content = result["choices"][0]["message"]["content"]
        
        # Log the full response
        logger.info(f"    📥 THOUGHTS:")
        logger.info(f"    {'-'*50}")
        for line in response_content.split('\n'):
            logger.info(f"    {line}")
        logger.info(f"    {'-'*50}")
        
        return response_content
    
    async def _call_cerebras_streaming(self, prompt: str) -> AsyncIterator[Thought]:
        """Make streaming API call to Cerebras."""
        session = await self._get_session()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.config.cerebras_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
            "reasoning_effort": self.config.reasoning_effort
        }
        try:
            logger.info("    🌐 POST /chat/completions (generation streaming)")
            logger.info(f"    Request params: model={data['model']}, temperature={data['temperature']}, max_tokens={data['max_tokens']}, stream=True, reasoning_effort={self.config.reasoning_effort}")
        except Exception:
            pass
        
        # Implement streaming with initial retry loop to obtain a stream, handling 429/5xx
        max_attempts = max(1, int(self.config.max_retries) + 1)
        attempt = 1
        while attempt <= max_attempts:
            try:
                async with _CEREBRAS_SEMAPHORE:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            status = response.status
                            logger.warning(f"    ❌ Cerebras non-200(stream): {status} - {error_text[:500]}")
                            if status in (429, 500, 502, 503, 504) and attempt < max_attempts:
                                retry_after_hdr = response.headers.get("Retry-After")
                                parsed_retry_after = await _parse_retry_after_seconds(retry_after_hdr)
                                base_sleep = min(20.0, 0.5 * (2.0 ** (attempt - 1)))
                                jitter = random.uniform(0, base_sleep * 0.25)
                                sleep_s = parsed_retry_after if parsed_retry_after is not None else base_sleep + jitter
                                logger.info(f"    ⏳ streaming: Retrying in {sleep_s:.2f}s (attempt {attempt}/{max_attempts})")
                                await asyncio.sleep(max(0.05, sleep_s))
                                attempt += 1
                                continue
                            raise RuntimeError(f"Cerebras API error: {status} - {error_text}")
                        
                        collected_content = ""
                        async for line in response.content:
                            line_str = line.decode('utf-8').strip()
                            if not line_str or line_str == "data: [DONE]":
                                continue
                            
                            if line_str.startswith("data: "):
                                try:
                                    json_str = line_str[6:]  # Remove "data: " prefix
                                    chunk = json.loads(json_str)
                                    
                                    if "choices" in chunk and chunk["choices"]:
                                        delta = chunk["choices"][0].get("delta", {})
                                        # Append both content and reasoning tokens if present
                                        if "content" in delta:
                                            collected_content += delta["content"]
                                        if "reasoning" in delta:
                                            collected_content += delta["reasoning"]
                                except json.JSONDecodeError:
                                    continue
                        # After stream completion, log once and parse once
                        if collected_content.strip():
                            logger.info(f"    📥 THOUGHTS (stream final):")
                            logger.info(f"    {'-'*50}")
                            for line in collected_content.split('\n'):
                                logger.info(f"    {line}")
                            logger.info(f"    {'-'*50}")
                            thoughts = self._parse_thoughts(collected_content)
                            for thought in thoughts:
                                yield thought
                        else:
                            # Fallback: try non-streaming call if nothing was collected
                            try:
                                response_text = await self._call_cerebras_async(prompt)
                                thoughts = self._parse_thoughts(response_text)
                                for thought in thoughts:
                                    yield thought
                            except Exception:
                                return
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < max_attempts:
                    base_sleep = min(20.0, 0.5 * (2.0 ** (attempt - 1)))
                    jitter = random.uniform(0, base_sleep * 0.25)
                    sleep_s = base_sleep + jitter
                    logger.info(f"    ⏳ streaming: Transient error {e}; retrying in {sleep_s:.2f}s (attempt {attempt}/{max_attempts})")
                    await asyncio.sleep(max(0.05, sleep_s))
                    attempt += 1
                    continue
                raise
    
    def _parse_thoughts(self, text: str) -> List[Thought]:
        """Parse thoughts from Cerebras response."""
        # Check for empty response keywords
        text_lower = text.lower()
        empty_keywords = ["no strong ideas", "no critical insights needed"]
        for keyword in empty_keywords:
            if keyword in text_lower:
                return []
        
        thoughts = []
        
        # Prefer splitting by explicit branch headers like: "1. Branch: ..."
        branch_header_re = re.compile(r'^\s*(?:\*\*)?\d+\.\s*Branch\s*:\s*.*$', re.IGNORECASE | re.MULTILINE)
        header_matches = list(branch_header_re.finditer(text))
        segments: List[str] = []
        if header_matches:
            for idx, match in enumerate(header_matches):
                start = match.start()
                end = header_matches[idx + 1].start() if idx + 1 < len(header_matches) else len(text)
                segment = text[start:end].strip()
                if segment:
                    segments.append(segment)
        else:
            # Fallback: split on horizontal rule separators (---)
            hr_splitter = re.compile(r'^\s*-{3,}\s*$', re.MULTILINE)
            parts = [s for s in hr_splitter.split(text) if s.strip()] or [text]
            segments = [p.strip() for p in parts]
        
        for segment in segments:
            thought = self._parse_single_thought(segment)
            if thought:
                thoughts.append(thought)
        
        return thoughts
    
    def _parse_single_thought(self, item: str) -> Optional[Thought]:
        """Parse a single thought from text."""
        try:
            metadata: Dict[str, Any] = {"raw_item": item}

            # Primary: New Branch/Why/Answer format
            branch_match = re.search(r'Branch:\s*(.+?)(?=\n|Why:|\n|$)', item, re.IGNORECASE | re.DOTALL)
            # Support "Why:" or variants like "Why this matters –"
            why_match = re.search(r'(?:Why\s*:\s*|\*?Why\s+this\s+matters\*?\s*[–-]\s*)(.+?)(?=\n\s*Answer|\n\s*\*?Insights\*?|\n\s*-{3,}|\n\s*$)', item, re.IGNORECASE | re.DOTALL)
            # Capture Insights bullet block after an Insights header
            insights_block_match = re.search(r'\n\s*\*?Insights\*?\s*:?\s*\n(?P<ins>(?:[ \t]*[\-•–].*(?:\n|$))+)', item, re.IGNORECASE)
            # Capture Answer blocks as multi-line up to a separator (---) or end of item
            tentative_answer_match = re.search(r'Answer\s*\(tentative\)\s*:\s*(.+?)(?=\n\s*-{3,}\s*$|\Z)', item, re.IGNORECASE | re.DOTALL | re.MULTILINE)
            answer_match = re.search(r'Answer\s*:\s*(.+?)(?=\n\s*-{3,}\s*$|\Z)', item, re.IGNORECASE | re.DOTALL | re.MULTILINE)

            # Legacy support: Idea/Insight/Risk
            legacy_idea_match = re.search(r'(?:Idea|Insight|Risk):\s*(.+?)(?=\n|Why:|$)', item, re.IGNORECASE | re.DOTALL)
            # Ignore any legacy Confidence fields; we no longer use confidence
            intent_field_match = re.search(r'Intent:\s*([\w\-]+)', item, re.IGNORECASE)

            text: Optional[str] = None
            rationale = why_match.group(1).strip() if why_match else "No rationale provided"
            candidate: Optional[str] = None
            # We no longer use confidence; set to 0.0 placeholder
            confidence = 0.0

            # Determine text and candidate from new format first
            if branch_match:
                text = branch_match.group(1).strip()
                if tentative_answer_match:
                    candidate = tentative_answer_match.group(1).strip()
                    metadata["tentative_answer"] = True
                elif answer_match:
                    candidate = answer_match.group(1).strip()
            elif legacy_idea_match:
                text = legacy_idea_match.group(1).strip()
                # Legacy: try to find Answer: as candidate
                if answer_match:
                    candidate = answer_match.group(1).strip()
                metadata["legacy_format"] = True
            else:
                return None

            # Clean markdown artifacts from extracted fields
            def _strip_md_edges(s: str) -> str:
                s = s.strip()
                s = re.sub(r'^\*+\s*', '', s)
                s = re.sub(r'\s*\*+$', '', s)
                return s.strip()

            text = _strip_md_edges(text)
            if candidate:
                candidate = _strip_md_edges(candidate)
            if rationale and rationale != "No rationale provided":
                rationale = _strip_md_edges(rationale)

            # Extract and normalize insights bullets
            insights_list: Optional[List[str]] = None
            if insights_block_match:
                raw_block = insights_block_match.group('ins')
                lines = [ln.strip() for ln in raw_block.splitlines()]
                bullets = []
                for ln in lines:
                    m = re.match(r'^[\-•–]\s*(.*)$', ln)
                    if m:
                        bullets.append(m.group(1).strip())
                if bullets:
                    insights_list = bullets
                    # Combine into rationale if Why was missing or to enrich it
                    insights_text = "\n".join([f"- {b}" for b in bullets])
                    if rationale == "No rationale provided":
                        rationale = f"Insights:\n{insights_text}"
                    else:
                        rationale = f"{rationale}\nInsights:\n{insights_text}"
                    metadata["insights"] = bullets

            # Map or infer intent to the new set {idea, answer}
            intent_raw = intent_field_match.group(1).strip().lower() if intent_field_match else None
            intent_mapped = None
            if intent_raw in {"answer", "solve"}:
                intent_mapped = "answer"
            elif intent_raw in {"idea", "analyze", "analysis", "explore", "propose", "check", "compute", "decide", "clarify", "verify", "optimize", "reframe", "experiment", "mitigate", "investigate", "conclude"}:
                intent_mapped = "idea"

            if intent_mapped is None:
                # Infer from presence of a definitive Answer (not tentative)
                if answer_match and not tentative_answer_match:
                    intent_mapped = "answer"
                else:
                    intent_mapped = "idea"

            # Clamp confidence (noop with 0.0 but keep for safety)
            confidence = max(0.0, min(1.0, confidence))

            return Thought(
                text=text,
                rationale=rationale,
                confidence=confidence,
                intent=intent_mapped,
                candidate=candidate,
                metadata=metadata
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse thought item: {e}")
            return None


class RuleBasedIdeaValidator(IdeaValidator):
    """Rule-based validator for thoughts."""
    
    async def evaluate(self, state: NodeState, thought: Thought) -> float:
        """Evaluate thought using heuristic scoring."""
        # No longer use model-provided confidence; start from neutral baseline
        score = 0.5
        
        # Check for duplication against scratchpad
        normalized_new = normalize_text(thought.text)
        for existing in state.scratchpad:
            if normalize_text(existing.text) == normalized_new:
                return 0.0  # Duplicate
        
        # Reject trivial/empty thoughts
        if len(thought.text.strip()) < 10:
            return 0.0
        
        # Boost based on simplified intents
        intent_boosts = {
            "idea": 0.05,
            "answer": 0.15
        }
        score += intent_boosts.get(thought.intent, 0.0)
        
        # Boost thoughts with concrete candidates
        if thought.candidate:
            score += 0.1
        
        # Length bonus for detailed thoughts
        if len(thought.text) > 50:
            score += 0.05
        
        return min(1.0, score)


class DefaultSolutionValidator(SolutionValidator):
    """Default validator for solution checking."""
    
    async def is_solved(self, state: NodeState) -> bool:
        """Check if state has a candidate solution."""
        return state.candidate_solution is not None and state.candidate_solution.strip() != ""
    
    async def quality(self, state: NodeState) -> Optional[float]:
        """Attempt numeric cast for optimization tasks."""
        if not state.candidate_solution:
            return None
        
        try:
            # Try to extract numeric value
            numeric_match = re.search(r'[-+]?[0-9]*\.?[0-9]+', state.candidate_solution)
            if numeric_match:
                return float(numeric_match.group())
        except (ValueError, AttributeError):
            pass
        
        return None


class LLMIdeaValidator(IdeaValidator):
    """LLM-backed validator for thoughts using Cerebras."""
    
    def __init__(self, config: SolverConfig):
        self.config = config
        self.api_key = os.getenv("CEREBRAS_API_KEY")
        if not self.api_key:
            raise ValueError("CEREBRAS_API_KEY environment variable is required")
        
        self.base_url = "https://api.cerebras.ai/v1"
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def evaluate(self, state: NodeState, thought: Thought) -> float:
        """Evaluate thought using LLM scoring."""
        try:
            # Build validation prompt
            context = build_context_string(state.scratchpad, state.candidate_solution)
            
            # Map simplified internal intents to validation prompt intents
            validation_intent = "solve" if thought.intent == "answer" else "analyze"

            prompt = PROMPTS["thought_validation"].format(
                problem=state.subproblem_text,
                context=context,
                thought_text=thought.text,
                thought_rationale=thought.rationale,
                thought_intent=validation_intent
            )
            
            # Log the validation prompt
            logger.info(f"    🤖 THOUGHT VALIDATION:")
            logger.info(f"    {'-'*50}")
            for line in prompt.split('\n'):
                logger.info(f"    {line}")
            logger.info(f"    {'-'*50}")
            
            # Call Cerebras
            response_text = await self._call_cerebras_async(prompt)
            
            # Log the validation response
            logger.info(f"    📥 VALIDATION:")
            logger.info(f"    {'-'*50}")
            for line in response_text.split('\n'):
                logger.info(f"    {line}")
            logger.info(f"    {'-'*50}")
            
            # Parse score
            score = self._parse_validation_score(response_text)
            
            return score
            
        except Exception as e:
            logger.warning(f"LLM validation failed: {e}, falling back to rule-based score")
            # Fallback to rule-based validator instead of confidence
            try:
                rb = RuleBasedIdeaValidator()
                return await rb.evaluate(state, thought)
            except Exception as inner:
                logger.warning(f"Rule-based fallback also failed: {inner}; returning neutral 0.5")
                return 0.5
    
    async def _call_cerebras_async(self, prompt: str) -> str:
        """Make async API call to Cerebras for validation."""
        session = await self._get_session()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.config.cerebras_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "reasoning_effort": self.config.reasoning_effort
        }
        # Log request (without API key)
        try:
            logger.info("    🌐 POST /chat/completions (idea validation)")
            logger.info(f"    Request params: model={data['model']}, temperature={data['temperature']}, max_tokens={data['max_tokens']}, stream=False, reasoning_effort={self.config.reasoning_effort}")
        except Exception:
            pass
        
        result = await http_post_json_with_retries(
            session,
            f"{self.base_url}/chat/completions",
            headers,
            data,
            timeout_total=30,
            max_attempts=max(1, int(self.config.max_retries) + 1),
            log_context="idea validation: "
        )
        # Log raw JSON (truncated)
        try:
            raw_str = json.dumps(result)[:2000]
            logger.info("    🧾 RAW RESPONSE JSON (truncated):")
            for line in raw_str.split('\n'):
                logger.info(f"    {line}")
        except Exception:
            pass
        if "choices" not in result or not result["choices"]:
            raise RuntimeError("No choices in Cerebras response")
        
        choice0 = result["choices"][0]
        # Robustly extract content across possible payload shapes
        message = choice0.get("message") or {}
        content = message.get("content")
        if not content:
            content = choice0.get("text")
        if not content:
            # Some providers may put partials under delta even in non-stream situations
            delta = choice0.get("delta") or {}
            content = delta.get("content")
        if not content:
            raise KeyError("content")
        return content
    
    def _parse_validation_score(self, text: str) -> float:
        """Parse validation score from response."""
        # Look for "Score: X.X" pattern
        score_match = re.search(r'Score:\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
        if score_match:
            score = float(score_match.group(1))
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        
        # Fallback: if no score present, return neutral
        return 0.5
        


class LLMSolutionValidator(SolutionValidator):
    """LLM-backed solution validator using Cerebras."""
    
    def __init__(self, config: SolverConfig):
        self.config = config
        self.api_key = os.getenv("CEREBRAS_API_KEY")
        if not self.api_key:
            raise ValueError("CEREBRAS_API_KEY environment variable is required")
        
        self.base_url = "https://api.cerebras.ai/v1"
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def is_solved(self, state: NodeState) -> bool:
        """Check if solution is correct using LLM validation."""
        if not state.candidate_solution:
            return False
        
        try:
            prompt = PROMPTS["solution_validation"].format(
                problem=state.subproblem_text,
                solution=state.candidate_solution
            )
            
            # Log the solution validation prompt
            logger.info(f"    🤖 SOLUTION VALIDATION:")
            logger.info(f"    {'-'*50}")
            for line in prompt.split('\n'):
                logger.info(f"    {line}")
            logger.info(f"    {'-'*50}")
            
            response_text = await self._call_cerebras_async(prompt)
            
            # Log the solution validation response
            logger.info(f"    📥 SOLUTION CHECK:")
            logger.info(f"    {'-'*50}")
            for line in response_text.split('\n'):
                logger.info(f"    {line}")
            logger.info(f"    {'-'*50}")
            
            # Parse correctness
            is_correct = self._parse_correctness(response_text)
            
            return is_correct
            
        except Exception as e:
            logger.warning(f"LLM solution validation failed: {e}, falling back to simple check")
            return state.candidate_solution is not None and state.candidate_solution.strip() != ""
    
    async def quality(self, state: NodeState) -> Optional[float]:
        """Return quality score from LLM confidence."""
        if not state.candidate_solution:
            return None
        
        # For now, use simple numeric extraction as fallback
        try:
            numeric_match = re.search(r'[-+]?[0-9]*\.?[0-9]+', state.candidate_solution)
            if numeric_match:
                return float(numeric_match.group())
        except (ValueError, AttributeError):
            pass
        
        return None
    
    async def _call_cerebras_async(self, prompt: str) -> str:
        """Make async API call to Cerebras for solution validation."""
        session = await self._get_session()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.config.cerebras_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "reasoning_effort": self.config.reasoning_effort
        }
        # Log request (without API key)
        try:
            logger.info("    🌐 POST /chat/completions (solution validation)")
            logger.info(f"    Request params: model={data['model']}, temperature={data['temperature']}, max_tokens={data['max_tokens']}, stream=False, reasoning_effort={self.config.reasoning_effort}")
        except Exception:
            pass
        
        result = await http_post_json_with_retries(
            session,
            f"{self.base_url}/chat/completions",
            headers,
            data,
            timeout_total=30,
            max_attempts=max(1, int(self.config.max_retries) + 1),
            log_context="solution validation: "
        )
        # Log raw JSON (truncated)
        try:
            raw_str = json.dumps(result)[:2000]
            logger.info("    🧾 RAW RESPONSE JSON (truncated):")
            for line in raw_str.split('\n'):
                logger.info(f"    {line}")
        except Exception:
            pass
        if "choices" not in result or not result["choices"]:
            raise RuntimeError("No choices in Cerebras response")
        
        choice0 = result["choices"][0]
        message = choice0.get("message") or {}
        content = message.get("content")
        if not content:
            content = choice0.get("text")
        if not content:
            delta = choice0.get("delta") or {}
            content = delta.get("content")
        if not content:
            raise KeyError("content")
        return content
    
    def _parse_correctness(self, text: str) -> bool:
        """Parse correctness from validation response."""
        text_lower = text.lower()
        
        # Look for explicit "Correct: yes/no" pattern
        correct_match = re.search(r'correct:\s*(yes|no)', text_lower)
        if correct_match:
            return correct_match.group(1) == "yes"
        
        # Look for other indicators
        positive_indicators = ["correct", "yes", "accurate", "right", "valid"]
        negative_indicators = ["incorrect", "no", "wrong", "invalid", "false"]
        
        positive_count = sum(1 for word in positive_indicators if word in text_lower)
        negative_count = sum(1 for word in negative_indicators if word in text_lower)
        
        if positive_count > negative_count:
            return True
        elif negative_count > positive_count:
            return False
        
        # Default to false for ambiguous responses
        logger.warning(f"Ambiguous correctness response: {text[:100]}...")
        return False


# ============================================================================
# Main Solver
# ============================================================================

@dataclass
class SearchResult:
    """Result of a tree search."""
    expanded_nodes: int
    solved: bool
    trace: List[str] = field(default_factory=list)
    final_solution_text: Optional[str] = None
    root_candidates: List[Dict[str, Any]] = field(default_factory=list)


class TreeOfThoughtSolver:
    """Main Tree-of-Thoughts solver using beam search with async support and public tree access."""
    
    def __init__(
        self,
        config: SolverConfig,
        thought_generator: ThoughtGenerator,
        idea_validator: IdeaValidator,
        solution_validator: SolutionValidator
    ):
        self.config = config
        self.thought_generator = thought_generator
        self.idea_validator = idea_validator
        self.solution_validator = solution_validator
        # Synthesis prompt support
        self._synth_api_key = os.getenv("CEREBRAS_API_KEY")
        self._synth_base_url = "https://api.cerebras.ai/v1"
        # Leaf post-hoc override removed; selection is per-node based on mode
        
        # Public tree state attributes
        self.root_node: Optional[SubProblemNode] = None
        self.current_frontier: List[SubProblemNode] = []
        self.expanded_nodes: int = 0
        self.current_depth: int = 0
        self.is_solving: bool = False
        self.solving_complete: bool = False
        self.trace: List[str] = []
        self.final_solution_text: Optional[str] = None
        self.root_candidates: List[Dict[str, Any]] = []
        
    def get_tree_state(self) -> Dict[str, Any]:
        """Get current tree state for external access."""
        return {
            "root_node": self.root_node.to_dict() if self.root_node else None,
            "current_frontier": [node.to_dict() for node in self.current_frontier],
            "expanded_nodes": self.expanded_nodes,
            "current_depth": self.current_depth,
            "is_solving": self.is_solving,
            "solving_complete": self.solving_complete,
            "trace": self.trace.copy(),
            "frontier_size": len(self.current_frontier),
            "final_solution_text": self.final_solution_text,
            "root_candidates": self.root_candidates
        }
    
    def get_solution_path(self) -> List[Dict[str, Any]]:
        """Get the path from root to best solution."""
        if not self.root_node:
            return []
        return [{
            "node": self.root_node.to_dict(),
            "chain_of_thought": [
                {
                    "text": t.text,
                    "rationale": t.rationale,
                    "confidence": t.confidence,
                    "intent": t.intent,
                    "candidate": t.candidate
                } for t in self.root_node.get_chain_of_thought()
            ]
        }]
    
    async def solve(self, problem_text: str, objective: str = "default", log: bool = True) -> SearchResult:
        """Solve the problem using Tree-of-Thoughts."""
        if not log:
            logging.disable(logging.CRITICAL)
        
        self.is_solving = True
        self.solving_complete = False
        self.expanded_nodes = 0
        self.current_depth = 0
        self.trace = []
        
        logger.info(f"🌳 Starting Tree-of-Thoughts (per-node recursion)")
        logger.info(f"📋 Problem: {problem_text}")
        logger.info(f"🎯 Objective: {objective}")
        
        # Initialize root node
        root_state = NodeState(
            subproblem_text=problem_text,
            objective=objective
        )
        self.root_node = SubProblemNode(state=root_state, depth=0)
        self.root_node.input_context = {
            "original_problem": problem_text,
            "objective": objective,
            "config": self.config.__dict__
        }
        logger.info(f"🌱 Created root node")
        
        # Reset frontier-style public attrs for backward-compat public access
        self.current_frontier = [self.root_node]
        self.final_solution_text = None
        self.root_candidates = []
        
        # Hard cap depth at 5 as requested
        depth_cap = self.config.max_depth
        
        # Compute best solution text recursively from root context
        result = await self._compute_best_recursive(self.root_node, depth_cap)
        
        # Package final answer at root level
        chosen_idea_text: Optional[str] = None
        if self.root_node.children:
            # If best came from a specific child, we cannot directly know which one here without storing.
            # Instead, set idea to None unless a direct child was chosen and recorded on root.
            pass
        
        # For root, we package using the chosen first-level idea if available; otherwise just solution
        final_solution_text = result.get("solution_text")
        final_reasoning_text = result.get("reasoning_text")
        self.final_solution_text = final_solution_text
        self.root_candidates = result.get("candidates") or []
        if final_solution_text:
            # If root has an incoming_thought (it should not), include; otherwise omit
            if self.root_node.incoming_thought and self.root_node.incoming_thought.text:
                package = self._package_idea_solution_reasoning(self.root_node.incoming_thought.text, final_solution_text, final_reasoning_text)
            else:
                # Root: no incoming idea; return solution only
                package = self._package_idea_solution_reasoning(None, final_solution_text, final_reasoning_text)
            self.root_node.state.candidate_solution = package
            self.root_node.terminal_status = "solved" if result.get("solved") else None
            # Expose packaged final answer with reasoning
            self.final_solution_text = package
        
        # No post-hoc leaf override; selection handled per-node based on mode
        else:
            self.root_node.state.candidate_solution = None
            self.root_node.terminal_status = None
        
        # Expose public attributes
        self.current_frontier = []
        self.current_depth = 0
        
        self.is_solving = False
        self.solving_complete = True
        
        return SearchResult(
            expanded_nodes=self.expanded_nodes,
            solved=bool(result.get("solved")),
            trace=self.trace,
            final_solution_text=self.final_solution_text,
            root_candidates=self.root_candidates
        )

    def _package_idea_and_solution(self, idea_text: Optional[str], solution_text: str) -> str:
        """Create the 'inputted idea + solution' package string."""
        if idea_text is None or not str(idea_text).strip():
            return f"Solution: {solution_text}"
        return f"Idea: {idea_text}\nSolution: {solution_text}"

    def _package_idea_solution_reasoning(self, idea_text: Optional[str], solution_text: str, reasoning_text: Optional[str]) -> str:
        """Create a package string including idea, solution, and reasoning."""
        base = self._package_idea_and_solution(idea_text, solution_text)
        if reasoning_text and str(reasoning_text).strip():
            return f"{base}\nReasoning: {reasoning_text}"
        return base

    async def _evaluate_in_parent_context(self, parent_problem_text: str, solution_text: str, reasoning_text: Optional[str]) -> Dict[str, Any]:
        """Evaluate solution+reasoning in the given problem context using the solution validator."""
        combined = solution_text if not (reasoning_text and reasoning_text.strip()) else f"{solution_text}\n\nReasoning: {reasoning_text}"
        temp_state = NodeState(subproblem_text=parent_problem_text, candidate_solution=combined)
        try:
            solved = await self.solution_validator.is_solved(temp_state)
        except Exception as e:
            logger.warning(f"Solution validation error: {e}")
            solved = False
        quality: Optional[float] = None
        try:
            quality = await self.solution_validator.quality(temp_state)
        except Exception as e:
            logger.warning(f"Solution quality error: {e}")
        return {"solved": solved, "quality": quality}

    async def _compute_best_recursive(self, node: SubProblemNode, depth_cap: int) -> Dict[str, Any]:
        """Compute the best solution text at this node, returning dict with keys: solution_text, solved, quality."""
        indent = "  " * node.depth
        logger.info(f"\n{indent}📍 Node at Depth {node.depth}")
        logger.info(f"{indent}   Problem: {node.state.subproblem_text}")
        node.processing_status = "processing"
        
        # Decide mode and expected branching per fixed topology
        if node.depth < 2:
            mode = "idea"
            
            required_branches = 3 if node.depth == 0 else 2
            thoughts = await self.thought_generator.generate(
                node.state,
                None,
                mode=mode,
                num_branches=required_branches
            )
            # Failfast: ensure exactly required number of unique ideas
            if len(thoughts) != required_branches:
                logger.warning(f"{indent}Expected exactly {required_branches} idea branches, got {len(thoughts)}")
                # Retry once with the same parameters
                retry = await self.thought_generator.generate(
                    node.state,
                    None,
                    mode=mode,
                    num_branches=required_branches
                )
                thoughts = retry
            if len(thoughts) != required_branches:
                raise ValueError(f"Fixed-topology violation at depth {node.depth}: needed {required_branches} ideas, got {len(thoughts)}")
        else:
            mode = "solve"
            required_branches = 1  # single solve attempt set
            thoughts = await self.thought_generator.generate(
                node.state,
                None,
                mode=mode,
                num_branches=None
            )
            if len(thoughts) < 1:
                logger.warning(f"{indent}Leaf produced no solutions; retrying once")
                retry = await self.thought_generator.generate(
                    node.state,
                    None,
                    mode=mode,
                    num_branches=None
                )
                thoughts = retry
        node.generated_thoughts = thoughts.copy()
        logger.info(f"{indent}💭 Generated {len(thoughts)} thoughts")
        
        # Score thoughts (no external validation; neutral scores)
        scored_thoughts: List[tuple] = []
        for t in thoughts:
            node.thought_scores[t.text] = 0.0
            scored_thoughts.append((t, 0.0))
        
        # Separate by mode
        if mode == "solve":
            direct_candidates: List[Thought] = [t for t, _ in scored_thoughts if t.candidate]
            branch_ideas_scored = []
        else:
            direct_candidates = []
            branch_ideas_scored: List[tuple] = [(t, s) for t, s in scored_thoughts]
        
        # Enforce exact branching counts at internal nodes
        total_ideas = len(branch_ideas_scored)
        if mode == "idea":
            branch_ideas_scored.sort(key=lambda x: x[1], reverse=True)
            if total_ideas < required_branches:
                logger.warning(f"{indent}Fewer ideas than required ({total_ideas} < {required_branches}); proceeding with available")
            branch_ideas_scored = branch_ideas_scored[: required_branches]
            logger.info(f"{indent}✅ Selected {len(branch_ideas_scored)} ideas for branching (required {required_branches})")
        
        # Recursively compute child solutions for each idea
        child_results: List[Dict[str, Any]] = []
        for idea, idea_score in branch_ideas_scored:
            # Create child node for this idea
            new_scratchpad = node.state.scratchpad + [idea]
            child_state = NodeState(
                subproblem_text=f"{node.state.subproblem_text}\n\nAssume/Idea: {idea.text}\nWhy: {idea.rationale}",
                scratchpad=new_scratchpad,
                objective=node.state.objective,
                candidate_solution=None,
                derived_facts=node.state.derived_facts.copy()
            )
            child_node = SubProblemNode(
                state=child_state,
                depth=node.depth + 1,
                incoming_thought=idea
            )
            node.children.append(child_node)
            self.expanded_nodes += 1
            logger.info(f"{indent}🌿 Spawning child for idea: {idea.text[:60]}...")
            child_best = await self._compute_best_recursive(child_node, depth_cap)
            # Store association with parent idea score for tie-breaks if needed
            child_results.append({
                "idea": idea,
                "idea_score": idea_score,
                "solution_text": child_best.get("solution_text"),
                "reasoning_text": child_best.get("reasoning_text"),
                "solved": child_best.get("solved", False),
                "quality": child_best.get("quality")
            })
        
        # Build candidate packages for evaluation in parent context (this node is the parent)
        parent_problem = node.state.subproblem_text
        candidates: List[Dict[str, Any]] = []
        
        # Helper to build reasoning text from a Thought (use Why + Insights bullets if any)
        def build_reasoning(thought: Thought) -> str:
            parts: List[str] = []
            if thought.rationale and thought.rationale.strip() and thought.rationale.strip().lower() != "no rationale provided":
                parts.append(thought.rationale.strip())
            insights = thought.metadata.get("insights") if isinstance(thought.metadata, dict) else None
            if insights and isinstance(insights, list):
                bullets = [f"- {str(x).strip()}" for x in insights if str(x).strip()]
                if bullets:
                    parts.append("Insights:\n" + "\n".join(bullets))
            return "\n".join(parts).strip()

        # Direct solutions proposed at this node (no validation)
        for t in direct_candidates:
            sol_text = t.candidate or ""
            reasoning_text = build_reasoning(t)
            # If this is a leaf node (mode == 'solve' and no further branching), collect globally
            if mode == "solve":
                try:
                    self._leaf_candidates.append({
                        "origin": "leaf",
                        "solution_text": sol_text,
                        "reasoning_text": reasoning_text,
                        "path_idea": node.incoming_thought.text if node.incoming_thought else None
                    })
                except Exception:
                    pass
            candidates.append({
                "origin": "direct",
                "idea_text": None,
                "solution_text": sol_text,
                "reasoning_text": reasoning_text,
                "solved": None,
                "quality": 0.0,
                "tie_score": node.thought_scores.get(t.text, 0.0)
            })
        
        # Child-derived solutions, repackaged under this node's idea when returning upward
        for child_info in child_results:
            sol_text = child_info.get("solution_text")
            if not sol_text:
                continue
            reasoning_text = child_info.get("reasoning_text")
            candidates.append({
                "origin": "child",
                "idea_text": child_info["idea"].text,
                "solution_text": sol_text,
                "reasoning_text": reasoning_text,
                "solved": None,
                "quality": 0.0,
                "tie_score": float(child_info.get("idea_score") or 0.0)
            })
        
        best_candidate: Optional[Dict[str, Any]] = None
        if candidates:
            mode_selection = self.config.candidate_selection_mode
            if mode_selection == "synthesis":
                logger.info(f"{indent}🧪 Synthesizing across {len(candidates)} candidate(s)")
                synthesis = await self._synthesize_candidates(parent_problem, candidates)
                if synthesis and synthesis.get("solution_text"):
                    best_candidate = {
                        "origin": "synthesized",
                        "idea_text": None,
                        "solution_text": synthesis.get("solution_text"),
                        "reasoning_text": synthesis.get("reasoning_text"),
                        "solved": True,
                        "quality": synthesis.get("quality") or 0.0,
                        "tie_score": 0.0
                    }
                    logger.info(f"{indent}🏆 Synthesized candidate selected")
                else:
                    logger.warning(f"{indent}Synthesis failed; falling back to first candidate")
                    best_candidate = candidates[0]
            else:
                # validation mode: evaluate candidates relative to parent problem
                logger.info(f"{indent}🧪 Validating {len(candidates)} candidate(s)")
                validated: List[Dict[str, Any]] = []
                for c in candidates:
                    sol_text = c.get("solution_text") or ""
                    rsn_text = c.get("reasoning_text") or ""
                    # Use signed confidence score for ranking
                    signed_score = await self._score_solution_via_validation(parent_problem, sol_text, rsn_text)
                    cc = dict(c)
                    cc["signed_score"] = signed_score
                    validated.append(cc)
                def rank_key(c: Dict[str, Any]):
                    return (c.get("signed_score") or 0.0, c.get("tie_score") or 0.0)
                validated.sort(key=rank_key, reverse=True)
                best_candidate = validated[0]
                logger.info(f"{indent}🏆 Selected candidate (signed_score={best_candidate.get('signed_score')})")
        else:
            logger.info(f"{indent}❌ No candidates produced at this node")
        
        # Prepare return to parent: package with this node's incoming idea
        if best_candidate:
            selected_solution_text = best_candidate["solution_text"]
            selected_reasoning_text = best_candidate.get("reasoning_text")
            # Update node state with local packaging for visibility
            incoming_idea_text = node.incoming_thought.text if node.incoming_thought else None
            node.state.candidate_solution = self._package_idea_solution_reasoning(incoming_idea_text, selected_solution_text, selected_reasoning_text)
            node.terminal_status = "solved" if best_candidate["solved"] else None
            return {
                "solution_text": selected_solution_text,
                "reasoning_text": selected_reasoning_text,
                "solved": best_candidate["solved"],
                "quality": best_candidate.get("quality"),
                "candidates": candidates
            }
        else:
            node.state.candidate_solution = None
            node.terminal_status = None
            return {"solution_text": None, "reasoning_text": None, "solved": False, "quality": None, "candidates": candidates}

    async def _synthesize_candidates(self, problem_text: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Call Cerebras to synthesize the best answer from multiple candidates with reasoning."""
        try:
            # Format candidates as labeled blocks A, B, C, ...
            labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            blocks: List[str] = []
            for idx, c in enumerate(candidates):
                label = labels[idx % len(labels)]
                sol = str(c.get("solution_text") or "").strip()
                rsn = str(c.get("reasoning_text") or "").strip()
                block = f"Solution {label}: {sol}\nReasoning {label}: {rsn}\n---"
                blocks.append(block)
            candidates_block = "\n".join(blocks)
            prompt = PROMPTS["solution_synthesis"].format(problem=problem_text, candidates=candidates_block)

            # Log synthesis prompt
            logger.info(f"    🤖 SOLUTION SYNTHESIS:")
            logger.info(f"    {'-'*50}")
            for line in prompt.split('\n'):
                logger.info(f"    {line}")
            logger.info(f"    {'-'*50}")

            # Reuse ProposeThoughtGenerator HTTP logic with retries
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self._synth_api_key}", "Content-Type": "application/json"}
                data = {
                    "model": self.config.cerebras_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "reasoning_effort": self.config.reasoning_effort
                }
                logger.info("    🌐 POST /chat/completions (synthesis)")
                logger.info(f"    Request params: model={data['model']}, temperature={data['temperature']}, max_tokens={data['max_tokens']}, stream=False, reasoning_effort={self.config.reasoning_effort}")
                result = await http_post_json_with_retries(
                    session,
                    f"{self._synth_base_url}/chat/completions",
                    headers,
                    data,
                    timeout_total=30,
                    max_attempts=max(1, int(self.config.max_retries) + 1),
                    log_context="synthesis: "
                )
                content = result.get("choices", [{}])[0].get("message", {}).get("content")
                if not content:
                    return None

            # Log synthesis output
            logger.info(f"    📥 SYNTHESIS OUTPUT:")
            logger.info(f"    {'-'*50}")
            for line in content.split('\n'):
                logger.info(f"    {line}")
            logger.info(f"    {'-'*50}")

            # Parse using existing thought parser to extract Why/Answer
            parsed = ProposeThoughtGenerator(self.config)._parse_thoughts(content)
            # Expect a single branch; pick first
            if not parsed:
                return None
            t = parsed[0]
            solution_text = (t.candidate or "").strip()
            reasoning_text = (t.rationale or "").strip()
            if not solution_text:
                return None
            return {"solution_text": solution_text, "reasoning_text": reasoning_text, "quality": 0.0}
        except Exception as e:
            logger.warning(f"Synthesis error: {e}")
            return None

    async def _score_solution_via_validation(self, problem_text: str, solution_text: str, reasoning_text: Optional[str]) -> float:
        """Score a solution using the solution_validation prompt; parse Confidence as score."""
        try:
            combined = solution_text if not (reasoning_text and reasoning_text.strip()) else f"{solution_text}\n\nReasoning: {reasoning_text}"
            prompt = PROMPTS["solution_validation"].format(problem=problem_text, solution=combined)
            # Log the full validation prompt
            logger.info("    🤖 LEAF SOLUTION VALIDATION:")
            logger.info(f"    {'-'*50}")
            for line in prompt.split('\n'):
                logger.info(f"    {line}")
            logger.info(f"    {'-'*50}")
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self._synth_api_key}", "Content-Type": "application/json"}
                data = {
                    "model": self.config.cerebras_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "reasoning_effort": self.config.reasoning_effort
                }
                result = await http_post_json_with_retries(
                    session,
                    f"{self._synth_base_url}/chat/completions",
                    headers,
                    data,
                    timeout_total=30,
                    max_attempts=max(1, int(self.config.max_retries) + 1),
                    log_context="leaf validation: "
                )
                content = result.get("choices", [{}])[0].get("message", {}).get("content") or ""
            # Log the validation output
            logger.info("    📥 VALIDATION OUTPUT:")
            logger.info(f"    {'-'*50}")
            for line in content.split('\n'):
                logger.info(f"    {line}")
            logger.info(f"    {'-'*50}")
            # Parse Correct and Confidence
            correct_match = re.search(r"Correct:\s*(yes|no)", content, re.IGNORECASE)
            conf_match = re.search(r"Confidence:\s*([0-9]*\.?[0-9]+)", content, re.IGNORECASE)
            conf_val: Optional[float] = None
            if conf_match:
                try:
                    conf_val = float(conf_match.group(1))
                    conf_val = max(0.0, min(1.0, conf_val))
                except Exception:
                    conf_val = None
            if correct_match:
                is_yes = correct_match.group(1).strip().lower() == "yes"
                if is_yes:
                    return conf_val if conf_val is not None else 0.6
                else:
                    # Negative confidence if incorrect
                    return -(conf_val if conf_val is not None else 0.6)
            # No explicit Correct field: use confidence if present, else 0.0
            return conf_val if conf_val is not None else 0.0
        except Exception:
            return 0.0
    
    async def _expand_node(self, node: SubProblemNode) -> List[SubProblemNode]:
        """Expand a node by generating and validating thoughts."""
        # Node logging
        indent = "  " * node.depth
        logger.info(f"\n{indent}📍 Node at Depth {node.depth}")
        logger.info(f"{indent}   Problem: {node.state.subproblem_text}")
        if node.state.scratchpad:
            logger.info(f"{indent}   Previous thoughts: {len(node.state.scratchpad)}")
        
        # Store input context for this node
        node.input_context = {
            "subproblem": node.state.subproblem_text,
            "scratchpad_length": len(node.state.scratchpad),
            "depth": node.depth,
            "max_ideas_hint": self.config.max_ideas_hint_per_node
        }
        
        # Generate thoughts
        thoughts = await self.thought_generator.generate(
            node.state, 
            self.config.max_ideas_hint_per_node
        )
        
        # Store generated thoughts in the node
        node.generated_thoughts = thoughts.copy()
        
        logger.info(f"💭 Generated {len(thoughts)} thoughts")
        
        # Log detailed thoughts
        if thoughts:
            content_logger.info("💭 PARSED THOUGHTS")
            content_logger.info("=" * 60)
            for i, thought in enumerate(thoughts, 1):
                content_logger.info(f"Thought {i}:")
                content_logger.info(f"  Text: {thought.text}")
                content_logger.info(f"  Rationale: {thought.rationale}")
                content_logger.info(f"  Intent: {thought.intent}")
                if thought.candidate:
                    content_logger.info(f"  Candidate Answer: {thought.candidate}")
                if thought.metadata:
                    content_logger.info(f"  Metadata: {thought.metadata}")
                content_logger.info("")
            content_logger.info("=" * 60)
        
        # Brief summary for main logger
        for i, thought in enumerate(thoughts, 1):
            logger.info(f"  {i}. {thought.text[:50]}... (intent: {thought.intent})")
            if thought.candidate:
                logger.info(f"     → Candidate answer: {thought.candidate}")
        
        if not thoughts:
            logger.info(f"❌ No thoughts generated, marking as dead end")
            node.terminal_status = "dead_end"
            return []
        
        # Score thoughts
        logger.info(f"📊 Scoring {len(thoughts)} thoughts...")
        scored_thoughts = []
        for i, thought in enumerate(thoughts):
            # If the thought proposes a direct answer, skip idea validation
            if thought.candidate:
                scored_thoughts.append((thought, 0.0))
                node.thought_scores[thought.text] = 0.0
                logger.info(f"  {i+1}. Direct answer provided; skipping idea validation")
                continue
            try:
                score = await self.idea_validator.evaluate(node.state, thought)
                if not (0 <= score <= 1):
                    raise ValueError(f"Validator returned invalid score: {score}")
                scored_thoughts.append((thought, score))
                # Store score in node for external access
                node.thought_scores[thought.text] = score
                logger.info(f"  {i+1}. Score: {score:.3f} - {thought.text[:40]}...")
            except Exception as e:
                logger.warning(f"Failed to validate thought: {e}")
                continue
        
        # Select top thoughts (ideas only) for branching
        idea_only = [(t, s) for (t, s) in scored_thoughts if not t.candidate]
        idea_only.sort(key=lambda x: x[1], reverse=True)
        top_thoughts = idea_only[:self.config.selection_top_k]
        logger.info(f"✅ Selected top {len(top_thoughts)} ideas for branching (from {len(idea_only)} idea thoughts)")
        
        # Create children
        children = []
        for thought, score in top_thoughts:
            # Create new state
            new_scratchpad = node.state.scratchpad + [thought]
            new_candidate = thought.candidate if thought.candidate else node.state.candidate_solution
            
            # Compose child subproblem text as parent text plus full branch (ideas + insights)
            branch_parts = []
            for item in new_scratchpad:
                text_part = f"- {item.text}" if item.text else "-"
                rationale_part = f"  Insights: {item.rationale}" if item.rationale else ""
                branch_parts.append(f"{text_part}\n{rationale_part}".rstrip())
            branch_str = "\n".join(branch_parts)
            child_subproblem_text = f"{node.state.subproblem_text}\n\nBranch:\n{branch_str}" if branch_str else node.state.subproblem_text

            child_state = NodeState(
                subproblem_text=child_subproblem_text,
                scratchpad=new_scratchpad,
                objective=node.state.objective,
                candidate_solution=new_candidate,
                derived_facts=node.state.derived_facts.copy()
            )
            
            child_node = SubProblemNode(
                state=child_state,
                depth=node.depth + 1,
                incoming_thought=thought,
                value_estimate=score
            )
            
            children.append(child_node)
            node.children.append(child_node)
        
        return children


# ============================================================================
# Factory Functions
# ============================================================================

def create_default_solver(config: Optional[SolverConfig] = None, use_llm_validators: bool = False) -> TreeOfThoughtSolver:
    """Create a solver with default implementations."""
    if config is None:
        # Try to load from JSON config if available
        config_path = os.getenv("TOT_CONFIG_PATH") or os.path.join(os.path.dirname(__file__), "config.json")
        loaded: Optional[Dict[str, Any]] = None
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    loaded = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            loaded = None

        if loaded and isinstance(loaded, dict):
            # Only pass recognized fields; ignore unknowns to be fail-fast-safe
            allowed_keys = {
                "beam_width", "max_depth", "selection_top_k", "max_ideas_hint_per_node",
                "temperature", "max_retries", "cerebras_model", "reasoning_effort", "max_tokens"
            }
            filtered = {k: v for k, v in loaded.items() if k in allowed_keys}
            try:
                config = SolverConfig(**filtered)
            except Exception as e:
                logger.warning(f"Invalid values in config file, falling back to defaults: {e}. Loaded keys: {list(filtered.keys())}")
                config = SolverConfig()
        else:
            config = SolverConfig()
    
    thought_generator = ProposeThoughtGenerator(config)
    
    if use_llm_validators:
        idea_validator = LLMIdeaValidator(config)
        solution_validator = LLMSolutionValidator(config)
    else:
        idea_validator = RuleBasedIdeaValidator()
        solution_validator = DefaultSolutionValidator()
    
    return TreeOfThoughtSolver(
        config=config,
        thought_generator=thought_generator,
        idea_validator=idea_validator,
        solution_validator=solution_validator
    )


# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """Example usage of the async TreeOfThoughtSolver."""
    import logging
    
    # Set up clean logging - show only our tree structure and API calls
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',  # Clean format, just the message
        force=True
    )
    
    # Turn off other noisy loggers but keep our main one
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    
    logger.info("🌳 Async Tree-of-Thoughts with Full Logging")
    logger.info("="*60)
    
    # Test with the water molecule problem
    PROBLEM1 = "Add (normal) quotes to the make the following true in three different ways: \n\n Sam is flying to Europe is a sentence and is an expression."
    PROBLEM2 = 'We fill a glass with water up to the brim. we turn it upsidedown. give an estimate for how many water molecules are in the glass.'
    PROBLEM3 = "You have access to a 2sided dice, 3, 5... up to 41 (only the prime numbered ones). What is a strategy that strictly guarantees a 42 sided dice by rolling twice? One roll operation is picking 1 dice and rolling it once."
    PROBLEM4 = "My pottery person made my mug wrong. The bottom is open and the top has been water-proof sealed. Can I still use the mug to hold water?"
    PROBLEM5 = "What you might make to express an emotion; what you need to do with your problems to defeat them? Clue: 4 letter word with the last letter being e"
    PROBLEM6 = "What you might put out to ascertain where someone's at? Clue: *E**E*"
    PROBLEM7 = "By sounding it out, and counting with your fingers, the answer will come. Clue: *A**U"
    problem = PROBLEM3
    logger.info(f"Problem: {problem}")
    logger.info("="*60)
    
    config = SolverConfig(beam_width=2, max_depth=2)
    solver = create_default_solver(config, use_llm_validators=True)
    
    # Solve asynchronously
    result = await solver.solve(problem)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎯 FINAL RESULTS:")
    logger.info(f"   Solved: {result.solved}")
    logger.info(f"   Expanded nodes: {result.expanded_nodes}")
    if result.final_solution_text:
        logger.info(f"   Final solution: {result.final_solution_text}")
    logger.info("="*60)
    
    # Example of accessing tree state during/after solving
    tree_state = solver.get_tree_state()
    logger.info(f"\n🌳 TREE STATE:")
    logger.info(f"   Root node: {tree_state['root_node'] is not None}")
    logger.info(f"   Current depth: {tree_state['current_depth']}")
    logger.info(f"   Is solving: {tree_state['is_solving']}")
    logger.info(f"   Solving complete: {tree_state['solving_complete']}")
    logger.info(f"   Frontier size: {tree_state['frontier_size']}")
    
    # Close HTTP sessions
    await solver.thought_generator.close()
    if hasattr(solver.idea_validator, 'close'):
        await solver.idea_validator.close()
    if hasattr(solver.solution_validator, 'close'):
        await solver.solution_validator.close()


if __name__ == "__main__":
    asyncio.run(main())
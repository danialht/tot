"""
Tree-of-Thoughts (ToT) implementation with Cerebras integration.
Based on arXiv:2305.10601 with fail-fast behavior and extensible design.
"""

import os
import re
import json
import logging
import asyncio
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
        valid_intents = ["propose", "decompose", "check", "compute", "decide", 
                        "analyze", "solve", "clarify", "verify", "optimize", 
                        "explore", "reframe", "experiment", "mitigate", "investigate", "conclude"]
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
        return {
            "id": f"node_{id(self)}",
            "depth": self.depth,
            "subproblem": self.state.subproblem_text,
            "chain_of_thought": [{
                "text": t.text,
                "rationale": t.rationale,
                "confidence": t.confidence,
                "intent": t.intent,
                "candidate": t.candidate
            } for t in self.state.scratchpad],
            "generated_thoughts": [{
                "text": t.text,
                "rationale": t.rationale,
                "confidence": t.confidence,
                "intent": t.intent,
                "candidate": t.candidate,
                "score": self.thought_scores.get(t.text, 0.0)
            } for t in self.generated_thoughts],
            "candidate_solution": self.state.candidate_solution,
            "value_estimate": self.value_estimate,
            "terminal_status": self.terminal_status,
            "processing_status": self.processing_status,
            "input_context": self.input_context,
            "children_count": len(self.children),
            "incoming_thought": {
                "text": self.incoming_thought.text,
                "rationale": self.incoming_thought.rationale,
                "confidence": self.incoming_thought.confidence,
                "intent": self.incoming_thought.intent,
                "candidate": self.incoming_thought.candidate
            } if self.incoming_thought else None
        }


# ============================================================================
# Abstract Interfaces
# ============================================================================

class ThoughtGenerator(ABC):
    """Abstract base class for generating thoughts."""
    
    @abstractmethod
    async def generate(self, state: NodeState, max_ideas_hint: Optional[int] = None) -> List[Thought]:
        """Generate thoughts for the given node state."""
        pass
    
    async def generate_streaming(self, state: NodeState, max_ideas_hint: Optional[int] = None) -> AsyncIterator[Thought]:
        """Generate thoughts with streaming support. Default implementation calls generate()."""
        thoughts = await self.generate(state, max_ideas_hint)
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
    max_depth: int = 4
    selection_top_k: int = 2
    max_ideas_hint_per_node: Optional[int] = None
    temperature: float = 0.7
    max_retries: int = 2
    cerebras_model: str = "gpt-oss-120b"
    reasoning_effort: str = "low"

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
    
    async def generate(self, state: NodeState, max_ideas_hint: Optional[int] = None) -> List[Thought]:
        """Generate thoughts using Cerebras API."""
        max_ideas = max_ideas_hint or self.config.max_ideas_hint_per_node
        
        logger.info(f"🤖 Calling Cerebras API for thought generation")
        logger.info(f"📝 Subproblem: {state.subproblem_text}")
        logger.info(f"📊 Current scratchpad has {len(state.scratchpad)} thoughts")
        
        prompt = self._build_prompt(state, max_ideas)
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
                
                return deduplicated
                
            except Exception as e:
                if attempt == self.config.max_retries:
                    logger.error(f"Failed to generate thoughts after {self.config.max_retries + 1} attempts: {e}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
        
        return []
    
    async def generate_streaming(self, state: NodeState, max_ideas_hint: Optional[int] = None) -> AsyncIterator[Thought]:
        """Generate thoughts with streaming support."""
        max_ideas = max_ideas_hint or self.config.max_ideas_hint_per_node
        
        logger.info(f"🤖 Starting streaming thought generation")
        logger.info(f"📝 Subproblem: {state.subproblem_text}")
        
        prompt = self._build_prompt(state, max_ideas)
        
        async for thought in self._call_cerebras_streaming(prompt):
            yield thought
    
    def _build_prompt(self, state: NodeState, max_ideas: Optional[int]) -> str:
        """Build the prompt for thought generation using configured templates."""
        # Build context string
        context = build_context_string(state.scratchpad, state.candidate_solution)
        
        # Format max ideas hint
        max_hint = f" (aim for {max_ideas} strong insights)" if max_ideas else ""
        
        # Get template and format it
        template = PROMPTS["thought_generation"]
        prompt = template.format(
            problem=state.subproblem_text,
            context=context,
            max_hint=max_hint
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
            "max_tokens": 1000
        }
        
        async with session.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Cerebras API error: {response.status} - {error_text}")
            
            result = await response.json()
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
            "max_tokens": 1000,
            "stream": True
        }
        
        async with session.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Cerebras API error: {response.status} - {error_text}")
            
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
                            if "content" in delta:
                                content = delta["content"]
                                collected_content += content
                                
                                # Try to parse thoughts from accumulated content
                                # This is a simple approach - in practice you might want more sophisticated parsing
                                if collected_content.count('\n\n') > 0 or "Confidence:" in collected_content:
                                    thoughts = self._parse_thoughts(collected_content)
                                    for thought in thoughts:
                                        yield thought
                                    collected_content = ""  # Reset for next thought
                                    
                    except json.JSONDecodeError:
                        continue
            
            # Parse any remaining content
            if collected_content.strip():
                thoughts = self._parse_thoughts(collected_content)
                for thought in thoughts:
                    yield thought
    
    def _parse_thoughts(self, text: str) -> List[Thought]:
        """Parse thoughts from Cerebras response."""
        # Check for empty response keywords
        text_lower = text.lower()
        empty_keywords = ["no strong ideas", "no critical insights needed"]
        for keyword in empty_keywords:
            if keyword in text_lower:
                return []
        
        thoughts = []
        
        # Split into numbered items
        items = re.split(r'\n\s*\d+\.\s*', text)
        if len(items) > 1:
            items = items[1:]  # Remove text before first numbered item
        else:
            # Try alternative splitting
            items = [text]
        
        for item in items:
            thought = self._parse_single_thought(item.strip())
            if thought:
                thoughts.append(thought)
        
        return thoughts
    
    def _parse_single_thought(self, item: str) -> Optional[Thought]:
        """Parse a single thought from text."""
        try:
            # Extract fields using regex - support multiple field names
            idea_match = re.search(r'(?:Idea|Insight|Risk):\s*(.+?)(?=\n|Why:|$)', item, re.IGNORECASE | re.DOTALL)
            why_match = re.search(r'Why:\s*(.+?)(?=\n|Confidence:|$)', item, re.IGNORECASE | re.DOTALL)
            conf_match = re.search(r'Confidence:\s*([0-9]*\.?[0-9]+)', item, re.IGNORECASE)
            intent_match = re.search(r'Intent:\s*(\w+)', item, re.IGNORECASE)
            answer_match = re.search(r'Answer:\s*(.+?)(?=\n\s*\n|\n\s*\d+\.|$)', item, re.IGNORECASE | re.DOTALL)
            
            if not idea_match:
                return None
            
            text = idea_match.group(1).strip()
            rationale = why_match.group(1).strip() if why_match else "No rationale provided"
            confidence = float(conf_match.group(1)) if conf_match else 0.4
            intent = intent_match.group(1).lower() if intent_match else "propose"
            candidate = answer_match.group(1).strip() if answer_match else None
            
            # Validate confidence range
            confidence = max(0.0, min(1.0, confidence))
            
            # Validate intent - expanded set for new prompt types
            valid_intents = ["propose", "decompose", "check", "compute", "decide", 
                           "analyze", "solve", "clarify", "verify", "optimize", 
                           "explore", "reframe", "experiment", "mitigate", "investigate", "conclude"]
            if intent not in valid_intents:
                intent = "analyze"  # Default for insight-based prompts
            
            return Thought(
                text=text,
                rationale=rationale,
                confidence=confidence,
                intent=intent,
                candidate=candidate,
                metadata={"raw_item": item}
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse thought item: {e}")
            return None


class RuleBasedIdeaValidator(IdeaValidator):
    """Rule-based validator for thoughts."""
    
    async def evaluate(self, state: NodeState, thought: Thought) -> float:
        """Evaluate thought using heuristic scoring."""
        score = thought.confidence
        
        # Check for duplication against scratchpad
        normalized_new = normalize_text(thought.text)
        for existing in state.scratchpad:
            if normalize_text(existing.text) == normalized_new:
                return 0.0  # Duplicate
        
        # Reject trivial/empty thoughts
        if len(thought.text.strip()) < 10:
            return 0.0
        
        # Boost actionable intents
        intent_boosts = {
            "compute": 0.1,
            "check": 0.1,
            "decide": 0.15,
            "propose": 0.05,
            "decompose": 0.05
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
            
            prompt = PROMPTS["thought_validation"].format(
                problem=state.subproblem_text,
                context=context,
                thought_text=thought.text,
                thought_rationale=thought.rationale,
                thought_intent=thought.intent
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
            logger.warning(f"LLM validation failed: {e}, falling back to confidence score")
            return thought.confidence
    
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
            "temperature": 0.3,  # Lower temperature for validation
            "max_tokens": 200
        }
        
        async with session.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Cerebras API error: {response.status} - {error_text}")
            
            result = await response.json()
            if "choices" not in result or not result["choices"]:
                raise RuntimeError("No choices in Cerebras response")
            
            return result["choices"][0]["message"]["content"]
    
    def _parse_validation_score(self, text: str) -> float:
        """Parse validation score from response."""
        # Look for "Score: X.X" pattern
        score_match = re.search(r'Score:\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
        if score_match:
            score = float(score_match.group(1))
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        
        # Fallback: look for any number between 0 and 1
        number_match = re.search(r'\b([0-9]*\.?[0-9]+)\b', text)
        if number_match:
            score = float(number_match.group(1))
            if 0.0 <= score <= 1.0:
                return score
        
        # Default fallback
        logger.warning(f"Could not parse validation score from: {text[:100]}...")
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
            "temperature": 0.1,  # Very low temperature for validation
            "max_tokens": 300
        }
        
        async with session.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Cerebras API error: {response.status} - {error_text}")
            
            result = await response.json()
            if "choices" not in result or not result["choices"]:
                raise RuntimeError("No choices in Cerebras response")
            
            return result["choices"][0]["message"]["content"]
    
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
    best_node: Optional[SubProblemNode]
    expanded_nodes: int
    solved: bool
    trace: List[str] = field(default_factory=list)


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
        
        # Public tree state attributes
        self.root_node: Optional[SubProblemNode] = None
        self.current_frontier: List[SubProblemNode] = []
        self.best_node: Optional[SubProblemNode] = None
        self.expanded_nodes: int = 0
        self.current_depth: int = 0
        self.is_solving: bool = False
        self.solving_complete: bool = False
        self.trace: List[str] = []
        
    def get_tree_state(self) -> Dict[str, Any]:
        """Get current tree state for external access."""
        return {
            "root_node": self.root_node.to_dict() if self.root_node else None,
            "current_frontier": [node.to_dict() for node in self.current_frontier],
            "best_node": self.best_node.to_dict() if self.best_node else None,
            "expanded_nodes": self.expanded_nodes,
            "current_depth": self.current_depth,
            "is_solving": self.is_solving,
            "solving_complete": self.solving_complete,
            "trace": self.trace.copy(),
            "frontier_size": len(self.current_frontier)
        }
    
    def get_solution_path(self) -> List[Dict[str, Any]]:
        """Get the path from root to best solution."""
        if not self.best_node:
            return []
        
        path = []
        current = self.best_node
        
        # Traverse up to root (we'd need parent pointers for this - simplified approach)
        # For now, just return the chain of thought from the best node
        path.append({
            "node": current.to_dict(),
            "chain_of_thought": [
                {
                    "text": t.text,
                    "rationale": t.rationale,
                    "confidence": t.confidence,
                    "intent": t.intent,
                    "candidate": t.candidate
                } for t in current.get_chain_of_thought()
            ]
        })
        
        return path
    
    async def solve(self, problem_text: str, objective: str = "default", log: bool = True) -> SearchResult:
        """Solve the problem using Tree-of-Thoughts."""
        if not log:
            logging.disable(logging.CRITICAL)
        
        self.is_solving = True
        self.solving_complete = False
        self.expanded_nodes = 0
        self.current_depth = 0
        self.trace = []
        
        logger.info(f"🌳 Starting Tree-of-Thoughts search")
        logger.info(f"📋 Problem: {problem_text}")
        logger.info(f"🎯 Objective: {objective}")
        logger.info(f"⚙️ Config: beam_width={self.config.beam_width}, max_depth={self.config.max_depth}")
        
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
        
        # Initialize search state
        self.current_frontier = [self.root_node]
        self.best_node = None
        
        # Tree is now ready for solving
        
        # Beam search by depth
        for depth in range(self.config.max_depth + 1):
            self.current_depth = depth
            
            if not self.current_frontier:
                logger.info(f"🚫 No nodes in frontier at depth {depth}, stopping search")
                self.trace.append(f"Depth {depth}: No nodes in frontier, stopping")
                break
            
            logger.info(f"🔍 Depth {depth}: Processing {len(self.current_frontier)} nodes")
            self.trace.append(f"Depth {depth}: {len(self.current_frontier)} nodes in frontier")
            next_frontier = []
            
            # Process nodes at this depth
            
            for i, node in enumerate(self.current_frontier):
                logger.info(f"🔄 Processing node {i+1}/{len(self.current_frontier)} at depth {depth}")
                node.processing_status = "processing"
                
                # Check if node is solved
                
                # Check if solved
                if await self.solution_validator.is_solved(node.state):
                    node.terminal_status = "solved"
                    node.processing_status = "completed"
                    self.best_node = node
                    logger.info(f"✅ Solution found at depth {depth}!")
                    logger.info(f"💡 Solution: {node.state.candidate_solution}")
                    self.trace.append(f"Solution found at depth {depth}")
                    
                    # Solution found, return result
                    
                    self.is_solving = False
                    self.solving_complete = True
                    return SearchResult(
                        best_node=self.best_node,
                        expanded_nodes=self.expanded_nodes,
                        solved=True,
                        trace=self.trace
                    )
                
                # Skip if at max depth
                if depth >= self.config.max_depth:
                    logger.info(f"⏹️ Reached max depth {self.config.max_depth}, not expanding")
                    continue
                
                # Expand node
                logger.info(f"🌿 Expanding node at depth {depth}")
                children = await self._expand_node(node)
                self.expanded_nodes += 1
                logger.info(f"📈 Generated {len(children)} children")
                
                # Update value estimates for optimization
                for child in children:
                    quality = await self.solution_validator.quality(child.state)
                    if quality is not None:
                        if objective == "optimize:min":
                            child.value_estimate = -quality  # Higher is better
                        else:
                            child.value_estimate = quality
                
                next_frontier.extend(children)
            
            # Select top nodes for next frontier (beam search)
            if next_frontier:
                logger.info(f"📊 Beam selection: {len(next_frontier)} candidates -> {min(self.config.beam_width, len(next_frontier))} selected")
                
                # Sort by value estimate (higher is better)
                next_frontier.sort(
                    key=lambda n: n.value_estimate if n.value_estimate is not None else 0.0,
                    reverse=True
                )
                
                # Log top candidates
                for i, node in enumerate(next_frontier[:5]):  # Show top 5
                    estimate = node.value_estimate or 0.0
                    logger.info(f"  #{i+1}: score={estimate:.3f}, solution='{node.state.candidate_solution or 'None'}'")
                
                self.current_frontier = next_frontier[:self.config.beam_width]
                
                # Update best node
                if self.current_frontier and (self.best_node is None or 
                    (self.current_frontier[0].value_estimate or 0) > (self.best_node.value_estimate or 0)):
                    old_best = self.best_node.state.candidate_solution if self.best_node else None
                    self.best_node = self.current_frontier[0]
                    new_best = self.best_node.state.candidate_solution
                    logger.info(f"🏆 New best node: '{old_best}' -> '{new_best}' (score: {self.best_node.value_estimate:.3f})")
        
        logger.info(f"🏁 Search completed after {self.expanded_nodes} expansions")
        logger.info(f"🎯 Final result: solved={False}, best_solution='{self.best_node.state.candidate_solution if self.best_node else None}'")
        
        return SearchResult(
            best_node=self.best_node,
            expanded_nodes=self.expanded_nodes,
            solved=False,
            trace=self.trace
        )
    
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
                content_logger.info(f"  Confidence: {thought.confidence}")
                content_logger.info(f"  Intent: {thought.intent}")
                if thought.candidate:
                    content_logger.info(f"  Candidate Answer: {thought.candidate}")
                if thought.metadata:
                    content_logger.info(f"  Metadata: {thought.metadata}")
                content_logger.info("")
            content_logger.info("=" * 60)
        
        # Brief summary for main logger
        for i, thought in enumerate(thoughts, 1):
            logger.info(f"  {i}. {thought.text[:50]}... (confidence: {thought.confidence:.2f}, intent: {thought.intent})")
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
        
        # Select top thoughts
        scored_thoughts.sort(key=lambda x: x[1], reverse=True)
        top_thoughts = scored_thoughts[:self.config.selection_top_k]
        logger.info(f"✅ Selected top {len(top_thoughts)} thoughts (from {len(scored_thoughts)} valid)")
        
        # Create children
        children = []
        for thought, score in top_thoughts:
            # Create new state
            new_scratchpad = node.state.scratchpad + [thought]
            new_candidate = thought.candidate if thought.candidate else node.state.candidate_solution
            
            child_state = NodeState(
                subproblem_text=node.state.subproblem_text,
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
    problem = 'We fill a glass with water up to the brim. we turn it upsidedown. give an estimate for how many water molecules are in the glass'
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
    if result.best_node and result.best_node.state.candidate_solution:
        logger.info(f"   Best solution: {result.best_node.state.candidate_solution}")
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
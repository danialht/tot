"""
Simple prompt loader for Tree-of-Thoughts implementation.
"""

import json
import os
from typing import Dict

def load_prompts() -> Dict[str, str]:
    """Load prompts from JSON file."""
    prompts_file = os.path.join(os.path.dirname(__file__), 'prompts.json')
    with open(prompts_file, 'r') as f:
        return json.load(f)

def build_context_string(scratchpad, candidate_solution) -> str:
    """Build the context string from node state."""
    context_parts = []
    
    if scratchpad:
        context_parts.append("Previous insights:")
        for i, thought in enumerate(scratchpad, 1):
            context_parts.append(f"{i}. {thought.text}")
            if thought.candidate:
                context_parts.append(f"   → Candidate: {thought.candidate}")
    
    if candidate_solution:
        context_parts.append(f"Current best answer: {candidate_solution}")
    
    return "\n".join(context_parts) + "\n" if context_parts else ""

# Load prompts at module level
PROMPTS = load_prompts()
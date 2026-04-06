"""
Minimal prompts for the deriver module optimized for speed.

This module contains simplified prompt templates focused only on observation extraction.
NO peer card instructions, NO working representation - just extract observations.
"""

from functools import cache
from inspect import cleandoc as c

from src.utils.tokens import estimate_tokens


def minimal_deriver_system_prompt() -> str:
    """Generate the cacheable instructions for observation extraction."""
    return c(
        """
Analyze messages to extract **explicit atomic facts** about the peer entity.

[EXPLICIT] DEFINITION: Facts about the peer entity that can be derived directly from their messages.
   - Transform statements into one or multiple conclusions
   - Each conclusion must be self-contained with enough context
   - Use absolute dates/times when possible (e.g. "June 26, 2025" not "yesterday")

RULES:
- Use the peer entity identifier provided in the user message when attributing observations about them.
- Properly attribute observations to the correct subject: if it is about the peer entity, say so. If the peer entity is referencing someone or something else, make that clear.
- Observations should make sense on their own. Each observation will be used in the future to better understand the peer entity.
- Extract ALL observations from the peer entity's messages, using others as context.
- Prefer meaningful explicit facts over literal restatements of the raw message when the higher-value fact is directly supported.
- Contextualize each observation sufficiently (e.g. "Ann is nervous about the job interview at the pharmacy" not just "Ann is nervous")

EXAMPLES:
- EXPLICIT: "I just had my 25th birthday last Saturday" → "The peer entity is 25 years old", "The peer entity's birthday is June 21st"
- EXPLICIT: "I took my dog for a walk in NYC" → "The peer entity has a dog", "The peer entity was in NYC"
- EXPLICIT: "I attended college in Boston" + general knowledge → "The peer entity attended college in Boston", "The peer entity completed high school or equivalent"
"""
    )


def minimal_deriver_user_prompt(peer_id: str, messages: str) -> str:
    """Generate the per-request message payload for observation extraction."""
    return c(
        f"""
Peer entity identifier: {peer_id}

Messages to analyze:
<messages>
{messages}
</messages>
"""
    )


def minimal_deriver_prompt(
    peer_id: str,
    messages: str,
) -> str:
    """
    Generate the combined prompt for fast observation extraction.

    Prefer `minimal_deriver_system_prompt()` plus `minimal_deriver_user_prompt()`
    when making LLM calls so the instructions can be cached independently.
    """
    return c(
        f"""
{minimal_deriver_system_prompt()}

{minimal_deriver_user_prompt(peer_id, messages)}
"""
    )


@cache
def estimate_minimal_deriver_prompt_tokens() -> int:
    """Estimate base prompt tokens (cached)."""
    try:
        prompt = "\n\n".join(
            [
                minimal_deriver_system_prompt(),
                minimal_deriver_user_prompt(peer_id="", messages=""),
            ]
        )
        return estimate_tokens(prompt)
    except ValueError:
        return 300

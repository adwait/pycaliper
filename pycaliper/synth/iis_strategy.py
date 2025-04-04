"""
File: pycaliper/synth/iis_strategy.py
This file is a part of the PyCaliper tool.
See LICENSE.md for licensing information.

Author: Adwait Godbole, UC Berkeley
"""

import os
import logging
from openai import OpenAI
from random import choice, shuffle

DEFAULT_MODEL = "gpt-4o"

logger = logging.getLogger(__name__)


def get_client():
    """Load the OpenAI client using the API key from the environment variable.

    Returns:
        OpenAI: An instance of the OpenAI client.
    """
    logger.debug("Loading API key from environment variable")
    api_key = os.environ.get("OPENAI_API_KEY")
    return OpenAI(organization="org-msi6higGCSQ16ml5gj0krGkZ", api_key=api_key)


def extract_markdown(content: str) -> str:
    """Extract the markdown content from the provided text.

    Args:
        content (str): The text content to extract markdown from.

    Returns:
        str: The extracted markdown content.
    """
    test_lines = content.split("\n")
    backtick_indices = [
        i for i, line in enumerate(test_lines) if line.startswith("```")
    ]
    if len(backtick_indices) < 2:
        return ""
    return "\n".join(test_lines[backtick_indices[0] + 1 : backtick_indices[1]])


class IISStrategy:
    def __init__(self):
        """Initialize the IISStrategy."""
        pass

    def get_candidate_order(self, unexplored, **kwargs):
        """Get the order of candidates to explore.

        Args:
            unexplored: The unexplored candidates.
            **kwargs: Additional arguments.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError()

    def get_next_candidate(self, unexplored, **kwargs):
        """Get the next candidate to explore.

        Args:
            unexplored: The unexplored candidates.
            **kwargs: Additional arguments.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError()

    def get_stats(self) -> str:
        """Get the statistics of the strategy.

        Returns:
            str: The statistics of the strategy.
        """
        raise NotImplementedError()


class LLMStrategy(IISStrategy):
    def __init__(self):
        """Initialize the LLMStrategy."""
        super().__init__()

        self.client = get_client()
        self.messages = []

        # Model
        self.model = DEFAULT_MODEL

        self.samples = 0
        self.queries = 0

    def get_candidate_order(self, unexplored, ctx, prev_attempts=[], **kwargs):
        """Get the order of candidates using the LLM model.

        Args:
            unexplored: The unexplored candidates.
            ctx: The context of known invariants.
            prev_attempts: Previous attempts.
            **kwargs: Additional arguments.

        Returns:
            list: The ordered list of candidates.
        """
        if len(unexplored) == 1:
            return unexplored
        prompt = (
            f"Here are the list of known (GUARANTEED) invariants: {ctx}. \n\n"
            f"Please provide a list representing a reasonable candidate order on: {unexplored}. "
            f"The candidates first in the output list should be the most likely choices based on the "
            f"GUARANTEED candidates. Look for patterns in the GUARANTEED invariants and prioritize them in the generated list.\n"
            f"You have already generated the following candidate orders: {prev_attempts}. Do not regenerate. \n"
            f"Please return a single Python candidate list candidate in backticks, e.g. as:\n"
            f"```\n"
            f"[candidate1, candidate2, ...]\n"
            f"```\n"
        )
        response = self._query(prompt)
        logger.debug(f"Received response: {response}")
        # Parse the response and get element in backticks
        candidate = extract_markdown(response).strip()
        # This is a string representation of a list
        candidate = eval(candidate)
        logger.debug(f"Received candidate: {candidate}")
        return candidate

    def _query(self, prompt):
        """Query the LLM model with a prompt.

        Args:
            prompt (str): The prompt to query.

        Returns:
            str: The response from the LLM model.
        """
        self.messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model, messages=self.messages
        )
        content = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": content.strip()})
        self.queries += 1
        return content

    def get_next_candidate(self, unexplored, current, ctx, **kwargs):
        """Get the next candidate using the LLM model.

        Args:
            unexplored: The unexplored candidates.
            current: The current candidates.
            ctx: The context of known invariants.
            **kwargs: Additional arguments.

        Returns:
            str: The next candidate.
        """
        self.samples += 1
        if len(unexplored) == 1:
            return unexplored[0]
        prompt = (
            f"Here are the list of known (GUARANTEED) invariants: {ctx}. \n\n"
            f"Here is the list of current invariant candidates: {current}. "
            f"Please provide the next reasonable candidate from the following list: {unexplored}. "
            f"Attempt pattern matching based on GUARANTEED candidates first and then matching the current candidates to suggest the next one.\n"
            f"Please return a single candidate in backticks, e.g. as:\n"
            f"```\n"
            f"candidate\n"
            f"```\n"
        )
        response = self._query(prompt)
        logger.debug(f"Received response: {response}")
        # Parse the response and get element in backticks
        candidate = extract_markdown(response).strip()
        logger.debug(f"Received candidate: {candidate}")
        return candidate

    def get_stats(self) -> str:
        """Get the statistics of the LLM strategy.

        Returns:
            str: The statistics of the LLM strategy.
        """
        return f"LLMStrat: {self.samples} samples, {self.queries} queries"


class RandomStrategy(IISStrategy):
    def __init__(self):
        """Initialize the RandomStrategy."""
        self.samples = 0

    def get_candidate_order(self, unexplored, **kwargs):
        """Get the order of candidates randomly.

        Args:
            unexplored: The unexplored candidates.
            **kwargs: Additional arguments.

        Returns:
            list: The shuffled list of candidates.
        """
        shuffle(unexplored)
        return unexplored

    def get_next_candidate(self, unexplored, **kwargs):
        """Get the next candidate randomly.

        Args:
            unexplored: The unexplored candidates.
            **kwargs: Additional arguments.

        Returns:
            str: The next candidate.
        """
        self.samples += 1
        return choice(unexplored)

    def get_stats(self) -> str:
        """Get the statistics of the random strategy.

        Returns:
            str: The statistics of the random strategy.
        """
        return f"RandomStrat: {self.samples} samples"


class SeqStrategy(IISStrategy):
    def __init__(self):
        """Initialize the SeqStrategy."""
        self.samples = 0

    def get_candidate_order(self, unexplored, **kwargs):
        """Get the order of candidates sequentially.

        Args:
            unexplored: The unexplored candidates.
            **kwargs: Additional arguments.

        Returns:
            list: The list of candidates in order.
        """
        return unexplored

    def get_next_candidate(self, unexplored, **kwargs):
        """Get the next candidate sequentially.

        Args:
            unexplored: The unexplored candidates.
            **kwargs: Additional arguments.

        Returns:
            str: The next candidate.
        """
        self.samples += 1
        return unexplored[0]

    def get_stats(self) -> str:
        """Get the statistics of the sequential strategy.

        Returns:
            str: The statistics of the sequential strategy.
        """
        return f"SeqStrat: {self.samples} samples"

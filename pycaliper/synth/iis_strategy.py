import os
import logging
from openai import OpenAI
from random import choice, shuffle

DEFAULT_MODEL = "gpt-4o"

logger = logging.getLogger(__name__)


def get_client():
    # Load API key from api.key file
    logger.debug("Loading API key from environment variable")
    api_key = os.environ.get("OPENAI_API_KEY")
    return OpenAI(organization="org-msi6higGCSQ16ml5gj0krGkZ", api_key=api_key)


def extract_markdown(content: str) -> str:
    """Extracts the markdown from the test content"""
    test_lines = content.split("\n")
    backtick_indices = [
        i for i, line in enumerate(test_lines) if line.startswith("```")
    ]
    if len(backtick_indices) < 2:
        return ""
    return "\n".join(test_lines[backtick_indices[0] + 1 : backtick_indices[1]])


class IISStrategy:
    def __init__(self):
        pass

    def get_candidate_order(self, unexplored, **kwargs):
        raise NotImplementedError()

    def get_next_candidate(self, unexplored, **kwargs):
        raise NotImplementedError()

    def get_stats(self) -> str:
        raise NotImplementedError()


class LLMStrategy(IISStrategy):
    def __init__(self):
        super().__init__()

        self.client = get_client()
        self.messages = []

        # Model
        self.model = DEFAULT_MODEL

        self.samples = 0
        self.queries = 0

    def get_candidate_order(self, unexplored, ctx, prev_attempts=[], **kwargs):
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
        # Parse the response and get elemet in backticks
        candidate = extract_markdown(response).strip()
        # This is a string representation of a list
        candidate = eval(candidate)
        logger.debug(f"Received candidate: {candidate}")
        return candidate

    def _query(self, prompt):
        self.messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model, messages=self.messages
        )
        content = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": content.strip()})
        self.queries += 1
        return content

    def get_next_candidate(self, unexplored, current, ctx, **kwargs):
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
        # Parse the response and get elemet in backticks
        candidate = extract_markdown(response).strip()
        logger.debug(f"Received candidate: {candidate}")
        return candidate

    def get_stats(self) -> str:
        return f"LLMStrat: {self.samples} samples, {self.queries} queries"


class RandomStrategy(IISStrategy):
    def __init__(self):
        self.samples = 0

    def get_candidate_order(self, unexplored, **kwargs):
        shuffle(unexplored)
        return unexplored

    def get_next_candidate(self, unexplored, **kwargs):
        self.samples += 1
        return choice(unexplored)

    def get_stats(self) -> str:
        return f"RandomStrat: {self.samples} samples"


class SeqStrategy(IISStrategy):
    def __init__(self):
        self.samples = 0

    def get_candidate_order(self, unexplored, **kwargs):
        return unexplored

    def get_next_candidate(self, unexplored, **kwargs):
        self.samples += 1
        return unexplored[0]

    def get_stats(self) -> str:
        return f"SeqStrat: {self.samples} samples"

"""Shared test helpers."""

import _path  # noqa: F401

from roadglyph import prompts


class Scripted:
    """Mock responder: fixed classification, then a queue of transcription replies.

    A queue item that is an Exception instance is raised instead of returned.
    The last item repeats once the queue is exhausted.
    """

    def __init__(self, category, answers):
        self.category = category
        self.answers = list(answers)
        self.index = 0

    def __call__(self, image, system, user):
        if user == prompts.CLASSIFY_PROMPT:
            if isinstance(self.category, Exception):
                raise self.category
            return self.category
        item = self.answers[min(self.index, len(self.answers) - 1)]
        self.index += 1
        if isinstance(item, Exception):
            raise item
        return item

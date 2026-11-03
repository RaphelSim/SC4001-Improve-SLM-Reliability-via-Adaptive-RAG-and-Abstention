"""Structured errors for malformed or contract-invalid model output."""

from reliable_rag.contracts.enums import AbstentionReason


class ModelOutputError(RuntimeError):
    """A model response stayed invalid after the configured formatting retry."""

    def __init__(self, *, task: str, question_id: str, attempts: int, reason: str) -> None:
        self.task = task
        self.question_id = question_id
        self.attempts = attempts
        self.reason = reason
        self.abstention_reason = AbstentionReason.INVALID_MODEL_OUTPUT
        super().__init__(
            f"{task} output for question {question_id!r} remained invalid "
            f"after {attempts} attempt(s): {reason}"
        )

    def as_dict(self) -> dict[str, str | int]:
        """Return a safe structured failure summary without retaining raw model output."""
        return {
            "question_id": self.question_id,
            "task": self.task,
            "failure_reason": self.abstention_reason.value,
            "attempts": self.attempts,
            "message": self.reason,
        }

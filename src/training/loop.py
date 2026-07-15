"""Training-loop boundary.

Add one concern at a time: loss, optimizer, validation, checkpoints, resume, then precision and distributed execution.
"""


def train() -> None:
    """Run one reproducible training experiment from a resolved config."""
    raise NotImplementedError("Phase 5 exercise: implement a one-batch overfit loop first.")

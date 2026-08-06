from __future__ import annotations

from collections import deque

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch


class TargetHiddenLagCache:
    """Feeds the draft a *stale* target hidden -- one lagged by ``lag_steps``
    decode steps instead of the fresh one -- so we can measure how much accept
    length degrades under staleness (the parallel-DSpark go/no-go ablation).

    ``lag_steps == 0`` is the identity / vanilla case: ``enabled`` is False and
    the cache is never touched, so the default DSpark path is byte-identical and
    pays zero overhead. ``lag_steps >= 1`` turns the ablation on.

    Sits next to the injection seam (``TargetVerifyExecutor.commit_hidden``): the
    caller snapshots this step's fresh hidden, then swaps in the lagged one right
    before ``inject_target_hidden`` writes it into the draft KV pool.

    Wiring (env var, object, three call sites) guarantees depth-0 parity; the
    method bodies implement per-request keyed staleness -- keyed by ``rid``,
    which survives the batch reordering that continuous batching causes (the Q3
    gate) -- with a fresh fallback during each request's cold-start rounds.
    """

    def __init__(self, *, lag_steps: int) -> None:
        self._lag_steps = int(lag_steps)
        # Per-request store of recent fresh-hidden snapshots. Step 2 decides the
        # KEY (Q3: what identifies "the same request" across steps whose batch
        # rows are reordered as requests finish/join) and the ring structure.
        self._store: dict = {}

    @property
    def enabled(self) -> bool:
        return self._lag_steps > 0

    @property
    def lag_steps(self) -> int:
        return self._lag_steps

    def snapshot(self, *, batch: ScheduleBatch, fresh_hidden: torch.Tensor) -> None:
        """Step 2. Record this step's fresh target hidden for every request.

        ``fresh_hidden`` is ``[bs, verify_num_draft_tokens, H]``; row ``i``
        belongs to ``batch.reqs[i]``. Store a DETACHED CLONE of each row -- the
        underlying capture buffer is reused by the next forward, so keeping a
        view would alias and corrupt it (same reason ``write_target_hidden_kv``
        treats ``ctx_hidden`` as read-only). Keep at most ``lag_steps + 1`` per
        request; drop the oldest.

        Q3 gate: what you key the store on is the whole question -- reason it
        through before you implement (batch-row index vs request identity, under
        reordering).
        """
        for req, target_hidden in zip(batch.reqs, fresh_hidden):
            if req.rid not in self._store:
                self._store[req.rid] = deque(maxlen=self._lag_steps + 1)
            self._store[req.rid].append(target_hidden.detach().clone())

    def get_lagged(
        self, *, batch: ScheduleBatch, fresh_hidden: torch.Tensor
    ) -> torch.Tensor:
        """Step 2. Return a ``[bs, verify_num_draft_tokens, H]`` tensor whose row
        ``i`` is the snapshot from ``lag_steps`` steps ago for ``batch.reqs[i]``
        -- the caller injects it at THIS step's positions (value-stale).

        Open sub-decisions to pin (part of Q3 / the mechanism to confirm with
        zhendonghua before trusting any number):
          - a request with fewer than ``lag_steps + 1`` snapshots (just joined)
            has no lagged hidden -- fall back to fresh, or skip the request? Define it.
          - "old value at current positions" (value-stale) vs "old value at old
            positions" (positional-lag): the ByteDance design must confirm which.
            This class only supplies the value; positions come from the caller.
        """
        lagged_rows = []
        for req, fresh_row in zip(batch.reqs, fresh_hidden):
            ring = self._store[req.rid]
            if len(ring) < self._lag_steps + 1:
                # Cold start: this request hasn't accumulated lag_steps+1
                # snapshots yet, so no lag_steps-old hidden exists -> use fresh.
                lagged_rows.append(fresh_row)
            else:
                # Ring holds the last lag_steps+1 snapshots oldest-first, so the
                # oldest (index 0) is exactly the one from lag_steps steps ago.
                lagged_rows.append(ring[0])
        return torch.stack(lagged_rows, dim=0)

    def evict(self, *, rid: str) -> None:
        """Step 2. Drop a finished request's snapshots. Called from
        ``DSparkWorkerV2.note_request_finished``. Must be a no-op if ``rid`` is
        absent (a request can finish without ever reaching commit_hidden)."""
        if rid not in self._store:
            return
        del self._store[rid]

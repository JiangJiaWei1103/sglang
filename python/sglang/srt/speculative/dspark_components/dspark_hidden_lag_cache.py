from __future__ import annotations

from collections import deque

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch


class TargetHiddenLagCache:
    """Simulate a drafter running ``lag_steps`` rounds AHEAD of the verifier, the
    way parallel DSpark actually behaves, so we can measure the accept-len drops.

    BACKFILL: every committed position eventually holds its real target hidden;
    injected ``lag_steps`` rounds later.
    FRONTIER: at any draft the last ``lag_steps`` rounds' just-committed positions
    have no real hidden; they get a FILL until the real one arrives ``lag_steps``
    rounds later and overwrites it.

    Previous impl overwrote every position with a stale hidden and NEVER backfilled.

    ``lag_steps == 0`` -> ``enabled`` is False; commit_hidden keeps the stock
    fresh injection (byte-identical vanilla). ``lag_steps >= 1`` turns it on.

    ``fill_mode`` -- what fills the transient frontier hole:
      * ``"repeat"``  : the last REAL target hidden (h at the last verified
        position) broadcast across the frontier. Still a content-phase mismatch,
        but TRANSIENT + frontier-only (much milder than variant A).
      * ``"gap"``     : nothing / no rows. NOT WIRED here -- a skipped write leaves
        garbage in the reused slot; true "no rows" needs the draft attention to
        exclude un-injected positions (a masking change), which this seam can't do.
      * ``"self_kv"`` : the draft's own K/V. NOT WIRED -- this seam doesn't hold
        the draft-side K/V.

    Per-request keyed (``rid``) so it survives batch reordering. Each request's
    first ``lag_steps`` rounds inject fresh (cold start ~ synced start).
    """

    _FILL_MODES = ("repeat", "gap", "self_kv")

    def __init__(self, *, lag_steps: int, fill_mode: str = "repeat") -> None:
        self._lag_steps = int(lag_steps)
        if fill_mode not in self._FILL_MODES:
            raise ValueError(
                f"unknown fill_mode {fill_mode!r}; expected one of {self._FILL_MODES}"
            )
        self._fill_mode = fill_mode
        # rid -> deque(maxlen lag+1) of one round's
        # (hidden [W,H], cache_loc [W], positions [W], commit_lens [1], commit_len int).
        # Everything needed to REPLAY that round's inject at its own slots when it
        # is backfilled lag_steps rounds later.
        self._queue: dict = {}

    @property
    def enabled(self) -> bool:
        return self._lag_steps > 0

    @property
    def lag_steps(self) -> int:
        return self._lag_steps

    @property
    def fill_mode(self) -> str:
        return self._fill_mode

    def evict(self, *, rid: str) -> None:
        """Drop a finished request's queue. No-op if absent (a request can finish
        without ever reaching commit_hidden)."""
        self._queue.pop(rid, None)

    def plan_step(
        self,
        *,
        batch: ScheduleBatch,
        fresh_hidden: torch.Tensor,  # [bs, W, H] this round's fresh target hidden
        verify_cache_loc_2d: torch.Tensor,  # [bs, W]    per-position draft-KV slot
        positions_2d: torch.Tensor,  # [bs, W]    per-position logical position
        commit_lens: torch.Tensor,  # [bs]       accepted length per request
    ) -> list[dict]:
        """Return the ``inject_target_hidden(**kwargs)`` calls for THIS round: the
        delayed BACKFILL (real hidden at its own old slots) + the FRONTIER FILL
        (fill_mode at the current slots). Cold-start rounds inject fresh instead.

        NOTE (assumption): a committed position's draft-KV slot is stable for the
        life of the request, so backfilling to a slot recorded ``lag_steps`` rounds
        ago lands on the same position. Holds for an active, un-preempted request
        (the bs=1 ablation runs). Revisit for preemption / bs>1 keying."""
        plans: list[dict] = []
        for i, req in enumerate(batch.reqs):
            q = self._queue.get(req.rid)
            if q is None:
                q = deque(maxlen=self._lag_steps + 1)
                self._queue[req.rid] = q

            h = (
                fresh_hidden[i].detach().clone()
            )  # [W, H] -- clone: capture buffer is reused
            loc = verify_cache_loc_2d[i].clone()  # [W]
            pos = positions_2d[i].clone()  # [W]
            c_len = commit_lens[i : i + 1].clone()  # [1]
            c_len_v = int(commit_lens[i])
            q.append((h, loc, pos, c_len, c_len_v))

            if len(q) < self._lag_steps + 1:
                # cold start: no lag-old hidden yet -> inject fresh at current slots.
                plans.append(self._inject(h, loc, pos, c_len))
                continue

            # steady state:
            # (1) BACKFILL the block from lag_steps rounds ago at ITS OWN slots (real).
            h_old, loc_old, pos_old, c_len_old, c_len_v_old = q[0]
            plans.append(self._inject(h_old, loc_old, pos_old, c_len_old))
            # (2) FILL the current frontier (transient; backfilled lag_steps rounds later).
            fill = self._frontier_fill(
                h_old=h_old, clen_old=c_len_v_old, width=h.shape[0]
            )
            plans.append(self._inject(fill, loc, pos, c_len))
        return plans

    def _frontier_fill(
        self, *, h_old: torch.Tensor, c_len_v_old: int, width: int
    ) -> torch.Tensor:
        """[W, H] hidden to write at the current frontier slots (fill_mode)."""
        if self._fill_mode == "repeat":
            # last REAL verified position's hidden, broadcast across the frontier.
            last_real = h_old[c_len_v_old - 1]  # [H]
            return last_real.unsqueeze(0).repeat(width, 1)  # [W, H]
        raise NotImplementedError(
            f"fill_mode={self._fill_mode!r} not wired yet: 'gap' needs the draft "
            f"attention to exclude un-injected slots (a masking change, not a skipped "
            f"write); 'self_kv' needs the draft's own K/V, absent at this seam. Only "
            f"'repeat' is implemented. See ssd-stale-poc/docs/design-options.md."
        )

    @staticmethod
    def _inject(
        hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> dict:
        """kwargs for one TargetHiddenKvInjector.inject_target_hidden call (bs=1 shape)."""
        return {
            "target_hidden": hidden,  # [W, H]
            "cache_loc": cache_loc,  # [W]
            "cache_loc_2d": cache_loc.unsqueeze(
                0
            ),  # [1, W]  -> prefix-valid write path
            "positions": positions,  # [W]
            "commit_lens": commit_lens,  # [1]
        }

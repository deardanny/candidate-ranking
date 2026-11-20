"""Article-independent response features for explicitly delayed event-time replay.

Request timestamps are known; feedback arrival timestamps are NOT known. `delay`
is an experimental maturity assumption. Never call this a measured online SLA.
All queries at t are scored before events at t can enter the state.
"""
from collections import Counter, deque
import numpy as np

WINDOWS = (3600, 86400, 604800)
FEEDBACK_FEATURES = [f"response_{name}_{part}" for name in ("all", "1h", "24h", "7d")
                     for part in ("exposures", "clicks", "ctr", "uncertainty")]
FEEDBACK_FEATURES += ["response_24h_ctr_rank", "response_24h_ctr_gap"]


class FeedbackState:
    def __init__(self, delay=3600, strength=20.):
        if delay < 0 or strength <= 0:
            raise ValueError("Invalid maturity/prior")
        self.delay = int(delay)
        self.strength = float(strength)
        self.now = None
        self.pending = deque()
        self.queues = {w: deque() for w in WINDOWS}
        self.exposures = {w: Counter() for w in (0, *WINDOWS)}
        self.clicks = {w: Counter() for w in (0, *WINDOWS)}
        self.totals = {w: [0, 0] for w in (0, *WINDOWS)}

    def _update(self, window, ids, clicked, sign):
        for key in ids:
            self.exposures[window][key] += sign
        for key in clicked:
            self.clicks[window][key] += sign
        self.totals[window][0] += sign * len(ids)
        self.totals[window][1] += sign * len(clicked)

    def advance(self, now):
        now = int(now)
        if self.now is not None and now < self.now:
            raise ValueError("Chronological replay required")
        self.now = now
        cutoff = now - self.delay
        while self.pending and self.pending[0][0] < cutoff:
            stamp, ids, clicked = self.pending.popleft()
            self._update(0, ids, clicked, 1)
            for w in WINDOWS:
                if stamp >= cutoff - w:
                    self.queues[w].append((stamp, ids, clicked))
                    self._update(w, ids, clicked, 1)
        for w in WINDOWS:
            while self.queues[w] and self.queues[w][0][0] < cutoff - w:
                _, ids, clicked = self.queues[w].popleft()
                self._update(w, ids, clicked, -1)

    def matrix(self, now, ids):
        self.advance(now)
        ids = list(map(int, ids))
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Nonempty unique candidates required")
        columns = []
        for w in (0, *WINDOWS):
            n = np.array([self.exposures[w][i] for i in ids], dtype=float)
            c = np.array([self.clicks[w][i] for i in ids], dtype=float)
            ne, nc = self.totals[w]
            # Fixed fallback only before the first mature observation.
            prior = nc / ne if ne else .1
            p = (c + self.strength * prior) / (n + self.strength)
            sd = np.sqrt(p * (1 - p) / (n + self.strength + 1))
            columns.extend([np.log1p(n), np.log1p(c), p, sd])
        p = columns[10]
        rank = np.array([np.mean(p < v) + .5 * np.mean(p == v) for v in p])
        return np.column_stack([*columns, rank, p - p.max()]).astype(np.float32)

    def observe(self, now, ids, clicked):
        if int(now) != self.now:
            raise ValueError("Advance/score before observing")
        # The task is binary click-per-candidate; repeated clicks in a request
        # must not make the numerator exceed its exposure denominator.
        ids, clicked = tuple(map(int, ids)), tuple(sorted(set(map(int, clicked))))
        if len(set(ids)) != len(ids) or not set(clicked) <= set(ids):
            raise ValueError("Invalid candidate/click set")
        self.pending.append((int(now), ids, clicked))


def hit1(labels, scores, groups):
    """Exact expected Top1 under uniform ties, including all-zero-label requests."""
    groups = np.asarray(groups, dtype=int)
    if groups.sum() != len(labels) or len(scores) != len(labels) or np.any(groups <= 0):
        raise ValueError("Invalid groups")
    starts = np.r_[0, np.cumsum(groups)[:-1]]
    maxima = np.maximum.reduceat(scores, starts)
    tied = scores == np.repeat(maxima, groups)
    return np.add.reduceat(np.asarray(labels) * tied, starts) / np.add.reduceat(tied, starts)

"""Reverse-curriculum frontier over audited handover points (#6); pure torch, no simulator.

Points of each teacher variant are ordered from earliest to latest prefix (nearest upright). The
frontier starts at the latest point. A handover start samples any point of its variant with
probability `uniform_share`; otherwise the frontier point with probability `frontier_share`, else a
point of the active window [frontier, latest]. Attempts resolved at the frontier update a
bias-corrected exponential average of success. With at least `min_attempts` there and an average of
at least `promote_at`, the frontier moves one point earlier and its counters restart; it never retreats.
"""
import torch


class ReverseSchedule:
    def __init__(self, counts, device, uniform_share=.1, frontier_share=.5, promote_at=.5, min_attempts=50, alpha=.02):
        self.device = device
        self.counts = torch.as_tensor(counts, dtype=torch.long, device=device)
        self.frontier = self.counts - 1
        self.ema = torch.zeros(len(self.counts), device=device)
        self.attempts = torch.zeros_like(self.frontier)
        self.successes = torch.zeros_like(self.frontier)
        self.promotions = torch.zeros_like(self.frontier)
        self.uniform_share, self.frontier_share = float(uniform_share), float(frontier_share)
        self.promote_at, self.min_attempts, self.alpha = float(promote_at), int(min_attempts), float(alpha)

    def sample(self, variants):
        """Point index within each given variant's ordered points."""
        n = len(variants)
        count, frontier = self.counts[variants], self.frontier[variants]
        any_point = torch.minimum((torch.rand(n, device=self.device) * count).long(), count - 1)
        window = torch.minimum(frontier + (torch.rand(n, device=self.device) * (count - frontier)).long(), count - 1)
        draw = torch.rand(n, device=self.device)
        to_frontier = self.uniform_share + (1 - self.uniform_share) * self.frontier_share
        return torch.where(draw < self.uniform_share, any_point, torch.where(draw < to_frontier, frontier, window))

    def rolling_success(self):
        weight = 1 - (1 - self.alpha) ** self.attempts.float()
        return torch.where(self.attempts > 0, self.ema / weight.clamp_min(1e-12), torch.zeros_like(self.ema))

    def record(self, variants, points, success):
        """Resolved attempts; only those started at their variant's current frontier count."""
        at_frontier = points == self.frontier[variants]
        rows = variants[at_frontier]
        outcome = success[at_frontier].float()
        tries = torch.zeros_like(self.ema).scatter_add_(0, rows, torch.ones_like(outcome))
        wins = torch.zeros_like(self.ema).scatter_add_(0, rows, outcome)
        # Treat this step's outcomes as one batch at their mean: exact for the bias-corrected average.
        decay = (1 - self.alpha) ** tries
        self.ema = self.ema * decay + (wins / tries.clamp_min(1)) * (1 - decay)
        self.attempts += tries.long()
        self.successes += wins.long()
        promote = (self.attempts >= self.min_attempts) & (self.rolling_success() >= self.promote_at) & (self.frontier > 0)
        self.frontier -= promote.long()
        self.promotions += promote.long()
        for state in (self.ema, self.attempts, self.successes):
            state.masked_fill_(promote, 0)
        return promote

    def state(self):
        return {name: getattr(self, name).cpu().tolist() for name in ('counts', 'frontier', 'ema', 'attempts', 'successes', 'promotions')}

    def load(self, state):
        if state['counts'] != self.counts.cpu().tolist():
            raise ValueError('Schedule state belongs to a different set of handover points')
        for name in ('frontier', 'ema', 'attempts', 'successes', 'promotions'):
            current = getattr(self, name)
            current.copy_(torch.tensor(state[name], dtype=current.dtype, device=self.device))

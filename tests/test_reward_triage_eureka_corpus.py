"""Typical EUREKA-style snippets must not NameError on harmless builtins."""

from __future__ import annotations

from reward_triage.executor import run_candidate_on_probes
from reward_triage.probes import make_synthetic_probe_batch

# 20 typical patterns using round/pow/any/all/sorted/divmod/reversed/classes.
CORPUS = [
    """
def compute_reward(transition):
    return round(float(transition.get('progress', 0.0)), 3)
""",
    """
def compute_reward(transition):
    return pow(float(transition.get('progress', 0.0)), 2)
""",
    """
def compute_reward(transition):
    xs = [float(transition.get('progress', 0.0)), float(transition.get('clearance', 0.0))]
    return 1.0 if any(x > 0 for x in xs) else 0.0
""",
    """
def compute_reward(transition):
    xs = [float(transition.get('collision', False)), False]
    return 1.0 if all(not x for x in xs) else -1.0
""",
    """
def compute_reward(transition):
    vals = [float(transition.get('progress', 0.0)), float(transition.get('clearance', 0.0))]
    return float(sorted(vals)[-1])
""",
    """
def compute_reward(transition):
    q, r = divmod(int(abs(float(transition.get('progress', 0.0)) * 10)), 3)
    return float(q + r)
""",
    """
def compute_reward(transition):
    seq = [1.0, 2.0, float(transition.get('progress', 0.0))]
    return float(sum(reversed(seq)))
""",
    """
class Shaper:
    def __init__(self, gamma=0.9):
        self.gamma = gamma
    def __call__(self, transition):
        return self.gamma * float(transition.get('progress', 0.0))

_shaper = Shaper()
def compute_reward(transition):
    return float(_shaper(transition))
""",
    """
def compute_reward(transition):
    p = float(transition.get('progress', 0.0))
    return abs(p) + min(1.0, max(-1.0, p))
""",
    """
import math
def compute_reward(transition):
    return float(math.tanh(float(transition.get('progress', 0.0))))
""",
    """
import numpy as np
def compute_reward(transition):
    a = np.asarray(transition.get('action', [0.0, 0.0]), dtype=float)
    return float(np.sum(a * a))
""",
    """
import numpy as np
def compute_reward(transition):
    d = float(transition.get('goal_distance', 0.0))
    return float(np.exp(-d) - np.exp(-float(transition.get('prev_goal_distance', d))))
""",
    """
def compute_reward(transition):
    c = 1.0 if transition.get('collision') else 0.0
    s = 1.0 if transition.get('success') else 0.0
    return s - c
""",
    """
def compute_reward(transition):
    p = float(transition.get('progress', 0.0))
    return (p, {'progress': p})
""",
    """
def compute_reward(transition):
    vals = [float(transition.get('progress', 0.0)) for _ in range(3)]
    return float(sum(vals) / len(vals))
""",
    """
def compute_reward(transition):
    a = list(transition.get('action', [0.0, 0.0]))
    return float(sum(map(abs, a)))
""",
    """
def compute_reward(transition):
    items = [('p', float(transition.get('progress', 0.0))), ('c', float(transition.get('clearance', 0.0)))]
    return float(dict(items)['p'])
""",
    """
def compute_reward(transition):
    x = float(transition.get('progress', 0.0))
    try:
        return x / (abs(x) + 1.0)
    except ZeroDivisionError:
        return 0.0
""",
    """
def compute_reward(transition):
    return float(isinstance(transition, dict))
""",
    """
def compute_reward(transition):
    keys = sorted(str(k) for k in transition)
    return float(len(keys))
""",
]


def test_eureka_style_corpus_no_nameerror():
    tr = make_synthetic_probe_batch("navigation_2d", n_per_policy=1, seed=1).transitions[:3]
    assert len(CORPUS) == 20
    for i, code in enumerate(CORPUS):
        result = run_candidate_on_probes(code, tr, timeout_s=10.0, in_process=True)
        assert result.report.status == "ok", f"snippet {i}: {result.report.reasons}"
        assert len(result.rewards) == 3

"""Model-free tests of the decision logic: hand-over sequences, voting, decision-mode parsing, refinement."""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import handover  # noqa: E402
from pipeline.analyze import refine_times  # noqa: E402
from pipeline.rules import Check, decide, plan, votes  # noqa: E402
from pipeline.timeline import Event, StateSample  # noqa: E402
from pipeline.vlm import Answer  # noqa: E402
from pipeline.vlm_backends import normalize, option_token_ids  # noqa: E402


def frames(seq):
    """seq: list of letters (dominant option per 0.5-s frame) -> times, probs."""
    times = [i * 0.5 for i in range(len(seq))]
    probs = [{o: (0.9 if o == s else 0.02) for o in handover.OPTIONS} for s in seq]
    return times, probs


def test_id_then_cash_is_clean():
    t, p = frames(list("FFAAEEFFCCC"))
    r = handover.decide_episode(t, p)
    assert r.kind is None and r.id_seen_at == 1.0 and r.first_cash == 4.0


def test_phone_digital_id_counts_as_id():
    t, p = frames(list("FFBBFFCCC"))
    assert handover.decide_episode(t, p).kind is None


def test_cash_without_id_is_v05_with_handover_timecode():
    t, p = frames(list("FFFFCCCC"))
    r = handover.decide_episode(t, p)
    assert r.kind == "client_no_id" and r.t == 2.0


def test_bag_without_documents_is_v13():
    t, p = frames(list("FFDDDFF"))
    r = handover.decide_episode(t, p)
    assert r.kind == "collector_no_id" and r.t == 1.0


def test_single_flicker_is_not_a_handover():
    t, p = frames(list("FFCFFFF"))
    assert handover.decide_episode(t, p).kind is None


def test_id_after_money_does_not_excuse():
    t, p = frames(list("FCCCFAAA"))
    assert handover.decide_episode(t, p).kind == "client_no_id"


def test_votes_two_of_three_or_mean():
    c = Check("money_left_unattended", "money_on_desk", 10.0, [1, 2, 3], min_yes=2)
    c.answers = [Answer("yes", "", "", 0.9), Answer("no", "", "", 0.1), Answer("yes", "", "", 0.6)]
    assert votes(c)[0]
    c.answers = [Answer("no", "", "", 0.35), Answer("no", "", "", 0.38), Answer("no", "", "", 0.36)]
    assert not votes(c)[0]  # mean 0.36 < 0.40 threshold of money_on_desk


def test_handover_rule_fires_once_per_matching_kind():
    ev = [Event("visitor_arrived", 2.0), Event("visitor_left", 20.0, info={"occupancy": 1})]
    checks = [c for c in plan(ev, 30.0) if c.question == "handover_sequence"]
    assert {c.rule for c in checks} == {"client_no_id", "collector_no_id"}
    assert checks[0].times[0] == 2.0 and checks[0].times[-1] <= 20.0
    t, p = frames(list("FFFFCCCC"))
    res = handover.decide_episode(t, p)
    for c in checks:
        c.answers = [res]
    v = [x for x in decide(checks) if x.type_id in ("client_no_id", "collector_no_id")]
    assert [x.type_id for x in v] == ["client_no_id"] and v[0].code == "V05"


def test_normalize_and_token_variants():
    class Tok:
        vocab = {"yes": 1, "Yes": 2, "no": 3, "No": 4, " yes": 5, " Yes": 6, " no": 7, " No": 8, "YES": 9, "NO": 10}

        def encode(self, s, add_special_tokens=False):
            return [self.vocab[s]] if s in self.vocab else [99]

    ids = option_token_ids(Tok(), ["yes", "no"])
    assert ids[1] == "yes" and ids[4] == "no"
    p = normalize({1: math.log(0.6), 2: math.log(0.2), 3: math.log(0.1)}, ids, ["yes", "no"])
    assert abs(p["yes"] - 0.8 / 0.9) < 1e-6


def test_refine_only_where_state_changes():
    s = [StateSample(0.5, 1, True, False), StateSample(2.0, 1, True, False), StateSample(3.5, 0, False, False)]
    assert refine_times(s, {0.5, 2.0, 3.5}) == [2.5, 3.0]

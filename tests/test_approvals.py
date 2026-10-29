from datetime import date

import pytest

from inventory_ai.approvals import ApprovalError, decide
from inventory_ai.models import Recommendation


def rec(session):
    r = Recommendation(run_date=date.today(), kind="purchase", item_code="X", qty=10, value=100, reason="r")
    session.add(r)
    session.flush()
    return r


@pytest.mark.parametrize("who", ["", "  ", "system", "agent", "Claude"])
def test_only_a_named_person_can_decide(session, who):
    with pytest.raises(ApprovalError):
        decide(session, rec(session).id, "approve", who)


def test_approve_with_changed_quantity_scales_value(session):
    r = decide(session, rec(session).id, "approve", "Fatima", qty=4)
    assert (r.status, r.qty, r.value, r.decided_by) == ("approved", 4, 40, "Fatima")


def test_cannot_decide_twice(session):
    r = rec(session)
    decide(session, r.id, "reject", "Omar", note="too early")
    with pytest.raises(ApprovalError, match="already rejected"):
        decide(session, r.id, "approve", "Omar")

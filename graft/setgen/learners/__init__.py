"""The nine evaluated arms of decision 1, and nothing else.

**Fix F6, asserted rather than intended.** No module in this package imports
``StateGraph``, ``LatticeInstance`` or an atom id — exit criterion 6 is a test,
not a convention. Every loss here reads a :class:`~graft.setgen.trainer.Batch`
of padded tensors indexed by position in a trajectory; where those tensors came
from is the trainer's business and the featurizer's, and at Phase 9 it will be
the Stage-B graph encoder instead with nothing in this package changing.

======  =========================  ==========================================
Arm     Objective                  Role
======  =========================  ==========================================
L1      supervised stepwise        **[EVIDENCE]** Graph-S3 (ACL 2026) makes
                                   stepwise supervised graph retrieval a
                                   strong baseline
L2      canonical set imitation    does distribution training add anything
                                   over imitating one gold set?
L3      GRPO                       policy-gradient control, matched policy
L4      Trajectory Balance         **[EVIDENCE]** NeurIPS 2022
L5      SubTB(λ)                   **[EVIDENCE]** ICML 2023
L6      LED-DB                     **[EVIDENCE]** LED-GFN; the control C3 must
                                   beat
L7      checker-conditioned LED    **the proposed method** — L6 **plus ``Δd``
                                   as input features and nothing else**
L7b     L7 + forward-looking head  ablation (fix F11, decision 26)
GAFN    augmented TB               **[EVIDENCE]** GAFlowNets (ICLR 2023); the
                                   second required control (plan §4.5.4)
======  =========================  ==========================================

**L6 and L7 differ by one boolean and by nothing else.** They share this file's
``led_db_loss`` verbatim; the difference is ``delta_d=True`` on L7's featurizer,
which fills the ``Δd`` block of ``action_repr`` that L6 leaves zeroed. Because
the block is *present and zeroed* rather than absent, the two arms have byte-for-
byte identical parameter shapes and their capacity match is exact rather than
within 1% — which is the strongest form decision 11 can take.
"""

from __future__ import annotations

from graft.setgen.learners.gaflownet import augmented_tb_loss
from graft.setgen.learners.l1_supervised import supervised_loss
from graft.setgen.learners.l2_imitation import imitation_loss
from graft.setgen.learners.l3_grpo import grpo_loss
from graft.setgen.learners.l4_tb import tb_loss
from graft.setgen.learners.l5_subtb import subtb_loss
from graft.setgen.learners.l6_led import decomposition_loss, led_db_loss
from graft.setgen.learners.l7_checker_led import checker_led_loss
from graft.setgen.learners.l7b_aux import checker_led_aux_loss

__all__ = [
    "supervised_loss",
    "imitation_loss",
    "grpo_loss",
    "tb_loss",
    "subtb_loss",
    "led_db_loss",
    "decomposition_loss",
    "checker_led_loss",
    "checker_led_aux_loss",
    "augmented_tb_loss",
    "ARMS",
    "build_arm",
    "FLOW_FAMILY",
    "SUPERVISED_FAMILY",
]

#: The flow family, for which exact TV is the **primary** metric (decision 12).
FLOW_FAMILY: tuple[str, ...] = (
    "l4_tb", "l5_subtb", "l6_led", "l7_checker_led", "l7b_aux", "gaflownet",
)

#: L1–L3, reported in a **separate** table with TV descriptive and labelled as
#: such (G2, decision 12).  One table implying three methods failed at a task
#: two of them never attempted is the specific error this split prevents.
SUPERVISED_FAMILY: tuple[str, ...] = ("l1_supervised", "l2_imitation", "l3_grpo")


def build_arm(name: str, *, hidden: int | None = None):
    """The one place an arm's switches are set (decision 19a's routing).

    Constructing an ``Arm`` by hand anywhere else would make ``delta_d`` a thing
    a caller could flip, and the whole L6/L7 comparison rests on it not being.
    """
    from graft.setgen.trainer import Arm

    if name not in ARMS:
        raise ValueError(f"unknown arm {name!r}; expected one of {sorted(ARMS)}")
    kwargs = dict(ARMS[name])
    loss = kwargs.pop("loss")
    return Arm(name, loss, hidden=hidden, **kwargs)


#: Decision 19a's routing table, in one place so it can be read as a table.
#: ``delta_d`` policy-visible: **L7 and L7b only**.  GAFlowNet is loss-only.
#: Everything else: neither.
ARMS: dict[str, dict] = {
    "l1_supervised": {"loss": supervised_loss, "supervised": True},
    "l2_imitation": {"loss": imitation_loss, "supervised": True},
    "l3_grpo": {"loss": grpo_loss},
    "l4_tb": {"loss": tb_loss},
    "l5_subtb": {"loss": subtb_loss, "needs_flow": True},
    "l6_led": {
        "loss": led_db_loss,
        "needs_flow": True,
        "needs_potential": True,
        "trains_potential": True,
    },
    "l7_checker_led": {
        "loss": checker_led_loss,
        "delta_d": True,
        "needs_flow": True,
        "needs_potential": True,
        "trains_potential": True,
    },
    "l7b_aux": {
        "loss": checker_led_aux_loss,
        "delta_d": True,
        "needs_flow": True,
        "needs_potential": True,
        "trains_potential": True,
        "needs_aux": True,
    },
    "gaflownet": {"loss": augmented_tb_loss, "needs_flow": True},
}

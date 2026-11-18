"""Cost-based decision threshold for loan approval.

Decision rule: reject the application when the predicted default probability is >= threshold.
- Approving a defaulter (false negative) costs LGD x AMT_CREDIT.
- Rejecting a good customer (false positive) costs MARGIN x AMT_CREDIT (lost profit).
- Correct decisions cost 0.
LGD and MARGIN are assumptions, not values provided with the data. Amounts are in the data's (undisclosed)
currency.
"""
import numpy as np
import pandas as pd

BASE_LGD = 0.60
BASE_MARGIN = 0.15
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def threshold_sweep(y, prob, credit, thresholds=THRESHOLDS, lgd=BASE_LGD, margin=BASE_MARGIN):
    """Total cost, its two components, approval rate and default rate among approved for each threshold."""
    y = np.asarray(y)
    prob = np.asarray(prob)
    credit = np.asarray(credit, dtype="float64")
    rows = []
    for t in thresholds:
        approve = prob < t
        fn = approve & (y == 1)
        fp = ~approve & (y == 0)
        fn_cost = lgd * credit[fn].sum()
        fp_cost = margin * credit[fp].sum()
        rows.append({
            "threshold": t,
            "total_cost": fn_cost + fp_cost,
            "fn_cost": fn_cost,
            "fp_cost": fp_cost,
            "approval_rate": approve.mean(),
            "default_rate_approved": y[approve].mean() if approve.any() else np.nan,
            "n_fn": int(fn.sum()),
            "n_fp": int(fp.sum()),
        })
    return pd.DataFrame(rows)


def approve_all_cost(y, credit, lgd=BASE_LGD):
    """Cost of approving every applicant: every defaulter's loss."""
    return float(lgd * np.asarray(credit, dtype="float64")[np.asarray(y) == 1].sum())


def optimal_threshold(sweep):
    """Threshold with the lowest total cost (first one if tied)."""
    return float(sweep.loc[sweep["total_cost"].idxmin(), "threshold"])


def decision_summary(y, prob, credit, threshold, lgd=BASE_LGD, margin=BASE_MARGIN):
    """One row of the threshold sweep for a single threshold."""
    return threshold_sweep(y, prob, credit, [threshold], lgd, margin).iloc[0]


def cost_components(y, prob, credit, thresholds=THRESHOLDS):
    """Credit amount of approved defaulters (FN) and of rejected good applicants (FP) at each threshold.
    Total cost at (LGD, MARGIN) is LGD * fn_credit + MARGIN * fp_credit, so one pass serves every cost scenario."""
    y = np.asarray(y)
    prob = np.asarray(prob)
    credit = np.asarray(credit, dtype="float64")
    fn_credit = np.array([credit[(prob < t) & (y == 1)].sum() for t in thresholds])
    fp_credit = np.array([credit[(prob >= t) & (y == 0)].sum() for t in thresholds])
    return fn_credit, fp_credit


def optimal_threshold_table(y, prob, credit, lgds, margins, thresholds=THRESHOLDS):
    """Cost-optimal threshold for every (LGD, MARGIN) pair; same tie rule as `optimal_threshold` (first minimum)."""
    fn_credit, fp_credit = cost_components(y, prob, credit, thresholds)
    rows = []
    for lgd in lgds:
        for margin in margins:
            cost = lgd * fn_credit + margin * fp_credit
            rows.append({"LGD": lgd, "MARGIN": margin, "optimal_threshold": float(thresholds[int(np.argmin(cost))])})
    return pd.DataFrame(rows)


def sensitivity_grid(y_select, p_select, credit_select, y_eval, p_eval, credit_eval,
                     lgds=(0.4, 0.6, 0.8), margins=(0.10, 0.15, 0.20)):
    """For each (LGD, MARGIN): choose the threshold on the selection data, then report it on the evaluation data."""
    rows = []
    for lgd in lgds:
        for margin in margins:
            t = optimal_threshold(threshold_sweep(y_select, p_select, credit_select, lgd=lgd, margin=margin))
            at_t = decision_summary(y_eval, p_eval, credit_eval, t, lgd, margin)
            base = approve_all_cost(y_eval, credit_eval, lgd)
            rows.append({"LGD": lgd, "MARGIN": margin, "optimal_threshold": t,
                         "approval_rate": at_t["approval_rate"],
                         "default_rate_approved": at_t["default_rate_approved"],
                         "cost_at_threshold": at_t["total_cost"], "cost_approve_all": base,
                         "saving_vs_approve_all": 1 - at_t["total_cost"] / base})
    return pd.DataFrame(rows)

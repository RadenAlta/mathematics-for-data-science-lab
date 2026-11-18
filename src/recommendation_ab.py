"""Independent recommendation experiment: Welch's two-sided t-test."""
from __future__ import annotations
import numpy as np
from scipy.stats import t as student_t


def simulate_article_reads(n_control: int = 2058, n_treatment: int = 2120,
                           seed: int = 2026):
    rng = np.random.default_rng(seed)
    # Count data modelled with Poisson distributions for demonstration.
    return rng.poisson(4.72, n_control), rng.poisson(5.38, n_treatment)


def welch_test(control, treatment):
    """Welch's unequal-variance t-test; formula implemented explicitly."""
    a, b = np.asarray(control, dtype=float), np.asarray(treatment, dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError('Each group needs at least two observations.')
    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    va, vb = var_a/len(a), var_b/len(b)
    t_stat = float((mean_a - mean_b)/np.sqrt(va+vb))
    df = float((va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1)))
    pvalue = float(2*student_t.sf(abs(t_stat), df=df))
    return dict(n_control=len(a), n_treatment=len(b), control_mean=float(mean_a),
                treatment_mean=float(mean_b), absolute_lift=float(mean_b-mean_a),
                relative_lift=float(mean_b/mean_a-1), t_statistic=t_stat,
                degrees_of_freedom=df, p_value=pvalue, reject_null=pvalue < .05)


def evaluate(seed: int = 2026):
    groups = simulate_article_reads(seed=seed)
    return dict(**welch_test(*groups), groups=groups)

if __name__ == '__main__':
    r = evaluate()
    print(f"Recommendation AB | control={r['control_mean']:.3f} | treatment={r['treatment_mean']:.3f} | "
          f"t={r['t_statistic']:.3f} | p={r['p_value']:.4g}")

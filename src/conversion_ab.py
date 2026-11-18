"""Independent conversion A/B demonstration using aggregated, synthetic counts."""
from __future__ import annotations
import math
from scipy.stats import norm

# Demonstration counts aligned with the completed course's reported synthetic example.
CONTROL_USERS, CONTROL_CONVERSIONS = 3959, 2734
TREATMENT_USERS, TREATMENT_CONVERSIONS = 4041, 2906


def two_proportion_z_test(conversions_a: int, users_a: int,
                          conversions_b: int, users_b: int) -> dict:
    if not (0 <= conversions_a <= users_a and 0 <= conversions_b <= users_b and users_a > 0 and users_b > 0):
        raise ValueError('Valid nonnegative conversions and positive sample sizes are required.')
    pa, pb = conversions_a/users_a, conversions_b/users_b
    pooled = (conversions_a+conversions_b)/(users_a+users_b)
    se = math.sqrt(pooled*(1-pooled)*(1/users_a+1/users_b))
    if se == 0: raise ValueError('Pooled standard error is zero.')
    z_stat = (pa-pb)/se
    pvalue = float(2*norm.sf(abs(z_stat)))
    return dict(control_users=users_a, treatment_users=users_b, control_conversions=conversions_a,
                treatment_conversions=conversions_b, control_rate=pa, treatment_rate=pb,
                pooled_rate=pooled, absolute_lift=pb-pa,
                relative_lift=pb/pa-1 if pa else None, z_statistic=z_stat,
                p_value=pvalue, reject_null=pvalue < .05)


def evaluate():
    return two_proportion_z_test(CONTROL_CONVERSIONS,CONTROL_USERS,
                                 TREATMENT_CONVERSIONS,TREATMENT_USERS)

if __name__ == '__main__':
    r=evaluate()
    print(f"Conversion AB | {r['control_rate']:.4%} → {r['treatment_rate']:.4%} "
          f"| z={r['z_statistic']:.4f} | p={r['p_value']:.5f}")

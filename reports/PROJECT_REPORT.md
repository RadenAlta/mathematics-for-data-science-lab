# Project report: Applied Mathematics for Data Science

## Scope and provenance

This project presents three independently recreated statistical examples based on concepts studied in Dicoding Academy 831. The public portfolio repository intentionally omits graded exercise files and multiple-choice answer keys. All input datasets are synthetic. No real personal data, client records, or operational business metrics are used.

The **course outcome** (Advanced, 4.0/4.0; 15/15 Knowledge Check) is taken from the actual reviewer feedback reported by the learner. The **recreated portfolio experiment metrics** below are calculated by executing the independent source code in this repository. The two must not be conflated.

## 1 — Bird classification

**Problem:** classify three synthetic bird species using measurements whose class-conditional distributions differ.

- **Synthetic sample:** 960 observations (320 per species), 75/25 random train/test split.
- **Predictors:** wingspan, body mass, beak-to-head ratio, days spent singing in a 30-day window.
- **Likelihood model:** Gaussian for wingspan and mass, bounded density for beak ratio, binomial for singing days.
- **Estimation:** Gaussian sample mean and standard deviation, empirical bounds with a small margin, observed singing-day frequency.
- **Computation:** log priors and log likelihoods, followed by argmax.
- **Held-out accuracy:** 1.000 on the **artificially well-separated** example.

**Interpretation:** an illustrative demonstration of the mathematics, **not** a validation for live species recognition. A robust field study needs real data, separation by source/time where relevant, cross-validation and calibration. The uniform support fallback is a deliberately simplified feature model.

## 2 — Recommendation experiment

**Problem:** compare independent control and treatment groups for article-reading behaviour.

- **Simulation:** Poisson counts, group sizes 2,058 and 2,120, deterministic RNG seed 2026.
- **Test:** Welch's unequal-variance, two-sided t-test.
- **Control mean:** 4.8168 articles per person.
- **Treatment mean:** 5.2835 articles per person.
- **t-statistic:** approximately −6.7039.
- **p-value:** approximately 2.30e−11.

**Interpretation:** simulated data support a difference in expected article reading at the 5% level. This does not prove that a real recommendation system causes the improvement or that the improvement is economically valuable.

## 3 — Conversion experiment

**Problem:** compare a baseline software feature against a proposed variant for a binary conversion outcome.

- **Baseline:** 2,734 conversions / 3,959 users (69.06%).
- **Variant:** 2,906 conversions / 4,041 users (71.91%).
- **Pooled conversion rate:** 70.50%.
- **Two-sided z-test:** z = −2.7996; p ≈ 0.005116.
- **Absolute difference:** approximately +2.855 percentage points for treatment vs control.

**Interpretation:** the illustrative proportion difference is significant by a conventional preselected 5% threshold. Do not infer universal product superiority: evaluate randomisation, guardrails, confounding, practical value, uncertainty, and multiple comparisons before a real rollout.

## Testing and reproducibility

The repo provides five `unittest` checks spanning deterministic generation, basic classifier accuracy, Welch's null comparison, z-test directionality and symmetry. GitHub Actions runs these checks on pushes and pull requests. All three notebooks are saved with executed outputs.

## Technical improvements and reviewer-inspired learning

- Use log-probabilities in Naive Bayes to mitigate numerical underflow.
- Avoid early stopping based solely on repeated p-value inspection during A/B experiments.
- Welch's t-test works without requiring equal group variances; its degrees of freedom use the Satterthwaite approximation.
- Prefer clear condition dispatch (`match-case` when appropriate), while retaining simple expressions for short logic.

## Ethics and limitations

This work should not be treated as an answer-key repository or used to copy a graded submission. Synthetic data carry no guarantee of fairness, distributional realism, robustness, or external validity. Do not use these scripts for high-stakes decisions without an independent methodology and validation review.

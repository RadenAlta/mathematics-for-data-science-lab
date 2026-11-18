"""Build executable, portfolio-original notebooks (no Dicoding graded template)."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
NOTEBOOKS=ROOT/'notebooks'
NOTEBOOKS.mkdir(exist_ok=True)

header='''from pathlib import Path
import sys
root = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from IPython.display import Image, display
'''

cases = [
('01-bird-species-naive-bayes.ipynb', [
('md','''# 01 · Bird Species Classification with Naive Bayes

**Portfolio recreation · synthetic data · reproducible**

This notebook explores distribution-aware **Gaussian + Uniform + Binomial** features and a Naive Bayes model built with log-likelihoods. This is an independent portfolio example, not the original Dicoding assessment template.

**Question:** Can feature distributions help distinguish synthetic bird species?'''),
('code',header+'''from src.bird_naive_bayes import generate_bird_data, train_test_split, fit_model, predict, evaluate
from src.visualize import save_bird_chart
import pandas as pd
import numpy as np
'''),
('md','## 1 · Inspect the synthetic sample\n\nEvery row is simulated and does not represent field observations.'),
('code','''result = evaluate()
print('Sample size:', len(result['data']))
print('Train / test:', len(result['train']), '/', len(result['test']))
print('Species:', ', '.join(sorted(result['model'])))
display(result['data'].head())'''),
('md','## 2 · Learn the feature distributions\n\nWingspan and mass use Gaussian likelihoods; beak-to-head ratio uses a bounded density; the number of singing days uses a binomial distribution.'),
('code','''for species, params in result['model'].items():
    print(species, '| prior=', round(params['prior'], 3), '| singing probability=', round(float(params['sing_prob']),3))'''),
('md','## 3 · Evaluation\n\nPredictions use **log-probabilities** to avoid numerical underflow. A hold-out split is used for a transparent, reproducible accuracy figure.'),
('code','''print('Hold-out accuracy:', f"{result['accuracy']:.1%}")
display(result['confusion'])
chart_path, _ = save_bird_chart()
display(Image(filename=str(chart_path)))'''),
('md','''## Takeaways and limitations

- Density assumptions make model behaviour easier to explain than an opaque classifier.
- On this deliberately separable synthetic dataset, performance is high. **This is not evidence of accuracy on real birds.**
- More realistic experiments should use field-collected data, stratified splits, cross-validation, calibration, and checks for dataset shift.
- This portfolio variant uses log-probabilities (an enhancement suggested by the course reviewer).'''),
]),
('02-recommendation-ab-testing.ipynb',[
('md','''# 02 · Recommendation Experiment with Welch's t-test

**Portfolio recreation · synthetic data · reproducible**

**Question:** Does a simulated content recommendation feature increase mean articles read per user? We compare independent groups with Welch's unequal-variance t-test.'''),
('code',header+'''from src.recommendation_ab import simulate_article_reads, welch_test, evaluate
from src.visualize import save_reco_chart
'''),
('md','## 1 · Create the experiment'),
('code','''result = evaluate()
print(f"Control users: {result['n_control']:,}")
print(f"Treatment users: {result['n_treatment']:,}")
print(f"Mean reads (A): {result['control_mean']:.3f}")
print(f"Mean reads (B): {result['treatment_mean']:.3f}")
print(f"Relative uplift: {result['relative_lift']:.2%}")'''),
('md','## 2 · Hypothesis test\n\nThe null is equal population means. Use a **two-sided** hypothesis, sample variance estimates, Welch–Satterthwaite degrees of freedom, and the Student t distribution.'),
('code','''print(f"Welch t-statistic: {result['t_statistic']:.4f}")
print(f"Degrees of freedom: {result['degrees_of_freedom']:.2f}")
print(f"Two-sided p-value: {result['p_value']:.8g}")
print('Reject H0 at alpha=0.05:', result['reject_null'])
chart_path, _ = save_reco_chart()
display(Image(filename=str(chart_path)))'''),
('md','''## Takeaways and limitations

- An observed increase in means is not sufficient by itself; the test quantifies evidence against equal means.
- The data are independent **simulated count observations**, not a production experiment.
- A real A/B test should predefine metrics and analysis, randomise units, consider variance and practical effect size, and avoid peeking or stopping early based on p-values alone.'''),
]),
('03-conversion-ab-testing.ipynb',[
('md','''# 03 · Conversion Experiment with a Two-Proportion Z-Test

**Portfolio recreation · aggregated synthetic counts**

**Question:** Is the conversion rate different between a baseline and a new software feature? The two-proportion z-test is appropriate for a binary outcome when the normal approximation is adequate.'''),
('code',header+'''from src.conversion_ab import two_proportion_z_test, evaluate
from src.visualize import save_conversion_chart
'''),
('md','## 1 · View group metrics'),
('code','''r = evaluate()
print(f"Control:   {r['control_conversions']:,}/{r['control_users']:,} = {r['control_rate']:.2%}")
print(f"Treatment: {r['treatment_conversions']:,}/{r['treatment_users']:,} = {r['treatment_rate']:.2%}")
print(f"Absolute uplift: {r['absolute_lift']*100:.2f} percentage points")
print(f"Relative uplift: {r['relative_lift']:.2%}")'''),
('md','## 2 · Z-test with pooled proportion\n\nThe z-score is computed as the difference in proportions (A − B) divided by the pooled standard error. A negative z-score **does not** imply universal superiority of B.'),
('code','''print(f"Pooled conversion rate: {r['pooled_rate']:.4f}")
print(f"Z-statistic (A - B): {r['z_statistic']:.4f}")
print(f"Two-sided p-value: {r['p_value']:.6f}")
print('Reject H0 at alpha=0.05:', r['reject_null'])
chart_path, _ = save_conversion_chart()
display(Image(filename=str(chart_path)))'''),
('md','''## Takeaways and limitations

- The synthetic example indicates a statistically detectable difference at alpha=0.05.
- Significance alone cannot establish causal validity, business value, or universal product benefit.
- A production rollout would need randomisation checks, guardrail metrics, minimum sample planning, multiple-testing awareness, and consideration of confidence intervals and business impact.'''),
])]

for filename, steps in cases:
    cells=[]
    for kind,source in steps:
        cells.append(nbf.v4.new_markdown_cell(source) if kind=='md' else nbf.v4.new_code_cell(source))
    book=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'}, 'language_info':{'name':'python'}})
    client=NotebookClient(book,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
    executed=client.execute()
    target=NOTEBOOKS/filename
    nbf.write(executed,target)
    print('Executed',target.name,'cells',len(cells),'errors',sum(1 for c in executed.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error'))

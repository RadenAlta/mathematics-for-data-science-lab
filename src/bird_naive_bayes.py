"""Independent, reproducible bird-species Naive Bayes portfolio demonstration.

This is an original portfolio reimplementation, not the graded course template.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm, binom

FEATURES = ('wingspan_cm', 'weight_g', 'beak_head_ratio', 'sing_days')
SPECIES = {
    'Finch': {'wing': (20, 2.3), 'weight': (18, 2.1), 'beak': (.20, .37), 'sing_prob': .64},
    'Sparrow': {'wing': (25, 2.7), 'weight': (25, 2.8), 'beak': (.37, .53), 'sing_prob': .43},
    'Swallow': {'wing': (34, 3.0), 'weight': (37, 3.0), 'beak': (.54, .70), 'sing_prob': .78},
}


def generate_bird_data(n_per_species: int = 320, seed: int = 2026) -> pd.DataFrame:
    """Generate synthetic data. Nothing here represents observations of real animals."""
    rng = np.random.default_rng(seed)
    frames = []
    for species, cfg in SPECIES.items():
        wing_mu, wing_sigma = cfg['wing']
        weight_mu, weight_sigma = cfg['weight']
        low, high = cfg['beak']
        frames.append(pd.DataFrame({
            'species': species,
            'wingspan_cm': rng.normal(wing_mu, wing_sigma, n_per_species),
            'weight_g': rng.normal(weight_mu, weight_sigma, n_per_species),
            'beak_head_ratio': rng.uniform(low, high, n_per_species),
            'sing_days': rng.binomial(30, cfg['sing_prob'], n_per_species),
        }))
    return pd.concat(frames, ignore_index=True)


def train_test_split(data: pd.DataFrame, test_fraction: float = .25, seed: int = 44):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data))
    cut = int(len(data) * (1 - test_fraction))
    return data.iloc[idx[:cut]].reset_index(drop=True), data.iloc[idx[cut:]].reset_index(drop=True)


def fit_model(train: pd.DataFrame) -> dict:
    """Estimate priors and class-conditional distributions from the training data."""
    result = {}
    for species, frame in train.groupby('species'):
        ratio = frame['beak_head_ratio'].to_numpy()
        result[species] = {
            'prior': len(frame)/len(train),
            'gaussian': {
                col: (float(frame[col].mean()), max(float(frame[col].std(ddof=0)), 1e-8))
                for col in ('wingspan_cm', 'weight_g')
            },
            # A small support margin reduces accidental impossible likelihoods.
            'beak': (max(0., float(ratio.min()) - .03), min(1., float(ratio.max()) + .03)),
            'sing_prob': np.clip(float(frame['sing_days'].mean()/30), 1e-7, 1-1e-7),
        }
    return result


def predict(model: dict, observations: pd.DataFrame) -> np.ndarray:
    """Use log-likelihoods to avoid numerical underflow from tiny products."""
    scores = []
    for species, p in model.items():
        score = np.full(len(observations), np.log(p['prior']), dtype=float)
        for col, (mu, sigma) in p['gaussian'].items():
            score += norm.logpdf(observations[col], loc=mu, scale=sigma)
        low, high = p['beak']
        beak = observations['beak_head_ratio'].to_numpy()
        score += np.where((beak >= low) & (beak <= high), -np.log(high-low), -35.0)
        score += binom.logpmf(observations['sing_days'], 30, p['sing_prob'])
        scores.append(score)
    names = list(model)
    return np.array(names)[np.argmax(np.column_stack(scores), axis=1)]


def evaluate(seed: int = 2026) -> dict:
    data = generate_bird_data(seed=seed)
    train, test = train_test_split(data)
    model = fit_model(train)
    pred = predict(model, test)
    labels = sorted(model)
    confusion = pd.crosstab(pd.Categorical(test['species'], labels),
                            pd.Categorical(pred, labels), rownames=['actual'], colnames=['predicted'], dropna=False)
    return {'data': data, 'train': train, 'test': test, 'predictions': pred,
            'accuracy': float(np.mean(pred == test['species'].to_numpy())),
            'confusion': confusion, 'model': model}

if __name__ == '__main__':
    result = evaluate()
    print(f"Bird Naive Bayes | synthetic samples={len(result['data'])} | test accuracy={result['accuracy']:.3f}")
    print(result['confusion'].to_string())

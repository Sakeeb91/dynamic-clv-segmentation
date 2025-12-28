# Implementation Plan: Dynamic CLV Segmentation

**Expert Role:** ML Engineer specializing in Probabilistic Models & Customer Analytics

**Rationale:** This project requires deep understanding of probabilistic customer behavior models (BG/NBD family), Gaussian Mixture Models for unsupervised segmentation, stacked generalization techniques, online learning paradigms, and statistical inference for uncertainty quantification. This sits at the intersection of applied ML and marketing science.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DYNAMIC CLV SEGMENTATION SYSTEM                      │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │  Raw Transaction │
                              │      Data        │
                              │  (CSV/Parquet)   │
                              └────────┬─────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION LAYER                                │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │  Data Loader    │───▶│   Validator     │───▶│  Transaction Parser     │  │
│  │  (incremental)  │    │  (schema check) │    │  (normalize formats)    │  │
│  └─────────────────┘    └─────────────────┘    └───────────┬─────────────┘  │
└────────────────────────────────────────────────────────────┼────────────────┘
                                                             │
                                                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        FEATURE ENGINEERING LAYER                              │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    RFM Feature Extractor                                 │ │
│  │  ┌─────────┐ ┌───────────┐ ┌──────────┐ ┌────────┐ ┌─────────────────┐  │ │
│  │  │ Recency │ │ Frequency │ │ Monetary │ │ Tenure │ │ Derived Ratios  │  │ │
│  │  └─────────┘ └───────────┘ └──────────┘ └────────┘ └─────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                 Sequence Embedding Transformer                           │ │
│  │  ┌──────────────┐    ┌──────────────┐    ┌────────────────────────────┐ │ │
│  │  │ Transaction  │───▶│   Temporal   │───▶│  Fixed-Length Embedding    │ │ │
│  │  │  Sequences   │    │   Encoding   │    │  (behavioral fingerprint)  │ │ │
│  │  └──────────────┘    └──────────────┘    └────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
                    ▼                                       ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────────┐
│      SEGMENTATION MODULE          │   │         CLV PREDICTION MODULE          │
│                                   │   │                                         │
│  ┌─────────────────────────────┐ │   │  ┌─────────────────────────────────┐   │
│  │   Gaussian Mixture Model    │ │   │  │     Stacked Generalization      │   │
│  │                             │ │   │  │                                 │   │
│  │  ┌─────┐ ┌─────┐ ┌─────┐   │ │   │  │  ┌─────────────────────────┐   │   │
│  │  │ K=3 │ │ K=5 │ │ K=7 │   │ │   │  │  │  Segment-Specific       │   │   │
│  │  └─────┘ └─────┘ └─────┘   │ │   │  │  │  Base Regressors        │   │   │
│  │         │                  │ │   │  │  │  ┌─────┐ ┌─────┐ ┌─────┐│   │   │
│  │         ▼                  │ │   │  │  │  │Seg 1│ │Seg 2│ │Seg N││   │   │
│  │  ┌─────────────────────┐   │ │   │  │  │  └──┬──┘ └──┬──┘ └──┬──┘│   │   │
│  │  │  BIC Model Select   │   │ │   │  │  └─────┼───────┼───────┼───┘   │   │
│  │  └─────────────────────┘   │ │   │  │        │       │       │       │   │
│  │            │               │ │   │  │        ▼       ▼       ▼       │   │
│  │            ▼               │ │   │  │  ┌─────────────────────────┐   │   │
│  │  ┌─────────────────────┐   │ │   │  │  │    Meta-Learner         │   │   │
│  │  │ Soft Cluster Assign │   │◀┼───┼──│  │  (Ridge Regression)     │   │   │
│  │  │ (probabilistic)     │   │ │   │  │  └─────────────────────────┘   │   │
│  │  └─────────────────────┘   │ │   │  │                                 │   │
│  └─────────────────────────────┘ │   │  └─────────────────────────────────┘   │
└───────────────────────────────────┘   └───────────────────────────────────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        UNCERTAINTY QUANTIFICATION                             │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌──────────────────┐  │
│  │  Bootstrap Sampler  │───▶│  Prediction         │───▶│  Confidence      │  │
│  │  (n=1000 resamples) │    │  Distribution       │    │  Intervals       │  │
│  └─────────────────────┘    └─────────────────────┘    └──────────────────┘  │
└───────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           ONLINE LEARNING LAYER                               │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌──────────────────┐  │
│  │  Incremental Data   │───▶│  Warm-Start         │───▶│  Model State     │  │
│  │  Detector           │    │  Retraining         │    │  Persistence     │  │
│  └─────────────────────┘    └─────────────────────┘    └──────────────────┘  │
└───────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT LAYER                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Customer CLV Report                                                     │ │
│  │  ┌───────────┬─────────┬───────────┬────────────┬────────────────────┐  │ │
│  │  │ Customer  │ Segment │ CLV_30d   │ CLV_90d    │ 95% CI             │  │ │
│  │  │ ID        │ (prob)  │ (pred)    │ (pred)     │ [lower, upper]     │  │ │
│  │  └───────────┴─────────┴───────────┴────────────┴────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Selection

### Core Stack

| Component | Choice | Rationale | Tradeoff | Fallback |
|-----------|--------|-----------|----------|----------|
| **Language** | Python 3.10+ | Standard for ML, extensive ecosystem, junior-friendly | Slower than compiled languages | N/A |
| **Data Processing** | pandas + numpy | Well-documented, handles tabular data well | Memory-bound for very large datasets | polars (if pandas too slow) |
| **ML Framework** | scikit-learn | Robust GMM, ensemble methods, good docs | Less flexible than PyTorch for custom models | N/A |
| **Sequence Encoding** | Custom numpy-based | Avoid PyTorch complexity, sufficient for fixed-length | Less expressive than transformers | sentence-transformers (if needed) |
| **Online Learning** | river | Purpose-built for incremental ML, scikit-learn-like API | Smaller community than sklearn | Manual warm-start with sklearn |
| **Persistence** | joblib + JSON | Simple, no database needed | Not suitable for production at scale | SQLite |
| **Testing** | pytest | Industry standard, simple syntax | N/A | unittest |
| **Visualization** | matplotlib | Universally available, good for reports | Less interactive | plotly (if interactivity needed) |

### Key Design Patterns

| Pattern | Application | Benefit |
|---------|-------------|---------|
| **Strategy Pattern** | Swappable regressors in stacked ensemble | Easy to test different base learners |
| **Pipeline Pattern** | Feature engineering chain | Reproducible transforms, sklearn compatibility |
| **Factory Pattern** | Model instantiation | Centralized configuration, easy testing |
| **Observer Pattern** | Online learning triggers | Decoupled data arrival from model updates |

### Concepts Requiring Deeper Understanding

Before implementing, you should understand:

1. **BG/NBD Model Intuition**: How recency and frequency jointly predict future purchases
2. **GMM and EM Algorithm**: How soft clustering assigns probabilistic memberships
3. **Stacked Generalization**: Why out-of-fold predictions prevent leakage
4. **Bootstrap Sampling**: How resampling approximates prediction uncertainty

---

## Phased Implementation Plan

### Phase 1: Data Foundation & RFM Features

**Objective:** Build data loading and core RFM feature extraction pipeline

**Scope:**
- `src/data/loader.py` - Transaction data loading with incremental support
- `src/features/rfm.py` - RFM feature extractor class
- `src/data/validator.py` - Schema validation
- `tests/test_rfm.py` - Unit tests for feature extraction

**Deliverables:**
- Working data loader that handles CSV/Parquet
- RFM features computed correctly for sample dataset
- Validation that catches malformed data

**Verification Method:**
```bash
pytest tests/test_rfm.py -v
python -c "from src.features.rfm import RFMExtractor; print('Import OK')"
```

**Technical Challenges:**
- Handling timezone-naive vs timezone-aware datetimes
- Customers with single transaction (edge case for recency)
- Monetary values of zero or negative (refunds)

**Definition of Done:**
- [ ] RFM features match expected values on synthetic test data
- [ ] Loader handles both CSV and Parquet formats
- [ ] Validator rejects data with missing required columns
- [ ] All tests pass with >90% coverage on these modules

**Code Skeleton:**

```python
# src/features/rfm.py
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np

@dataclass
class RFMConfig:
    """Configuration for RFM feature extraction."""
    customer_id_col: str = "customer_id"
    transaction_date_col: str = "transaction_date"
    monetary_col: str = "amount"
    reference_date: datetime | None = None  # None = use max date in data

class RFMExtractor:
    """
    Extracts Recency, Frequency, Monetary, and Tenure features.

    Follows BG/NBD conventions:
    - Recency: Time since last purchase (in days)
    - Frequency: Number of repeat purchases (total - 1)
    - Monetary: Average transaction value
    - Tenure: Time since first purchase (in days)
    """

    def __init__(self, config: RFMConfig | None = None):
        self.config = config or RFMConfig()
        self._reference_date: datetime | None = None

    def fit(self, transactions: pd.DataFrame) -> "RFMExtractor":
        """Compute reference date from data."""
        # TODO: Implement
        raise NotImplementedError

    def transform(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Extract RFM features per customer."""
        # TODO: Implement
        raise NotImplementedError

    def fit_transform(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(transactions).transform(transactions)
```

---

### Phase 2: Sequence Embedding Transformer

**Objective:** Encode variable-length transaction sequences into fixed-length embeddings

**Scope:**
- `src/features/sequence_encoder.py` - Sequence embedding transformer
- `src/features/temporal.py` - Temporal feature utilities
- `tests/test_sequence_encoder.py` - Unit tests

**Deliverables:**
- Transformer that converts transaction sequences to fixed-length vectors
- Captures temporal patterns (inter-purchase times, trend, seasonality indicators)

**Verification Method:**
```bash
pytest tests/test_sequence_encoder.py -v
```

**Technical Challenges:**
- Customers with very few transactions (padding strategy)
- Very long sequences (truncation vs summarization)
- Numerical stability of temporal features

**Definition of Done:**
- [ ] Embeddings have consistent shape regardless of sequence length
- [ ] Embeddings capture inter-purchase time distribution
- [ ] Handles edge cases (1 transaction, 1000 transactions)
- [ ] Sklearn-compatible transformer interface

**Code Skeleton:**

```python
# src/features/sequence_encoder.py
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class TransactionSequenceEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes variable-length transaction sequences into fixed-length embeddings.

    Embedding components:
    - Inter-purchase time statistics (mean, std, trend)
    - Monetary value statistics (mean, std, trend)
    - Recency-weighted activity scores
    - Periodicity indicators
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        max_sequence_length: int = 100,
        customer_id_col: str = "customer_id",
        transaction_date_col: str = "transaction_date",
        monetary_col: str = "amount"
    ):
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.customer_id_col = customer_id_col
        self.transaction_date_col = transaction_date_col
        self.monetary_col = monetary_col

    def fit(self, X: pd.DataFrame, y=None) -> "TransactionSequenceEncoder":
        """Learn normalization parameters from training data."""
        # TODO: Implement
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform transaction data to embeddings."""
        # TODO: Implement
        raise NotImplementedError
```

---

### Phase 3: Gaussian Mixture Segmentation

**Objective:** Implement probabilistic customer segmentation with automatic K selection

**Scope:**
- `src/segmentation/gmm.py` - GMM wrapper with BIC-based model selection
- `src/segmentation/segment_profiler.py` - Segment interpretation utilities
- `tests/test_segmentation.py` - Unit tests

**Deliverables:**
- GMM that automatically selects optimal number of clusters
- Soft cluster assignments (probabilities, not hard labels)
- Segment profiling for interpretability

**Verification Method:**
```bash
pytest tests/test_segmentation.py -v
```

**Technical Challenges:**
- BIC can be unstable with small samples
- Degenerate clusters (very few members)
- Interpretability of high-dimensional GMM

**Definition of Done:**
- [ ] BIC correctly identifies known cluster count on synthetic data
- [ ] Soft assignments sum to 1.0 per customer
- [ ] Segment profiles provide actionable insights
- [ ] Handles edge case of K=1 (no meaningful segments)

**Code Skeleton:**

```python
# src/segmentation/gmm.py
from sklearn.mixture import GaussianMixture
import numpy as np
from dataclasses import dataclass

@dataclass
class SegmentationResult:
    """Container for segmentation outputs."""
    n_segments: int
    probabilities: np.ndarray  # Shape: (n_customers, n_segments)
    labels: np.ndarray  # Hard assignments for convenience
    bic_scores: dict[int, float]  # K -> BIC mapping
    segment_profiles: dict[int, dict]  # Segment statistics

class CustomerSegmenter:
    """
    Probabilistic customer segmentation using Gaussian Mixture Models.

    Automatically selects optimal K using BIC criterion.
    """

    def __init__(
        self,
        k_range: tuple[int, int] = (2, 10),
        random_state: int = 42,
        n_init: int = 5
    ):
        self.k_range = k_range
        self.random_state = random_state
        self.n_init = n_init
        self._best_model: GaussianMixture | None = None
        self._bic_scores: dict[int, float] = {}

    def fit(self, X: np.ndarray) -> "CustomerSegmenter":
        """Fit GMMs for each K and select best by BIC."""
        # TODO: Implement
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return soft cluster assignments."""
        # TODO: Implement
        raise NotImplementedError

    def get_result(self, X: np.ndarray, feature_names: list[str] | None = None) -> SegmentationResult:
        """Return full segmentation result with profiles."""
        # TODO: Implement
        raise NotImplementedError
```

---

### Phase 4: Stacked Generalization CLV Predictor

**Objective:** Build ensemble CLV predictor using segment-aware stacking

**Scope:**
- `src/prediction/base_regressors.py` - Segment-specific base learners
- `src/prediction/stacker.py` - Meta-learner combining base predictions
- `src/prediction/clv_predictor.py` - Full CLV prediction pipeline
- `tests/test_prediction.py` - Unit tests

**Deliverables:**
- Stacked ensemble that trains per-segment regressors
- Out-of-fold predictions to prevent leakage
- Multi-horizon predictions (30d, 90d, 180d, 365d)

**Verification Method:**
```bash
pytest tests/test_prediction.py -v
```

**Technical Challenges:**
- Proper cross-validation fold handling for stacking
- Segments with very few samples (regularization needed)
- Feature leakage between base and meta learners

**Definition of Done:**
- [ ] Out-of-fold predictions computed correctly
- [ ] Meta-learner trained only on OOF predictions
- [ ] Predictions reasonable on held-out test set
- [ ] Multi-horizon predictions internally consistent

**Code Skeleton:**

```python
# src/prediction/stacker.py
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import numpy as np

class StackedCLVPredictor(BaseEstimator, RegressorMixin):
    """
    Stacked generalization for CLV prediction.

    Architecture:
    1. Base regressors trained per segment (or segment-weighted)
    2. Out-of-fold predictions from base regressors
    3. Meta-learner combines base predictions
    """

    def __init__(
        self,
        base_regressors: list | None = None,
        meta_regressor: BaseEstimator | None = None,
        n_folds: int = 5,
        random_state: int = 42
    ):
        self.base_regressors = base_regressors
        self.meta_regressor = meta_regressor or Ridge(alpha=1.0)
        self.n_folds = n_folds
        self.random_state = random_state
        self._fitted_base: list = []
        self._fitted_meta: BaseEstimator | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, segment_probs: np.ndarray | None = None) -> "StackedCLVPredictor":
        """
        Fit stacked ensemble.

        Args:
            X: Feature matrix
            y: Target CLV values
            segment_probs: Soft segment assignments (optional)
        """
        # TODO: Implement
        raise NotImplementedError

    def predict(self, X: np.ndarray, segment_probs: np.ndarray | None = None) -> np.ndarray:
        """Predict CLV values."""
        # TODO: Implement
        raise NotImplementedError
```

---

### Phase 5: Bootstrap Confidence Intervals

**Objective:** Quantify uncertainty in CLV predictions

**Scope:**
- `src/uncertainty/bootstrap.py` - Bootstrap sampling and CI computation
- `src/uncertainty/prediction_intervals.py` - Interval construction methods
- `tests/test_uncertainty.py` - Unit tests

**Deliverables:**
- Bootstrap-based prediction intervals
- Configurable confidence levels (90%, 95%, 99%)
- Efficient parallelized bootstrap implementation

**Verification Method:**
```bash
pytest tests/test_uncertainty.py -v
```

**Technical Challenges:**
- Computational cost of many bootstrap iterations
- Proper handling of nested cross-validation in bootstrap
- Calibration of intervals (do 95% intervals contain 95% of true values?)

**Definition of Done:**
- [ ] Bootstrap correctly resamples with replacement
- [ ] Confidence intervals have correct coverage on synthetic data
- [ ] Runtime acceptable (<1 min for 1000 customers, 100 bootstrap samples)
- [ ] Intervals widen appropriately for uncertain predictions

**Code Skeleton:**

```python
# src/uncertainty/bootstrap.py
from dataclasses import dataclass
import numpy as np
from typing import Callable
from concurrent.futures import ProcessPoolExecutor

@dataclass
class PredictionInterval:
    """Container for prediction with uncertainty."""
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float

class BootstrapPredictor:
    """
    Wraps a predictor to provide bootstrap confidence intervals.
    """

    def __init__(
        self,
        base_predictor_factory: Callable,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.base_predictor_factory = base_predictor_factory
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._bootstrap_predictors: list = []

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "BootstrapPredictor":
        """Fit bootstrap ensemble."""
        # TODO: Implement
        raise NotImplementedError

    def predict_with_intervals(self, X: np.ndarray) -> list[PredictionInterval]:
        """Return predictions with confidence intervals."""
        # TODO: Implement
        raise NotImplementedError
```

---

### Phase 6: Online Learning Integration

**Objective:** Enable incremental model updates as new data arrives

**Scope:**
- `src/online/incremental_trainer.py` - Warm-start retraining logic
- `src/online/data_monitor.py` - New data detection
- `src/persistence/model_store.py` - Model serialization
- `tests/test_online.py` - Unit tests

**Deliverables:**
- Model persistence with versioning
- Incremental update triggering
- Warm-start retraining from saved state

**Verification Method:**
```bash
pytest tests/test_online.py -v
```

**Technical Challenges:**
- Concept drift detection
- When to fully retrain vs incrementally update
- Model state compatibility across versions

**Definition of Done:**
- [ ] Models save and load correctly
- [ ] Warm-start reduces training time vs cold start
- [ ] Incremental updates maintain prediction quality
- [ ] Version compatibility handled gracefully

**Code Skeleton:**

```python
# src/online/incremental_trainer.py
from pathlib import Path
import joblib
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ModelVersion:
    """Metadata for a saved model version."""
    version_id: str
    created_at: datetime
    training_data_hash: str
    metrics: dict

class IncrementalTrainer:
    """
    Manages incremental model updates.

    Strategies:
    1. Full retrain: When significant data shift detected
    2. Warm-start: Continue training from saved state
    3. Skip: No update if data unchanged
    """

    def __init__(
        self,
        model_dir: Path,
        retrain_threshold: float = 0.1,  # Fraction of new data triggering retrain
    ):
        self.model_dir = Path(model_dir)
        self.retrain_threshold = retrain_threshold
        self._current_version: ModelVersion | None = None

    def should_update(self, new_data_size: int, total_data_size: int) -> str:
        """Determine update strategy: 'full', 'warm', or 'skip'."""
        # TODO: Implement
        raise NotImplementedError

    def save_model(self, model, metrics: dict) -> ModelVersion:
        """Persist model with metadata."""
        # TODO: Implement
        raise NotImplementedError

    def load_latest(self):
        """Load most recent model version."""
        # TODO: Implement
        raise NotImplementedError
```

---

### Phase 7: Integration & CLI

**Objective:** Create end-to-end pipeline with command-line interface

**Scope:**
- `src/pipeline.py` - Full pipeline orchestration
- `src/cli.py` - Command-line interface
- `tests/test_integration.py` - End-to-end tests
- Sample dataset and demo script

**Deliverables:**
- Single entry point for training and prediction
- CLI for common operations
- Demo with sample data

**Verification Method:**
```bash
python -m src.cli train --data sample_data.csv --output models/
python -m src.cli predict --model models/latest --customers new_customers.csv
```

**Definition of Done:**
- [ ] End-to-end pipeline runs without errors
- [ ] CLI provides helpful error messages
- [ ] Demo produces interpretable outputs
- [ ] Documentation covers all CLI commands

---

## Risk Assessment

| Risk | Likelihood | Impact | Early Warning Signs | Mitigation |
|------|------------|--------|---------------------|------------|
| GMM fails to converge | Medium | High | Many EM iterations, unstable log-likelihood | Add convergence checks, fallback to K-Means init |
| Bootstrap too slow | High | Medium | Single iteration >1s | Reduce n_bootstrap, add parallelization, sample customers |
| Stacking leaks data | Medium | High | Unrealistically good validation scores | Strict OOF protocol, nested CV for tuning |
| Sequence encoder overcomplicates | Medium | Low | Diminishing returns on embeddings | Start simple (stats only), add complexity if needed |
| Online learning degrades quality | Low | High | Prediction drift over time | Regular full retraining, quality monitoring |
| Memory issues with large datasets | Medium | Medium | OOM errors, slow processing | Chunked processing, memory profiling |

---

## Testing Strategy

### Testing Pyramid

```
                    ┌─────────────┐
                    │   System    │  End-to-end pipeline tests
                    │   Tests     │  (1-2 tests)
                    └─────────────┘
               ┌────────────────────────┐
               │   Integration Tests    │  Component interaction
               │   (5-10 tests)         │  (segmentation -> prediction)
               └────────────────────────┘
          ┌──────────────────────────────────┐
          │         Unit Tests               │  Individual functions
          │         (30-50 tests)            │  (RFM, embeddings, GMM)
          └──────────────────────────────────┘
```

### Testing Framework

- **Framework:** pytest
- **Coverage Target:** >80% for core modules
- **Fixtures:** Synthetic datasets with known properties

### First Three Tests to Write

```python
# tests/test_rfm.py

import pytest
import pandas as pd
from datetime import datetime
from src.features.rfm import RFMExtractor, RFMConfig

@pytest.fixture
def sample_transactions():
    """Create synthetic transaction data with known RFM values."""
    return pd.DataFrame({
        "customer_id": ["A", "A", "A", "B", "B"],
        "transaction_date": pd.to_datetime([
            "2024-01-01", "2024-02-01", "2024-03-01",  # Customer A: 3 transactions
            "2024-01-15", "2024-03-15"  # Customer B: 2 transactions
        ]),
        "amount": [100.0, 150.0, 200.0, 50.0, 75.0]
    })

def test_rfm_frequency_counts_repeat_purchases(sample_transactions):
    """Frequency should be number of purchases minus 1 (repeat purchases only)."""
    extractor = RFMExtractor()
    rfm = extractor.fit_transform(sample_transactions)

    # Customer A: 3 transactions -> frequency = 2
    # Customer B: 2 transactions -> frequency = 1
    assert rfm.loc["A", "frequency"] == 2
    assert rfm.loc["B", "frequency"] == 1

def test_rfm_monetary_is_average_value(sample_transactions):
    """Monetary should be average transaction value."""
    extractor = RFMExtractor()
    rfm = extractor.fit_transform(sample_transactions)

    # Customer A: (100 + 150 + 200) / 3 = 150
    # Customer B: (50 + 75) / 2 = 62.5
    assert rfm.loc["A", "monetary"] == pytest.approx(150.0)
    assert rfm.loc["B", "monetary"] == pytest.approx(62.5)

def test_rfm_handles_single_transaction_customer():
    """Customers with single transaction should have frequency=0."""
    transactions = pd.DataFrame({
        "customer_id": ["C"],
        "transaction_date": pd.to_datetime(["2024-01-01"]),
        "amount": [100.0]
    })

    extractor = RFMExtractor()
    rfm = extractor.fit_transform(transactions)

    assert rfm.loc["C", "frequency"] == 0
```

---

## First Concrete Task

### File to Create

`src/features/rfm.py`

### Function Signature

```python
def compute_rfm_features(
    transactions: pd.DataFrame,
    customer_id_col: str = "customer_id",
    transaction_date_col: str = "transaction_date",
    monetary_col: str = "amount",
    reference_date: datetime | None = None
) -> pd.DataFrame:
    """
    Compute RFM features for each customer.

    Args:
        transactions: DataFrame with transaction records
        customer_id_col: Column name for customer identifier
        transaction_date_col: Column name for transaction timestamp
        monetary_col: Column name for transaction value
        reference_date: Reference date for recency calculation (default: max date in data)

    Returns:
        DataFrame indexed by customer_id with columns:
        - recency: Days since last purchase
        - frequency: Number of repeat purchases
        - monetary: Average transaction value
        - tenure: Days since first purchase
    """
```

### Copy-Paste Starter Code

```python
# src/features/rfm.py
"""RFM (Recency, Frequency, Monetary) feature extraction for CLV prediction."""

from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np


@dataclass
class RFMConfig:
    """Configuration for RFM feature extraction."""
    customer_id_col: str = "customer_id"
    transaction_date_col: str = "transaction_date"
    monetary_col: str = "amount"
    reference_date: datetime | None = None


def compute_rfm_features(
    transactions: pd.DataFrame,
    customer_id_col: str = "customer_id",
    transaction_date_col: str = "transaction_date",
    monetary_col: str = "amount",
    reference_date: datetime | None = None
) -> pd.DataFrame:
    """
    Compute RFM features for each customer.

    Args:
        transactions: DataFrame with transaction records
        customer_id_col: Column name for customer identifier
        transaction_date_col: Column name for transaction timestamp
        monetary_col: Column name for transaction value
        reference_date: Reference date for recency calculation (default: max date in data)

    Returns:
        DataFrame indexed by customer_id with columns:
        - recency: Days since last purchase
        - frequency: Number of repeat purchases
        - monetary: Average transaction value
        - tenure: Days since first purchase
    """
    # Validate required columns exist
    required_cols = [customer_id_col, transaction_date_col, monetary_col]
    missing = [c for c in required_cols if c not in transactions.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure datetime type
    df = transactions.copy()
    df[transaction_date_col] = pd.to_datetime(df[transaction_date_col])

    # Set reference date
    if reference_date is None:
        reference_date = df[transaction_date_col].max()

    # Group by customer and compute features
    # TODO: Implement the aggregation logic
    # Hint: Use groupby().agg() with appropriate functions

    raise NotImplementedError("Implement RFM aggregation logic")


class RFMExtractor:
    """
    Sklearn-compatible transformer for RFM feature extraction.

    Follows BG/NBD conventions:
    - Recency: Time since last purchase (in days)
    - Frequency: Number of repeat purchases (total - 1)
    - Monetary: Average transaction value
    - Tenure: Time since first purchase (in days)
    """

    def __init__(self, config: RFMConfig | None = None):
        self.config = config or RFMConfig()
        self._reference_date: datetime | None = None

    def fit(self, transactions: pd.DataFrame) -> "RFMExtractor":
        """Compute reference date from data."""
        date_col = self.config.transaction_date_col
        self._reference_date = pd.to_datetime(transactions[date_col]).max()
        return self

    def transform(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Extract RFM features per customer."""
        if self._reference_date is None:
            raise ValueError("Must call fit() before transform()")

        return compute_rfm_features(
            transactions,
            customer_id_col=self.config.customer_id_col,
            transaction_date_col=self.config.transaction_date_col,
            monetary_col=self.config.monetary_col,
            reference_date=self._reference_date
        )

    def fit_transform(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(transactions).transform(transactions)
```

### Verification Method

```bash
# After implementing compute_rfm_features:
cd /path/to/dynamic-clv-segmentation
python -c "
from src.features.rfm import RFMExtractor
import pandas as pd

# Quick smoke test
df = pd.DataFrame({
    'customer_id': ['A', 'A', 'B'],
    'transaction_date': ['2024-01-01', '2024-02-01', '2024-01-15'],
    'amount': [100, 200, 50]
})

extractor = RFMExtractor()
rfm = extractor.fit_transform(df)
print(rfm)
print('SUCCESS: RFM extraction working')
"
```

### First Commit Message

```
feat: Add RFM feature extraction module

Implement compute_rfm_features() and RFMExtractor class for
extracting Recency, Frequency, Monetary, and Tenure features
from transaction data following BG/NBD conventions.
```

---

## Project Structure

```
dynamic-clv-segmentation/
├── README.md
├── pyproject.toml
├── docs/
│   └── IMPLEMENTATION_PLAN.md
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── validator.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── rfm.py
│   │   ├── sequence_encoder.py
│   │   └── temporal.py
│   ├── segmentation/
│   │   ├── __init__.py
│   │   ├── gmm.py
│   │   └── segment_profiler.py
│   ├── prediction/
│   │   ├── __init__.py
│   │   ├── base_regressors.py
│   │   ├── stacker.py
│   │   └── clv_predictor.py
│   ├── uncertainty/
│   │   ├── __init__.py
│   │   ├── bootstrap.py
│   │   └── prediction_intervals.py
│   ├── online/
│   │   ├── __init__.py
│   │   ├── incremental_trainer.py
│   │   └── data_monitor.py
│   ├── persistence/
│   │   ├── __init__.py
│   │   └── model_store.py
│   ├── pipeline.py
│   └── cli.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_rfm.py
│   ├── test_sequence_encoder.py
│   ├── test_segmentation.py
│   ├── test_prediction.py
│   ├── test_uncertainty.py
│   ├── test_online.py
│   └── test_integration.py
└── examples/
    └── demo.py
```

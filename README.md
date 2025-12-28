# Dynamic CLV Segmentation

A probabilistic Customer Lifetime Value prediction system that discovers latent customer segments, handles non-contractual churn scenarios, and updates predictions incrementally as new transaction data arrives.

## Overview

This system addresses the challenge of predicting future customer value in non-contractual business settings where customers can churn silently without explicit notification. It combines probabilistic customer behavior modeling with unsupervised segmentation to provide actionable CLV estimates with uncertainty quantification.

## Core Capabilities

- **BG/NBD-Style Feature Engineering**: Extracts recency, frequency, monetary value, and tenure features following established customer behavior modeling frameworks
- **Probabilistic Segmentation**: Uses Gaussian Mixture Models to discover latent customer segments that inform value predictions
- **Stacked Generalization**: Combines segment-specific regressors through meta-learning for robust CLV estimation
- **Online Learning**: Supports incremental model updates as new transaction data becomes available
- **Sequence Embeddings**: Custom transformers encode transaction sequences into fixed-length representations capturing behavioral patterns
- **Uncertainty Quantification**: Bootstrap-based confidence intervals provide reliability bounds on lifetime value estimates

## Project Status

This project is currently in the implementation planning phase. See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the detailed development roadmap.

## Technical Requirements

- Python 3.10+
- Free/open-source dependencies only
- Designed to run on laptop hardware with optional free cloud services

## License

MIT License

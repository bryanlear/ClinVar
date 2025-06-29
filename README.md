# ClinVar Analysis Project

A comprehensive toolkit for downloading, processing, and analyzing ClinVar variant classification data to track reclassifications over time using LSTM-based models.

## Project Structure

```
ClinVar-1/
├── clinvar_toolkit/          # ClinVar data acquisition toolkit
│   ├── scripts/              # Download and utility scripts
│   ├── config/               # Configuration files
│   ├── examples/             # Usage examples
│   └── docs/                 # Documentation
├── data/                     # Data storage
│   ├── raw/                  # Raw downloaded ClinVar XML files
│   └── processed/            # Processed analysis data
├── logs/                     # Log files from operations
├── tools/                    # Additional analysis tools
├── results/                  # Analysis results and outputs
├── old_strategy/             # Previous analysis approach (archived)
└── setup.py                  # Project setup script
```

## Quick Start

### 1. Setup Project
```bash
python setup.py
```

### 2. Download ClinVar Data
```bash
cd clinvar_toolkit/scripts
python download_clinvar_vcv_releases.py
```

### 3. Check Download Status
```bash
python clinvar_download_utils.py status
```

## Strategy Evolution

**Note**: The analysis strategy has evolved. All previous data, scripts, and results have been archived in the `old_strategy` directory.

<!DOCTYPE html>
<html>
<head>
LSTM-Based Model Workflow for Variant Reclassification
</head>
<body>

<hr>

<h3>1. Data Ingestion and Structuring</h3>
<ul>
    <li>Collect historical variant classification data with time-stamps (monthly) XML format.</li>
    <li>Each unique genetic variant forms an independent time series.</li>
</ul>

<hr>

<h3>2. Feature Engineering and Encoding</h3>
<ul>
    <li><strong>Primary Feature:</strong> Encode categorical classification labels (B, LB, VUS, LP, P) using <b>Entity Embeddings</b> for a dense numerical representation capturing semantic relationships.</li>
    <li><strong>Temporal Features:</strong> Add time-based features such as time elapsed since initial classification, time since last change, and binary flags for significant events (e.g., pre/post-2015 ACMG guidelines).</li>
</ul>

<hr>

<h3>3. Data Preparation for LSTM</h3>
<ul>
    <li>Use<b>sliding window approach</b> (e.g., via <code>tf.keras.utils.timeseries_dataset_from_array</code>) to transform each variant's time series into input-output samples.</li>
    <li>Input: Sequence of feature vectors (embedding + temporal features).</li>
    <li>Target: Probability distribution of classification at a future time step.</li>
</ul>

<hr>

<h3>4. Robust Validation and Hyperparameter Tuning</h3>
<ul>
    <li>Employ <b>Nested Cross-Validation</b> for unbiased performance estimation.</li>
    <li>Use <b>scikit-learn's TimeSeriesSplit</b> for both inner (hyperparameter tuning) and outer (evaluation) loops to maintain temporal order.</li>
    <li>Key hyperparameters to tune: LSTM's learning rate, number of hidden units, dropout rate, and input window size.</li>
</ul>

<hr>

<h3>5. Model Architecture and Training</h3>
<ul>
    <li>Develop an <b>LSTM-based architecture</b>, starting with a Stacked LSTM with an Attention mechanism.</li>
    <li>Final layer: Dense layer with <b>softmax activation</b> and five output neurons for probability distribution over classification tiers.</li>
    <li>Train by minimizing a proper scoring rule, such as <b>Categorical Cross-Entropy (Logloss)</b>.</li>
</ul>

<hr>

<h3>6. Evaluation and Calibration</h3>
<ul>
    <li>Evaluate final model using <b>Logloss</b> and the <b>Brier Score</b>.</li>
    <li>Assess probability reliability with <b>Calibration Plots</b>. Apply <b>Platt Scaling</b> if poorly calibrated.</li>
</ul>

<hr>

<h3>7. Deployment and Monitoring for Concept Drift</h3>
<ul>
    <li>Deploy the trained model within an <b>online learning framework</b>.</li>
    <li>Continuously monitor prediction error on new data. Use a system (e.g., inspired by LSTMDD) to detect significant error increases, signaling concept drift and triggering retraining.</li>
</ul>

<hr>

<h3>Strategies for Model Interpretability in a Clinical Context</h3>
<p>In a high-stakes clinical environment model interpretability is crucial for building trust and augmenting human expertise (goal is augmenting human expertise, not replacing it).</p>

<hr>

<h3>Human-in-the-Loop (HITL) Framework</h3>
<ul>
    <li>The model should function as a <b>clinical decision support tool</b> prioritizing VUSs for manual review by expert curators.</li>
    <li>It would flag variants with high predicted probability of reclassification to a pathogenic category allowing experts to focus on critical cases.</li>
</ul>

<hr>

<h3>Explainability Techniques</h3>
<ul>
    <li><strong>Post-Hoc Explanations:</strong> Apply model-agnostic methods like <b>LIME</b> or <b>SHAP</b> to explain individual predictions. These tools can highlight influential past classifications.</li>
    <li><strong>Intrinsic Interpretability:</strong>
        <ul>
            <li>Visualize <b>learned attention weights</b> to show which time steps the model "focused" on.</li>
            <li>Analyze <b>learned entity embeddings</b> (e.g., using t-SNE) to confirm semantic relationships between classification tiers.</li>
        </ul>
    </li>
</ul>

</body>
</html>


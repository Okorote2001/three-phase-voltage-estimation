# Deep Learning for Three-Phase Voltage Parameter Estimation

**A MATLAB-based supervised learning project for estimating fundamental voltage parameters in three-phase power systems.**

## Overview

This project investigates the use of a fully connected feedforward neural network to estimate the fundamental amplitudes, phase angles, and frequency of three-phase voltage signals from sampled waveform data.

The study explores how sampling frequency, observation-window length, training dataset size, and neural network architecture affect estimation performance and computational cost.

The work is currently at the simulation and experimental-validation stage. Initial experiments have been conducted, and the experimental design is being refined to improve the reliability and reproducibility of the results.

## Objectives

- Estimate the fundamental amplitudes of three-phase voltages.
- Estimate the phase angles of the three voltage signals.
- Estimate the fundamental grid frequency.
- Investigate the effects of sampling frequency and observation-window length.
- Evaluate the influence of training dataset size and neural network architecture.
- Study the effects of voltage imbalance and waveform disturbances.

## Estimated Parameters

The neural network is designed to predict seven continuous-valued parameters:

\[
\mathbf{y} =
[A_a,A_b,A_c,\phi_a,\phi_b,\phi_c,f]^T
\]

where:

- \(A_a,A_b,A_c\): fundamental voltage amplitudes of phases A, B, and C.
- \(\phi_a,\phi_b,\phi_c\): fundamental phase angles of the three phases.
- \(f\): fundamental frequency.

The input consists of sampled voltage waveforms from all three phases. If each phase contributes \(N\) samples, the input contains \(3N\) features.

## Methodology

The project uses supervised regression with a fully connected feedforward neural network (multilayer perceptron).

The general workflow is:

1. Generate or prepare three-phase voltage signals.
2. Construct input features from the sampled voltage waveforms.
3. Associate each input with its known target parameters.
4. Train the neural network using labelled examples.
5. Evaluate predictions against known target values.
6. Compare experimental configurations using estimation errors and training time.

The experiments investigate the following factors:

- **Sampling frequency:** its effect on input dimensionality and estimation performance.
- **Observation-window length:** the amount of waveform information available to the network.
- **Training dataset size:** the effect of the number of labelled examples.
- **Network architecture and activation functions:** their influence on learning and generalisation.

## Signal Disturbances

The experimental design considers non-ideal three-phase voltage signals, including:

- Amplitude imbalance.
- Harmonic distortion.
- DC offset.
- Additive noise.
- Variation in fundamental frequency.

The revised experimental design includes the 5th, 7th, 13th, and 17th harmonics. These updated conditions are being incorporated into the validation process; they should not be assumed to be present in every previously reported experiment.

## Repository Structure

```text
three-phase-voltage-estimation/
├── README.md
├── src/
│   ├── Experiment1_DatasetSize_*.m
│   ├── Experiment2_Optimized_Frequency_*.m
│   ├── Experiment3_ObservationWindow_*.m
│   └── Goldilocks_Combined_*.m
├── results/
│   ├── Experiment1_DatasetSize_Results/
│   ├── Experiment2_Optimized_Frequency_Results/
│   ├── Experiment3_ObservationWindow_Results/
│   └── Goldilocks_Combined_Results/
├── docs/
└── presentations/
```

*The names shown above are illustrative patterns. The actual filenames and folder names may differ.*

## Evaluation Metrics

The project uses the following metrics to assess estimation performance:

- **Mean Absolute Error (MAE):** the average absolute difference between predicted and reference values.
- **Root Mean Squared Error (RMSE):** a measure that gives greater weight to larger prediction errors.
- **Training time:** the time required to train a model under a specified configuration.

Errors should be reported in appropriate units: volts for amplitude, degrees for phase angle, and hertz for frequency.

Training time is distinct from inference latency. Real-time suitability requires separate evaluation of prediction latency on the intended hardware.

## Current Status

**Status: Ongoing research and simulation-based validation.**

Initial experiments have investigated dataset size, sampling frequency, observation-window length, and combined experimental configurations.

The experimental design is being revised to address:

- The observation window required for the specified frequency range.
- Inclusion and evaluation of higher-order harmonics.
- Controlled comparisons of activation functions.
- Comparison of hidden-layer widths.
- Consistent validation and independent testing.

Previously reported results are preliminary and correspond to their original experimental settings. They should not be interpreted as validated results under the revised design.

## Planned Improvements

- Complete the revised signal-generation and sampling setup.
- Compare ReLU/ReLU and tanh/ReLU activation configurations under identical training conditions.
- Evaluate alternative hidden-layer widths using validation error and model complexity.
- Repeat the main experiments while preserving an independent test set.
- Document updated results and compare them with the preliminary experiments.
- Investigate performance on measured waveforms and assess inference latency on suitable hardware.

## Tools and Technologies

- MATLAB
- Supervised machine learning
- Feedforward neural networks
- Power-system signal analysis
- Regression evaluation and data visualisation

## Limitations

This project currently provides evidence from simulation-based experiments. It does not establish performance on real electrical grids or prove real-time operation on embedded hardware.

Results depend on the signal-generation assumptions, dataset, network configuration, and training procedure. Further validation is required before practical deployment.

## Research Context

This project explores the intersection of electrical engineering, power-system monitoring, signal processing, and artificial intelligence.

It is an ongoing technical learning and research project, with an emphasis on controlled experimentation, transparent reporting, and reproducible evaluation.

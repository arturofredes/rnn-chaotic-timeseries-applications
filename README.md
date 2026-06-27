# Recurrent Neural Networks for Time Series Forecasting in Complex Dynamical Systems

[lorenz](ssLorenz16.jpg)

**Author:** Arturo Fredes Cáceres
**Supervisors:** Dr. Sergio Gutiérrez Rodrigo ([sergut@unizar.es](mailto:sergut@unizar.es)) · César Arbiol Herrera
**Institution:** Facultad de Ciencias, Universidad de Zaragoza · Mathematics Bachelor Thesis · 2022–2023

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## Overview

This project studies how **Recurrent Neural Networks (RNNs)** perform when forecasting future steps of time series, working with datasets of increasing complexity. Three scenarios are explored:

1. **Toy Model** — a simple 1D signal (sum of two waves + noise) used as a proof of concept.
2. **Lorenz Attractor** — a 3D chaotic dynamical system to push the limits of the models.
3. **Business Data** — real demand forecasting for literature books in collaboration with Editorial Edelvives.

The code is written in **Python** using the **Keras** library. Models were trained on **Google Colab** (free GPU) and locally on an Intel Core i5 with 8 GB RAM.

For further reading, the full thesis report is available in this repository and in the [Universidad de Zaragoza institutional repository (Zaguan)](https://zaguan.unizar.es/record/134455/).

---

## Results Summary

### Toy Model (1D wave data)
Different cell types (Feedforward, Simple Recurrent, LSTM, GRU) were compared on 10,000 generated sequences of 50 steps. When organized into deep networks of two layers of 20 cells, LSTM achieved the best MSE of **0.0026**, an improvement of **86.9%** over the naïve baseline.

Three multi-step forecasting strategies were also compared using LSTM:

| Method | Average MSE | Last Step MSE |
|---|---|---|
| Iterative | 0.0183 | 0.0363 |
| Sequence-to-Vector | 0.0102 | 0.0193 |
| Sequence-to-Sequence | **0.0028** | **0.0042** |

### Lorenz Attractor (3D chaotic system)
RNNs were tested on 5,000 sequences of 1,000 steps generated from the Lorenz system with parameters σ=10, ρ=28, β=8/3. Feedforward networks failed to beat the naïve baseline, while LSTM achieved a test MSE of **0.0005** — an improvement of nearly **99.97%**.

For forecasting 100 steps ahead:

| Method | Sequence MSE | Last Step MSE |
|---|---|---|
| Iterative | 87.30 | 111.39 |
| Sequence-to-Vector | 3.47 | 8.17 |
| Sequence-to-Sequence | **1.18** | **2.27** |

### Business Data (Editorial Edelvives)
A sequence-to-sequence LSTM with MC Dropout was trained on monthly sales data for 4,081 book titles over 5 years, predicting demand 6 months ahead. The final model achieved a **MAE of ~160 books** overall, dropping to **~80 books** when excluding articles with unannounced promotional campaigns — a missing feature identified for future work.

---

## Repository Structure

```
├── toy_model/                  # Notebooks and scripts for the 1D wave proof-of-concept
│                               # Covers single-step and multi-step forecasting with
│                               # Feedforward, RNN, LSTM, and GRU architectures
│
├── lorenz_attractor_model/     # Notebooks and scripts for the Lorenz system experiments
│                               # Includes data generation via 4th-order Runge-Kutta,
│                               # model training, and iterative/S-V/S-S comparison
│
└── Recurrent Neural Networks [...].pdf   # Full thesis report
```

---

## Methods

### Cell Types Compared
- **Feedforward (Dense)** — baseline, no temporal memory
- **Simple Recurrent (RNN)** — limited short-term memory
- **LSTM** — long- and short-term memory via forget, input, and output gates
- **GRU** — simplified LSTM with a single gating mechanism; competitive performance with fewer parameters

### Multi-Step Forecasting Strategies
- **Iterative** — predict one step at a time, feeding predictions back as input; error accumulates
- **Sequence-to-Vector (S-V)** — predict all future steps at once from the input sequence
- **Sequence-to-Sequence (S-S)** — predict the next *m* steps at every timestep; best performance, higher computational cost

### MC Dropout (Business Model)
Dropout layers are kept active at inference time, and predictions are averaged over 100 forward passes to produce more robust estimates and implicitly quantify uncertainty.

---

## Requirements

```
Python 3.x
TensorFlow / Keras
NumPy
Matplotlib
SciPy          # for Runge-Kutta data generation
```

Training was done with the **Adam optimizer** and **MSE loss** throughout, with **MAE** used for final evaluation on the business dataset.

---

## Further Reading

The full thesis report is available:
- In this repository as a PDF
- In the [Universidad de Zaragoza institutional repository (Zaguan)](https://zaguan.unizar.es/record/134455/)

For questions, feel free to reach out to the supervisors.

> Distributed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](http://creativecommons.org/licenses/by-nc-sa/4.0/).

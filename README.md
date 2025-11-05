
# Predictive Maintenance of Aircraft Engines using LSTM

## Overview
This repository presents a deep learning-based approach for **Predicting the Remaining Useful Life (RUL)** of **turbofan aircraft engines** using **Long Short-Term Memory (LSTM)** networks.  
The project leverages temporal dependencies in multivariate sensor data to forecast potential failures, enabling **predictive maintenance** and reducing unplanned downtime.

---

## Dataset
**Source:** [NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

Each engine in the dataset runs to failure, generating multi-sensor time-series data.

| Feature Type | Description |
|---------------|-------------|
| **Cycle** | Operational time step (engine run cycle) |
| **Operational Settings (3)** | Environmental and load conditions |
| **Sensor Measurements (21)** | Engine health indicators (temperature, pressure, vibration, etc.) |
| **RUL** | Remaining Useful Life = (Max Cycle - Current Cycle) |

---

## ⚙Workflow

### **1. Data Acquisition & Preprocessing**
- Load the NASA C-MAPSS dataset (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`)
- Compute Remaining Useful Life (RUL) for each engine
- Remove non-informative/static features
- Normalize all sensor readings to [-1, 1] range using `MinMaxScaler`

### **2. Sequence Generation**
- Convert continuous time-series into overlapping sequences  
- Sequence length (**τ**) = 50 cycles  
- Each sequence represents the engine’s recent 50-cycle history

### **3. LSTM Model Design**
Four LSTM models were developed and compared:

| Model | Architecture | Dropout | Early Stopping | Description |
|--------|--------------|----------|----------------|--------------|
| **Model 1** | 1 LSTM (4 units) | ❌ | ❌ | Baseline network |
| **Model 2** | 1 LSTM (4 units) | ❌ | ✅ | Prevents overfitting |
| **Model 3** | 1 LSTM (4 units) | ✅ | ✅ | Improved generalization |
| **Model 4** | 2 LSTMs (8 + 4 units) | ✅ | ✅ | Stacked LSTM for deeper learning |

**Training Configuration**
- Loss: `binary_crossentropy`  
- Optimizer: `Adam`  
- Epochs: 100  
- Batch Size: 64  
- Class Weighting for imbalanced data  
- Validation Split: 0.1  

---

## Evaluation
The models were evaluated using multiple metrics and visualizations:

| Metric | Description |
|---------|--------------|
| **Accuracy** | Overall prediction correctness |
| **F1-Score** | Balance between precision & recall |
| **ROC-AUC** | Model’s discrimination ability |
| **Confusion Matrix** | True vs Predicted class visualization |

**Key Observations:**
- Stacked LSTM (Model 4) achieved the best stability and accuracy  
- Dropout + Early Stopping significantly reduced overfitting  
- Time-windowing effectively captured degradation patterns  

---

## Experimental Results
10-run experiments were conducted for robustness testing.

Results include:
- Performance boxplots for Accuracy, F1, Recall, AUC
- ROC-AUC Curves
- Confusion Matrices for each model

```

Raw Data → Normalized → Sequenced (τ = 50) → LSTM Training → RUL Prediction → Evaluation

```

---

## Key Insights
- **Temporal modeling** via LSTM captures long-term degradation dependencies  
- **Predictive maintenance** approach allows proactive scheduling of repairs  
- **Balanced data and early stopping** improve generalization in real-world applications  


---

## Keywords
`Predictive Maintenance` · `LSTM` · `RUL Prediction` · `Aircraft Engines` · `Deep Learning` · `NASA C-MAPSS` · `Time-Series Forecasting`

---

## References

- Saxena, A.; Goebel, K.; Simon, D.; Eklund, N. “Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.” *IEEE Transactions on Reliability*, vol. 57, no. 3, Sept. 2008, pp. Upage–Dpage.  
- NASA Ames Research Center. “C-MAPSS Turbofan Engine Degradation Simulation Data Set.” Prognostics Data Repository, 2008.  
- Bishal S. (2023). “NASA Turbofan Engine Degradation Simulation.” Kaggle. Available at: https://www.kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation 


---

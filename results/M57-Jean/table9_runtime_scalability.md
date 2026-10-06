## Table 9. Runtime & Scalability Analysis

| Pipeline Stage | M57-Jean Time (seconds) | MalMem2022 Time (seconds) | Peak Memory (MB) |
|---|---:|---:|---:|
| Data normalization | N/M | N/M | N/M |
| Temporal feature engineering | N/M | N/A | N/M |
| Temporal anomaly detection | N/M | N/A | N/M |
| Knowledge graph construction | N/M | N/A | N/M |
| Graph anomaly detection | N/M | N/A | N/M |
| SHAP explanation generation | N/M | N/A | N/M |

**Note:** N/M = not measured in the current reproducible branch. N/A indicates that the stage is not part of the MalMem2022 supervised classification pipeline. MalMem2022 is evaluated as a separate supervised memory-forensics classification track and is not forced through the M57 temporal/graph workflow.

# Report – PyTorch CV Feature Explorer

## 1. Project Overview
This project investigates **what a CNN learns internally** rather than only focusing on final accuracy.  
Using a custom CNN trained on **CIFAR‑10**, we analyze intermediate representations through multiple visualization techniques.

The goal is **interpretability**, not leaderboard performance.

---

## 2. Dataset
**CIFAR‑10**
- 60,000 RGB images (32×32)
- 10 classes:
  airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Split:
  - Train (45,000)
  - Validation (5,000)
  - Test (10,000)

---

## 3. Model Architecture
A lightweight CNN built from scratch using PyTorch.

**Key components**
- Convolutional blocks: Conv → ReLU → MaxPool
- Fully connected classifier head
- Cross‑entropy loss
- Adam optimizer

The architecture is intentionally simple to make feature interpretation clearer.

---

## 4. Training Setup
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Epochs: 20
- Best model selected using validation accuracy

Training metrics were logged per epoch and saved for visualization.

---

## 5. Learning Curves Analysis

### Loss Curves
![Loss Curves](outputs/plots/loss_curves.png)

**Observations**
- Both training and validation loss decrease steadily
- No strong divergence → limited overfitting
- Indicates reasonable generalization

### Accuracy Curves
![Accuracy Curves](outputs/plots/accuracies.png)

**Observations**
- Training accuracy increases smoothly
- Validation accuracy follows closely
- Slight gap suggests mild capacity limitation rather than overfitting

---

## 6. Confusion Matrix
![Confusion Matrix](outputs/plots/confusion_matrix.png)

**Insights**
- Strong diagonal dominance → correct predictions
- Common confusions:
  - cat ↔ dog
  - deer ↔ horse
  - automobile ↔ truck
- Confusions align with semantic similarity in CIFAR‑10

---

## 7. Misclassified Examples
![Misclassified Examples](outputs/plots/misclassified_examples.png)

**Why these fail**
- Low resolution (32×32)
- Background dominates object
- Object partially visible
- Class boundaries visually ambiguous

This confirms that errors are **data‑driven**, not random noise.

---

## 8. Conv1 Filter Visualization
![Conv1 Filters](outputs/plots/conv1_filters.png)

**What this shows**
- First‑layer filters learn:
  - Color contrasts
  - Edge detectors
  - Simple texture patterns
- Filters are not semantic; they capture low‑level statistics

This matches classical CNN theory.

---

## 9. Feature Map Visualization
Feature maps were extracted after each convolutional layer.

**Progression**
- Early layers → edges & color blobs
- Middle layers → object parts
- Deeper layers → sparse, abstract activations

Feature abstraction increases with depth.

---

## 10. Saliency Map Visualization
![Saliency Map](outputs/plots/saliency.png)

**Method**
- Gradient of correct class score w.r.t. input image
- Highlights pixels that most influence prediction

**Interpretation**
- High activation on object body
- Background mostly ignored
- Confirms model attends to meaningful regions

---

## 11. Key Takeaways
- CNNs build representations hierarchically
- Visualization validates learned behavior
- Errors are interpretable and reasonable
- Model decisions align with visual intuition

---

## 12. Limitations
- Simple CNN architecture
- Low image resolution
- No advanced explainability methods (Grad‑CAM, Integrated Gradients)

---

## 13. Future Work
- Grad‑CAM & Grad‑CAM++
- Compare shallow vs deep architectures
- Apply to higher‑resolution datasets
- Extend to object detection pipelines

---

## 14. Conclusion
This project demonstrates **end‑to‑end interpretability** of a CNN using PyTorch.  
Rather than treating neural networks as black boxes, we expose and analyze their internal logic in a structured, reproducible way.
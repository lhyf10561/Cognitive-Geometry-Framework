<img width="610" height="452" alt="image" src="https://github.com/user-attachments/assets/4eaf776d-beef-44b0-b04c-c2a4b5365105" /># 🧩 Cognitive Geometry Framework  
**Author:** Huayifu Lv  
**Manuscript:** *A unified theoretical framework for the geometry of cognitive dynamics* (under review at *Nature*)  
**License:** MIT  

---

## 🌐 Overview
This repository contains all minimal simulation scripts used in the manuscript.  
Each script corresponds to one main figure (Fig. 1–3) and can be run independently in **Spyder** or any Python IDE.  
The model unifies 0D (point), 1D (ring), and 2D (spherical) attractors within a single resource-conserving dynamical system, known as the **Camp–Citadel paradigm**.

---

## ▶️ How to Run
- Each figure has its own `.py` file, e.g.:
  - `figure1_unified.py`
  - `figure2_switching.py`
  - `figure3_collapse.py`
- Open any of them in **Spyder**.
- Run the file directly — it will generate the corresponding figure.
- Optional: `fig1_convergence.html` provides an interactive view for the convergence process of Figure 1.

No additional configuration or wrapping script is required.  
All codes are fully commented and self-contained.

---

## 🧠 Model Summary
The core dynamics follow:
\[
\frac{dX_k}{dt}
= \sum_{i \neq k} (S_{ki} X_i - S_{ik} X_k)
+ \sum_{j \neq k} (P_{kj} X_k - P_{jk} X_j)
\]
with total resource conservation  
\[
\sum_i X_i = C_{\text{total}}
\]

Topological rules:
- **Inter-Camp competition** → determines the **number** of attractors  
- **Intra-Camp (Citadel) independence** → determines the **dimension** of each attractor  
\[
\boxed{D = N_{\text{citadel}} - 1}
\]

---

## 📁 Folder Layout
Cognitive-Geometry-Framework/
├── figure1_unified.py
├── figure2_switching.py
├── figure3_collapse.py
├── fig1_convergence.html
├── figures/ # Output images
├── requirements.txt
└── README.md


---

## 📈 Requirements
Minimal dependencies:

numpy
matplotlib
scikit-learn


---

## 📘 Citation
> **Lv, H. (2025).** *A unified theoretical framework for the geometry of cognitive dynamics.*  
> Submitted to *Nature.*  
> Code available at [https://github.com/lhyf10561/Cognitive-Geometry-Framework](https://github.com/lhyf10561/Cognitive-Geometry-Framework)

---

## 📜 License
MIT License © 2025 Huayifu Lv

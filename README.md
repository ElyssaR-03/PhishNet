# 🎓 PhishNet: A Machine Learning-Based Phishing Detection and Simulation System

## Overview
**PhishNet** is a cybersecurity education and research platform designed to simulate phishing detection and visualize classification results. It combines phishing dataset analysis, machine learning models, and an interactive dashboard to demonstrate how phishing attacks are identified and mitigated.

---

## 🎯 Purpose
The system supports cybersecurity education by showing how machine learning can classify phishing URLs and emails. It also provides a safe environment to test and understand phishing detection mechanisms — without human subject testing.

---

## 🧩 Features
- **Phishing Detection Engine** using ML algorithms (SVM, Random Forest, Logistic Regression).  
- **Web Dashboard (React + TailwindCSS)** for real-time analysis and result visualization.  
- **Educational Mode** that explains why samples are classified as phishing.  
- **FastAPI Backend** for feature extraction, model inference, and data handling.  
- **SQLite/PostgreSQL Database** for storing logs, results, and metadata.  
- **UML Design Documentation** (Use Case, Class, and Activity diagrams).  
- **Comprehensive Unit Testing** for all system components.

---

## 🧠 Research Focus
This project serves as the computational component for a thesis exploring **phishing detection and mitigation through classification algorithms**. It investigates:
- What features best predict phishing behavior?
- Which algorithms perform most accurately on phishing datasets?
- How visualization improves understanding of phishing detection.

---

## 🧪 Architecture
**Frontend:** React, TailwindCSS  
**Backend:** FastAPI (Python)  
**ML Stack:** scikit-learn, pandas, NumPy  
**Database:** SQLite (development) → PostgreSQL (production)  
**Testing:** PyTest, Jest  
**Visualization:** Chart.js, Plotly  
**Documentation:** UML (PlantUML), Markdown

---

## ⚙️ UML Design
- **Use Case Diagram:** Illustrates user interactions (e.g., upload, classify, view results).  
- **Class Diagram:** Defines relationships among system modules (model, preprocessing, database).  
- **Activity Diagram:** Describes the data flow from input to classification.

---

## ✅ Unit Testing
Every functional component includes unit tests:
- Input validation  
- URL feature extraction  
- Model prediction  
- Result visualization  

Coverage reports ensure reliability and traceability for research replication.

---

## 📚 Datasets
The system integrates phishing datasets such as:
- **PhishTank**
- **UCI Machine Learning Repository Phishing Dataset**
- **Kaggle Phishing URLs Dataset**

Each dataset is preprocessed for balanced representation and feature extraction.

---

## 📈 Research Contributions
- Implements a reproducible phishing detection framework.
- Demonstrates classification accuracy and feature importance.
- Provides an educational visualization layer for cybersecurity awareness.

---

## 🧩 Future Work
- Integrate explainable AI (XAI) components.  
- Expand to real-time phishing URL monitoring.  
- Include comparative visual analytics for multiple classifiers.

---

## 🧾 Citation
If you use this project for research, please cite it as:

> Ratliff, E. (2025). *PhishScope: A Machine Learning-Based Phishing Detection and Simulation System for Cybersecurity Education.* Tennessee State University.

---

## 📜 License
MIT License © 2025 Elyssa Ratliff

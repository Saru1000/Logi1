# FastTrack Logistics Streamlit Dashboard

This repository contains an AI‑driven analytics dashboard for **FastTrack Logistics**.
The dashboard provides:

1. **Data Visualisation** – 10+ descriptive insights with interactive filters  
2. **Classification** – KNN, Decision Tree, Random Forest, Gradient Boosting  
3. **Clustering** – K‑Means with dynamic cluster slider and elbow chart  
4. **Association Rule Mining** – Apriori with configurable thresholds  
5. **Regression** – Linear, Ridge, Lasso, Decision Tree regressor  

## 📂 File Structure
```
.
├── app.py
├── fasttrack_logistics_synthetic.csv
├── requirements.txt
└── README.md
```

## 🚀 Quick Start (Local)
```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Run Streamlit
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud
1. Fork / upload this repo to your GitHub account  
2. Go to **https://share.streamlit.io/** and link your repo  
3. Set **main file** to `app.py` and deploy  
4. (Optional) Add environment variables / secrets in **Advanced settings**  

## 📊 Dataset
The included CSV (`fasttrack_logistics_synthetic.csv`) contains 50,000 synthetic survey responses
designed for market viability, operational and financial analysis.  
You can replace this file with a newer dataset of identical schema if needed.
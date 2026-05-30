# 🔄 Closed‑Loop ML: Auto Retrain & Rollback

[![CI](https://github.com/kallurayaankit/closed-loop-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/kallurayaankit/closed-loop-ml/actions/workflows/ci.yml)

A fully automated MLOps system that monitors live prediction errors, triggers retraining when performance degrades, and safely promotes or rolls back the new model — all without human intervention.

---

## 📌 Features

- **Real‑time error monitoring** – computes MAE on live data
- **Performance‑based trigger** – automatically starts retraining when MAE exceeds a threshold
- **Champion/challenger evaluation** – compares the new model against the current one
- **Safe promotion / rollback** – promotes the challenger only if it's strictly better
- **Dockerized** – entire pipeline runs in a single container
- **Ready for orchestration** – can be wired to Prometheus, Argo, or Kubeflow

---

## 📁 Project Structure

---

## ⚡ Quick Start (Local)

### 1. Set up environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/generate_data.py
python src/train.py
python src/monitor.py
python src/orchestrator.py
docker-compose up --build

Current champion MAE: 5.13
🚨 ALERT: Performance degradation detected! Starting retraining...
Challenger MAE: 5.13
Challenger rejected. Keeping current champion.


---

### 📤 How to add it

```cmd
git add README.md
git commit -m "Add comprehensive README"
git push

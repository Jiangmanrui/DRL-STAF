# DRL-STAF

This repository contains the code and data for **"DRL-STAF: A Deep Reinforcement Learning Framework for State-aware Forecasting of Complex Multivariate Hidden Markov Process"**.

## Repository Structure

- **sim_chosmm/**  
  Code for generating simulated datasets based on CHOSMM dynamics.  

- **data/**  
  Folder for storing datasets used in experiments (both simulated and real-world).  

- **TRODITIONAL/**  
  Implementations of traditional baselines:  
  - Parallel HMM  
  - Parallel HSMM  
  - Parallel HOHMM  
  - CHMM  

- **Stage_one/**  
  Training code for Stage One of DRL-STAF, where each variable is modeled independently with a DRL-based estimator.  
  - **results/**: Demo results of Stage One training.  

- **Stage_two/**  
  Training code for Stage Two of DRL-STAF, where estimators are integrated via graph-based coordination to capture cross-variable interactions.  
  - **results/**: Demo results of Stage Two training.  


## Environment Setup

This project was developed and tested under the following environment:

- **Operating System:** Windows 10 / 11 (64-bit)
- **Python Version:** 3.10

### Required Dependencies

The core third-party libraries used in the project are:

- torch==2.4.0
- numpy==1.26.4
- pandas==2.3.1
- gym==0.25.2
- matplotlib==3.9.2
- tqdm==4.65.0

Install all dependencies via:

```bash
pip install -r requirements.txt

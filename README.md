# DRL-STAF

This repository contains the code and data for "DRL-STAF: A Deep Reinforcement Learning Framework for State-aware Forecasting of Complex Multivariate Hidden Markov Process".

Repository Structure

**sim_chosmm**
Code for generating simulated datasets based on CHOSMM dynamics.

**data/**
Folder for storing datasets used in experiments (both simulated and real-world).

**TRODITIONAL/**
Implementations of traditional baselines:

- Parallel HMM

- Parallel HSMM

- Parallel HOHMM

- CHMM

**Stage_one/**
Training code for Stage One of DRL-STAF, where each variable is modeled independently with a DRL-based estimator.

**Stage_two/**
Training code for Stage Two of DRL-STAF, where estimators are integrated via graph-based coordination to capture cross-variable interactions.
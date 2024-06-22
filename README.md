# Transdimensional Predictive Least Squares (TPLS)

Note: MATLAB implementation (including reproducible figures) at [tpls_matlab](https://github.com/marija-iloska/tpls_matlab).

This code is a Python implementation of the algorithm TPLS proposed in our paper "Transdimensional Model Learning with Online Feature Selection based on Predictive Least Squares".
We provide an example code on how a user can run TPLS, as well as pre-coded feature bar plots. The code to reproduce the experiments presented in our paper is only available in MATLAB.

## Introduction
TPLS is a distribution-free online feature selection algorithm that is completely based on least squares (LS). With new data arrivals, TPLS recursively updates the parameter estimate not only
in value, but in dimension as well. What makes TPLS unique is the ability to recursively move up and down model dimension, and the fact that it uses the predictive (instead of the fitting) error as its criterium whether to add or remove features.
Specifically, the foundations of TPLS are recursive LS [(RLS)](https://dl.acm.org/doi/book/10.5555/151045), order recursive LS [(ORLS)](https://dl.acm.org/doi/book/10.5555/151045) and predictive LS [(PLS)](https://academic.oup.com/imamci/article-abstract/3/2-3/211/660741).

## How to Use Code

### How to run TPLS
Jupyter Notebook to run: <br/>
example_code.ipynb - a script that demonstrates how to call and run TPLS. It includes feature bar plots and predictive error. For the interested user, we demo how to compute the MSE [regret analysis](https://pubsonline.informs.org/doi/abs/10.1287/opre.30.5.961) calculations and plots. <br/> 
<br/> 

## About the Code
### LS_updates.py
A module which contains all LS related updates, including MSE and predictive error calculations. <br/> 
<br/>
Class RLS - updates the model recursively with new data point.  <br/> 
Attributes: ascend() <br/>
<br/>
Class ORLS - updates the model recursively in dimension (up/down --> add/remove feature) based on user's choice. <br/> 
Attributes: ascend(), descend() <br/>
<br/>
Class PredError - computes the predictive error for the present model. <br/>
Attributes: compute() <br/>
<br/>
Class Expectations - computes the MSE differnce between true model and neighbor model (up/down) for all given features. It can compute single time instant MSE and batch. <br/>
Attributes: model_up(), model_down(), batch() <br/>
<br/>

### algorithms.py
A module of two algorithms:  <br/> 
Class ModelJump - From the present model, loops over all features (add or remove) and computes the predictive error for each proposed model. <br/>
Attributes: up(), down(), stay() <br/>
<br/>
Class TPLS - Implements the final JPLS algorithm for one time step, using ModelJump and all in LS_updates.py.
Attributes: model_update(), time_update() <br/>
<br/>

### util.py
Helper functions for generating synthetic data, initialization, bar plotting, extracting feature indices, and finding min index. <br/>
Attributes: generate_data(), initialize(), bar_plot(), get_features(), get_min().














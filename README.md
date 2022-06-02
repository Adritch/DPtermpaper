# Term Paper

**Group members:**
- Silke Kofoed Christiansen
- Emil Bille
- Carl Adrian Møller

This repository contains  
1. Output folder - Results from running simulations.ipynb
2. Results.ipynb - Notebook where all results are displayed (loaded from data folder)
3. simulations.ipynb - Notebook used to generate results based on the optionpricingclass and utility functions
4. optionpricing - Contains the OptionPricingClass
5. utility.py - various functions used to generate the results
6. How the LSM works.ipynb - Notebook illustrating how the LSM algorithm works

# Valuing American Options by Simulation: Least Squares Monte Carlo

Our project is titled **Valuing American Options by Simulation: Least Squares Monte Carlo**. We investigate the well known Least Squares monte carlo approach to valuing american type options. We manage to reproduce their original results well, given we dont have their seed number. In order to reduce computational speed and memory requirement for large simulations, we implement a brownian bridge simulation technique into the original LSM algorithm. We improve computational speed greatly, and also obtain robust results.

The **results** of the project can be seen from running [Results.ipynb](Results.ipynb).

**Dependencies:** Apart from a standard Anaconda Python 3 installation, the project requires the following installations:

``pip install chaospy``
``pip install pathlib``

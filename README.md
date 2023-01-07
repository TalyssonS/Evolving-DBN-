# Evolving-DBN by an analytical threshold
This project presents the use of Evolving Dynamic Bayesian Networks for $CO_2$ Emissions Forecasting in Multi-Source Power Generation Systems of Different Countries.

# Dependencies
All dependencies are in requirements.txt.

Using anaconda it is possile create the enviroment using this command: conda create --name myenv --file requirements.txt

One additional functionality regarding PGMPY is the AIC score function. This additional functionality has already been merged in PGMPY official GitHub: https://github.com/pgmpy/pgmpy/tree/dev.
However, perhaps this new score option is not yet available on the current version of the package. In this case, the scripts altered are available in https://github.com/TalyssonS/Evolving-DBN-/tree/master/PGMPY%20-%20AIC%20score%20creation.

# Project Description
- **data_download.ipynb:** This jupyter notebook code presents the script to download data from the ENTSO-E platform via API. 

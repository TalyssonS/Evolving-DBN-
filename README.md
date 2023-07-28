# Evolving-DBN by an analytical threshold
This project presents the use of Evolving Dynamic Bayesian Networks for $CO_2$ Emissions Forecasting in Multi-Source Power Generation Systems of Different Countries.

![8014graphical_abstract](https://github.com/TalyssonS/Evolving-DBN-/assets/41801745/edf6d6c3-3c3a-4410-b4a6-4d06a8429e32)


# Dependencies
All dependencies are in requirements.txt.

Using anaconda it is possile create the enviroment using this command: conda create --name myenv --file requirements.txt

One additional functionality regarding PGMPY is the AIC score function. This additional functionality has already been merged in PGMPY official GitHub: https://github.com/pgmpy/pgmpy/tree/dev.
However, perhaps this new score option is not yet available on the current version of the package. In this case, the scripts altered are available in https://github.com/TalyssonS/Evolving-DBN-/tree/master/PGMPY%20-%20AIC%20score%20creation.

# Project Description
- **/original_datasets:** Original datasets of multi-source power generation systems of Belgium, Germany, Portugal and Spain. The dataset of each country comprises records from January 1, 2019 to December 31, 2021 with a one-hour sampling rate. The information is also available on schema "original_dataset" on the dump of database postgresql.
- **data_pre_processing.ipynb:** Script developed using jupyter notebook responsible to realise data pre-processing. This script reads the original information from /original_datasets and saves the outputs in /pre_processed_datasets. The information is also saved on schema "pre_processed_dataset" on the dump of database postgresql. Some figures are saved on **/figures** to illustrate steps of data_pre_processing.
- **edbn_emissions_forecast.py:** Script developed in python responsible to realise emissions forecasting using the proposed evolving DBN by analytical threshold. The dataset used is the dataset already pre-processed. The results are saved on schema results on postgresql database.
- **dbn_onestep_emissions_forecast.py:** Script developed in python responsible to realise emissions forecasting using the tradicional DBN. The dataset used is the dataset already pre-processed. The results are saved on schema results on postgresql database.
- **ann_emissions_forecast.py:** Script developed in python responsible to realise emissions forecasting using the ANN. The dataset used is the dataset already pre-processed. The results are saved on schema results on postgresql database.
- **xgboost_emission_forecast.py:** Script developed in python responsible to realise emissions forecasting using the XgBoost. The dataset used is the dataset already pre-processed. The results are saved on schema results on postgresql database.
- **results.ipynb** Script developed using jupyter notebook responsible to realise results apuration of all methods during $CO_2$ emissions forecasting of the multi-source power generation system of Belgium, Germany, Portugal and Spain. The script uses as input the information available on the schema results of the PostgreSQL database. This script saves figures to illustrate the results on **/figures**.

# Instructions for running the simulations
After installing all packages (available in the requirements.txt), the scripts can be used in the standard format of any Python script. **data_pre_processing.ipynb** and **results.ipynb** were developed using the Jupyter Notebook interface for Python. The other scripts were implemented as a Python archive (.py).

# SensorML Uncertainty Analysis #
This repository contains the code the analyze time series data with metrological information encoded in a SensorML file (JSON encoding). The code is shipped with an example of a SBE16 CTD instrument SensorML description and a CSV file containing data acquired at the [OBSEA underwater observatory](https://obsea.es) by the same instrument. This repository uses the OGC Connected Systems standard candidate as a git submodule to validate SensorML documents against its JSON schema.

## Requriments ##

To run this code python3, pip and git need to be installed. 

## Setup ##
1. Clone the repository

```bash
git clone --recurse-submodules https://github.com/obsea-upc/sensorml-uncertainty-analysis
```  

2. install dependencies 
```bash
cd sensorml-uncertainty-analysis
pip install -r requirements.txt # for linux/mac
py -m pip -r requirements.txt   # for windows
```
3. Run the data analysis with the default SensorML and CSV files 
```bash
python uncertainty_analysis.py SBE16_SensorML.json OBSEA_SBE16_CTD_30min_2017_2019.csv
```


## Contact info ##

* **author**: Enoc Martínez  
* **version**: v0.1    
* **organization**: Universitat Politècnica de Catalunya (UPC)  
* **contact**: enoc.martinez@upc.edu  


## Acknowledgements ##
This work has been funded by the [MINKE](https://minke.eu) project funded by the European Commission within the Horizon 2020 Programme (2014-2020)
Grant Agreement No. 101008724.


![logo-minke.png](docs/pics/logos.png)
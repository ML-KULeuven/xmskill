# xMskill

The application of xG for evaluating finishing skill in soccer remains a subject of debate. Players often struggle to consistently outperform their cumulative xG, raising questions about the reliability of this metric. In this analysis, we aim to confront the limitations and nuances surrounding the evaluation of finishing skill using xG statistics through simulation studies. 

## Datasets

We use two public datasets.

### FBref shot data

The dataset contains shooting and shot creation event data for the Big Five European leagues over a period of five seasons (2017-2018 until 2021-2022). The data includes who took the shot, when, with which body part and from how far away. Additionally, the player creating the chance and also the creation before this are included in the data. This dataset was scraped from FBref and can be obtained by running the ["01 EDA FBref shots" notebook](01-eda-fbref-shots.ipynb).

###  Open-source event data

We use public Statsbomb data to compute shot distributions and train xG models. The data can be downloaded from <https://github.com/statsbomb/open-data> for public non-commercial use. The ["02 create StatsBomb dataset" notebook](02-create-statsbomb-dataset.ipynb) can be run to prepare this dataset.

## Experiments

We use our [soccer-xg](https://github.com/ML-KULeuven/soccer_xg) Python library to train a logistic regression xG model. The following notebooks provide instructons for preprocessing the data and training the models:

- [Prepare StatsBomb data](https://github.com/ML-KULeuven/soccer_xg/blob/master/notebooks/1-load-and-convert-statsbomb-data.ipynb)
- [Prepare Wyscout data](https://github.com/ML-KULeuven/soccer_xg/blob/master/notebooks/1-load-and-convert-wyscout-data.ipynb)
- [Train model](https://github.com/ML-KULeuven/soccer_xg/blob/master/notebooks/2-basic-usage.ipynb)

Code for reproducing the simulation experiments will be released later.

## License

Copyright (c) DTAI - KU Leuven – All rights reserved.

All resources in this repository are released under the terms of the Apache License, Version 2.0;
you may not use these resources except in compliance with the License.

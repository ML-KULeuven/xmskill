# xMskill

The application of xG for evaluating finishing skill in soccer remains a subject of debate. Players often struggle to consistently outperform their cumulative xG, raising questions about the reliability of this metric. In this analysis, we aim to confront the limitations and nuances surrounding the evaluation of finishing skill using xG statistics. 

## Datasets

We use two public datasets.

### FBref shot data

The dataset contains shooting and shot creation event data for the top-5 European leagues over a period of five seasons (2017-2018 until 2021-2022). The data includes who took the shot, when, with which body part and from how far away. Additionally, the player creating the chance and also the creation before this are included in the data. This dataset was scraped from FBref and can be obtained by running the [01-EDA_fbref_shots notebook](01-EDA_fbref_shots.ipynb).

###  StatsBomb open data

We use public Statsbomb data to compute shot distributions and train xG models. The data can be downloaded from <https://github.com/statsbomb/open-data> for public non-commercial use. 

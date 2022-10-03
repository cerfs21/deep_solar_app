### Deep Solar App.

This project provides a demo application to assist public and private entities in exploring the market of solar energy, particularly the one of deploying solar panels on residential and corporate premises in a given region.

Based on Machine Learning (ML) applied to a Kaggle dataset(1), the proposed Python application delivers the following functions:
    • select a region and provide the total solar panel area already deployed in it;
    • predict the probable volume of additional sales in this region;
    • provide the possibility to adjust model input values to amend the prediction; 
    • log minimal application usage by recording user identification and connections.

This appplication was developped in two steps:

1) Select and build the best Machine Learning model to predict a target feature (solar panel area per capita) given a limited number of input features selected with Lasso Linear regression. The resulting model happened to be a Random Forest Regressor. For better performance, the computation to evaluate different regression models was executed within an Intel container optimized for Machine Learning(2). 
2) Develop a web application to perform the functions listed above. The front-end was programmed using Dash to provide a responsive user interface. The back-end tasks mainly consist in getting user input and invoking the ML model to compute a prediction, while the connection log is on SQLite.


    1. DeepSolar Dataset | Kaggle : https://www.kaggle.com/datasets/tunguz/deep-solar-dataset
    2. intel/intel-optimized-ml - Docker Image : https://hub.docker.com/r/intel/intel-optimized-ml

Deep Solar App.

The purpose of this project is to provide a demonstrator for an application to assist public and private entities in exploring the market of solar energy, particularly the one of deploying solar panels on residential and corporate premises in a given region.

Based on Machine Learning (ML) applied to a Kaggle dataset(1), the proposed Python application performs the following functions:
    • select a region and provide the total solar panel area already deployed in it;
    • predict the probable volume of additional sales in this region;
    • provide the possibility to adjust model input values to amend the prediction; 
    • log minimal application usage by recording user identification and connections.

The first part of this project consisted in selecting and building the best Machine Learning model to predict a target feature (solar panel area per capita) in a region selected, given a limited number of input features selected with Lasso Linear regression. The resulting choice happened to be Random Forest Regressor, the resulting model being saved for later use by the application. For better performance, the computation to evaluate different regression models was executed within an Intel container optimized for Machine Learning(2). 

The second part of this project consisted in developing the application to perform the functions listed above. The front-end was programmed using Dash to provide a responsive user interface. The back-end tasks mainly consisted in getting user input and invoking the previously saved model to compute a prediction, while the connection log was based on SQLite.


    1. DeepSolar Dataset | Kaggle : https://www.kaggle.com/datasets/tunguz/deep-solar-dataset
    2. intel/intel-optimized-ml - Docker Image : https://hub.docker.com/r/intel/intel-optimized-ml

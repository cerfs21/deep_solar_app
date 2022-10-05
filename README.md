### Deep Solar App.

This project provides a demo application to explore the market of solar energy, particularly the one of deploying solar panels on residential and corporate premises in a given region.

Based on Machine Learning (ML) applied to a Kaggle dataset[^first], the proposed Python application delivers the following functions:
+ select a region and provide information on the installed base;
+ predict the probable volume of additional sales in this region;
+ provide the possibility to adjust model feature values to amend the prediction; 
+ log minimal application usage by recording user identification and connections.

The project is carried out in two phases:
1) Select and build the best Machine Learning model to predict a target feature (solar panel area per capita) given a limited number of input features selected with Lasso. The resulting model happens to be an Hist-Gradient Boosting Regressor. For better performance, the computation to evaluate different regression models is executed within an Intel container optimized for Machine Learning[^second]. 
2) Develop a web application to perform the functions listed above. The front-end is built on Dash to provide a responsive user interface. The back-end tasks mainly consists in getting user input and invoking the ML model to compute a prediction, while the connection log is based on SQLAlchemy and SQLite.

[^first]: DeepSolar Dataset | Kaggle : https://www.kaggle.com/datasets/tunguz/deep-solar-dataset
[^second]: Intel/Intel-optimized-ML - Docker Image : https://hub.docker.com/r/intel/intel-optimized-ml
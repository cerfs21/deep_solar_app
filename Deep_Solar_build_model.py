# Deep_Solar_build_model v3.3
#   code refactoring of model optimization and metric plotting
#   additional comments

# Import

## Import Python librairies
import os, pickle
import numpy as np
import pandas as pd
import seaborn as sns
from math import sqrt
from tqdm.auto import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

## Import dataset
df_src = pd.read_csv('./data/deepsolar_tract.csv', encoding = "ISO-8859-1")
# encoding='utf-8', dtype({'fips':'str'})
# Original dataset kept in df_src while working DataFrame is df
df = df_src.copy()


# Common functions for model optimization and metric plotting

def predict_and_get_metrics(model, eval_set_type, X_eval, y_eval):

    '''
    This function applies the predict method to a given model and provides Score, MAE and MSE for a given evaluation set.
    
        Args:
            Fitted model
            Type of evaluation set: train for training, val for validation, test for testing, full for complete dataset
            X and y evaluation sets which may be normalized
        
        Returns:
            Prediction based on X evaluation set (done within the function)
            MAE, MSE and Score results (output within the function)
    '''


    pred = model.predict(X_eval)
    print(f"Score {eval_set_type} :", model.score(X_eval, y_eval))
    print(f"MAE   {eval_set_type} :", mae(y_eval,pred))
    print(f"MSE   {eval_set_type} :", mse(y_eval,pred))
    return


def look_for_best_score(model_long, model_type, X_train, y_train, X_val, y_val, hyperparam, hyperp_min, hyperp_max):

    '''
    This function looks for the hyperparamater value getting the best model score.
    
        Args:
            Model name in long format
            Type of regression model
            Training and validation sets which may be normalized
            Hyperparameter to optimize
            Range of values (defined by min and max) in which to look for best hyperpameter value
        
        Returns:
            Optimization results (output within the function)
            Fitted Model and hyperparameter value getting the best score
            Lists of MAE, MSE and Score results in the range of hyperparameter values
    '''

    print(f"{model_long} pour {hyperparam} = {n_min}~{n_max} :")

    # Initialize lists of MAE, MSE and Score results for a range of hyperparameter values
    mae_list = []
    mse_list = []
    score_list = []
    # Initialize best hyperparameter and score values 
    n_better = 0
    score_better = 0
    for n in tqdm(range(hyperp_min, hyperp_max+1)):
        # Initialize hyperparameters according to model type
        if  model_long == 'K-Neighbors Regressor':
            model = model_type(n_neighbors=n, n_jobs=-1)
        elif model_long == 'Gradient Boosting Regressor':
            model = model_type(n_estimators=n)
        elif model_long == 'Random Forest Regressor':
            model = model_type(n_estimators=n, n_jobs=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        # Feed MAE, MSE and Score lists for later plotting
        mae_list.append(mae(y_val,pred))
        mse_list.append(mse(y_val,pred))
        score = model.score(X_val, y_val)
        score_list.append(score)
        # Store best model and corresponding hyperparameter value 
        if score > score_better:
            score_better = score
            n_better = n
            model_better = model
    
    # Output optimization results
    print(f"Meilleure valeur de {hyperparam} :", n_better)
    print(f"Score val :", score_list[n_better - hyperp_min])
    print(f"MAE   val :", mae_list[n_better - hyperp_min])
    print(f"MSE   val :", mse_list[n_better - hyperp_min])

    return (model_better, n_better, score_list, mae_list, mse_list)


def plot_metric_2D (metric, metric_list, eval_set_type, model_long, model_short, hyperparam, hyperp_min, hyperp_max):
    '''
    This function plots a metric according to a range of hyperparameter values.
    
        Args:
            Name of Metric to plot: MAE, MSE or Score
            List of metric values returned from model optimization
            Type of evaluation set: train for training, val for validation
            Model name in long and short formats
            Hyperparameter to plot
            Range of hyperparameter values (defined by min and max)
        
        Returns:
            Plots and saves a figure within the function
    '''
    
    # Use function args to define figure title and labels
    plt.title(f'{model_long} - {metric} {eval_set_type}', fontsize=16)
    plt.xlabel(f'${hyperparam}$', fontsize=14)
    plt.ylabel(f'${metric}$', fontsize=14)
    # Plot x= hyperparameter, y = metric
    plt.plot([i for i in range(hyperp_min,hyperp_max+1)], metric_list, markersize=6)
    # Save figure named by model short name, plotted metric and range hyperparam values
    plt.savefig(f'{model_short}_{metric}_{str(n_min)}-{str(n_max)}')
    plt.clf()
    return


# Explore dataset

## Overview
print()
print("*** Exploration des données ***")
print(df.shape)
print()
print(df.info())

## Identify features

### Identify possible target features
solar_features = ['tile_count','tile_count_residential','tile_count_nonresidential','solar_system_count','solar_system_count_residential','solar_system_count_nonresidential','total_panel_area','total_panel_area_residential','total_panel_area_nonresidential','number_of_solar_system_per_household','heating_fuel_solar_rate','solar_panel_area_divided_by_area','solar_panel_area_per_capita','heating_fuel_solar']

### Identify geographical features while significant model features are climate and socioeconomic ones
geographical_features = ['county','state','fips','lon','lat']

### Identify information collection of which may be uncertain or illegal
political_features = ['voting_2016_dem_percentage','voting_2016_gop_percentage','voting_2016_dem_win','voting_2012_dem_percentage','voting_2012_gop_percentage','voting_2012_dem_win']

education_features = ['education_bachelor','education_college','education_doctoral','education_high_school_graduate','education_less_than_high_school','education_master','education_population','education_professional_school','education_less_than_high_school_rate','education_high_school_graduate_rate','education_college_rate','education_bachelor_rate','education_master_rate','education_professional_school_rate','education_doctoral_rate']

race_features = ['race_asian','race_black_africa','race_indian_alaska','race_islander','race_other','race_two_more','race_white','race_asian_rate','race_other_rate','race_two_more_rate']

age_features = ['age_median','age_18_24_rate','age_25_34_rate','age_more_than_85_rate','age_75_84_rate','age_35_44_rate','age_45_54_rate','age_65_74_rate','age_55_64_rate','age_10_14_rate','age_15_17_rate','age_5_9_rate']

### Detect non numerical features (Python objects) and output their content, to be ignored
print()
print("*** Colonnes autres que numériques et aperçu de leur contenu ***")
for col in df.columns:
    if df[col].dtype == 'O':
        print(col, df[col].unique(), '\n')
misc_features = ['county','state','electricity_price_transportation']   # 'county' and 'state' already identified as geographical_features

### index column, to be ignored
misc_features.append('Unnamed: 0')

### Gini index column, to be ignored
misc_features.append('gini_index')


# Preprocessing

## Select target feature and ignore irrelevant data

### Target feature
target = 'solar_panel_area_per_capita'  # Target feature retirée des solar_features avant leur suppression en tant qu'input du modèle de prédiction
solar_features.remove(target)

### Remove features to be ignored as mentioned above
features_removed = solar_features + geographical_features + political_features + education_features + race_features + age_features + misc_features
df.drop(features_removed, axis=1, inplace=True)

### Remove lines where data is incomplete
df.replace([np.inf, -np.inf], np.nan, inplace=True) # inf values replaced with NaN
df_target_noNaN = df.dropna(subset=[target])    # Create DataFrame excluding lines where target feature value is NaN
df.dropna(inplace=True) # Remove lines where at least one feature value is NaN

### Separate input X and target y
X = df.drop(columns=target)
y = df[target]


# Select features using Linear Regression

## Normalize using StandardScaler
from sklearn.preprocessing import StandardScaler

### Normalize X
scaler_X = StandardScaler()
scaler_X.fit(X)
X_norm = pd.DataFrame(scaler_X.transform(X), columns = X.columns)
X_norm.head()

### Normalize y
scaler_y = StandardScaler()
scaler_y.fit(y.to_frame())
y_norm = scaler_y.transform(y.to_frame())
y_norm

## Select features using Lasso
from sklearn.linear_model import Lasso

### alpha coefficient chosen to select 10 features approximately
select_algo = Lasso(alpha=0.09)

print()
print("*** Selection de features avec Lasso ***")
dict_feat = defaultdict()
select_features = []
selector = select_algo.fit(X_norm, y_norm)
for i, col in enumerate(X_norm.columns):
    if selector.coef_[i] != 0:
        print(col, round(selector.coef_[i], 3))
        dict_feat[col]=selector.coef_[i]
        select_features.append(col)

### Display features ordered by increasing Lasso coefficient
print()
print("Features sélectionnées par Lasso par ordre de coefficient croissant :")
print({k: v for k, v in sorted(dict_feat.items(), key=lambda x: x[1])})
print()
print("Simple liste des features sélectionnées par Lasso :")
print(select_features)

### Select significant features mainly based on Lasso results
input_features = ['median_household_income','electricity_price_commercial','electricity_price_industrial','housing_unit_median_gross_rent','frost_days','relative_humidity','daily_solar_radiation','incentive_count_residential',
'incentive_nonresidential_state_level']

print()
print("Features finalement retenues :")
print(input_features)
X_sel = X[input_features]


# Modelization

## Get reproducible results in train_test_split and certain regression algorithms
np.random.seed(14071789)

## Create train, val et test sets excluding NaN values

from sklearn.model_selection import train_test_split

### Extract static test set = 20% from selected dataset
X_train_val, X_test, y_train_val, y_test = train_test_split(X_sel, y, test_size=0.2, shuffle=True)

### Extract training and validation sets = respectively 60% and 20% from selected dataset
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, shuffle=True)

## Create train, val et test sets allowing NaN values for non target features
df_full = df_target_noNaN[input_features].join(df_target_noNaN[target])

print()
print("*** DataFrame excluant seulement les lignes où la feature à prédire vaut NaN ***")
print(df_full.shape)
print()
print(df_full.info())
print()
print("*** DataFrame de travail excluant les lignes où au moins une feature vaut NaN  ***")
print(X_sel.join(y).shape)
print()
print(X_sel.join(y).info())

### Extract static test set = 20% from dataset allowing NaN values for non target features
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(df_full[input_features], df_full[target], test_size=0.2, shuffle=True)

### Extract training and validation sets = respectively 60% and 20% from dataset allowing NaN values for non target features
X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(X_train_full, y_train_full, test_size=0.25, shuffle=True)


## Normalization for regression models needing it

scaler_X = StandardScaler()
scaler_X.fit(X_train)

scaler_y = StandardScaler()
scaler_y.fit(y_train.to_frame())

### Normalize X_train and y_train
X_train_norm = pd.DataFrame(scaler_X.transform(X_train), columns = X_train.columns)
y_train_norm = scaler_y.transform(y_train.to_frame())

### Normalize X_val and y_val
X_val_norm = pd.DataFrame(scaler_X.transform(X_val), columns = X_val.columns)
y_val_norm = scaler_y.transform(y_val.to_frame())


## Build, Train and Evaluate various regression models

print()
print("*** Construction, entraînement et évaluation de différents modèles de régression ***")

### Evaluate Linear Regression
from sklearn.linear_model import LinearRegression
print()
print("Linear Regression :")

LinReg = LinearRegression()
LinReg.fit(X_train_norm, y_train_norm)
predict_and_get_metrics(LinReg, 'train', X_train_norm, y_train_norm)
predict_and_get_metrics(LinReg, 'val', X_val_norm, y_val_norm)


### Evaluate Support Vector Regression (Radial Basis Function)
from sklearn import svm
print()
print("Support Vector Machines Regression (kernel = rbf) :")

SVR_r = svm.SVR(kernel='rbf')
SVR_r.fit(X_train_norm, np.ravel(y_train_norm))
predict_and_get_metrics(SVR_r, 'train', X_train_norm, y_train_norm)
predict_and_get_metrics(SVR_r, 'val', X_val_norm, y_val_norm)


### Evaluate KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor

#### Search interval for n_estimator
n_min = 10
n_max = 25
print()

#### Evaluate K-Neighbors Regressor by iteration over n-neighbors and output results
KNR_better, K_better, score_list, mae_list, mse_list = look_for_best_score('K-Neighbors Regressor', \
    KNeighborsRegressor, X_train_norm, y_train_norm, X_val_norm, y_val_norm, 'n-neighbors', n_min, n_max)

#### Plot Score, MAE and MSE for K-Neighbors Regressor
fig = plt.figure(figsize=(7,5))
plot_metric_2D ('Score', score_list, 'val', 'K-Neighbors Regressor', 'KNR', 'n-neighbors', n_min, n_max)
plot_metric_2D ('MAE', mae_list, 'val', 'K-Neighbors Regressor', 'KNR', 'n-neighbors', n_min, n_max)
plot_metric_2D ('MSE', mse_list, 'val', 'K-Neighbors Regressor', 'KNR', 'n-neighbors', n_min, n_max)


### Evaluate GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

#### Evaluate GradientBoostingRegressor using HistGradientBoostingRegressor which natively supports NaN values
print()
print(f"Histogram-based Gradient Boosting Regressor :")
HGB = HistGradientBoostingRegressor().fit(X_train_full, y_train_full)
predict_and_get_metrics(HGB, 'val', X_val, y_val)

#### Search interval for Evaluate GradientBoostingRegressor n_estimator
n_min = 290
n_max = 305
print()

#### Evaluate GradientBoostingRegressor by iteration over n_estimator and output results
XGB_better, N_better, score_list, mae_list, mse_list = look_for_best_score('Gradient Boosting Regressor', \
    GradientBoostingRegressor, X_train, y_train, X_val, y_val, 'n-estimators', n_min, n_max)

#### Plot Score, MAE and MSE for Gradient Boosting Regressor
fig = plt.figure(figsize=(7,5))
plot_metric_2D ('Score', score_list, 'val', 'Gradient Boosting Regressor', 'XGB', 'n-estimators', n_min, n_max)
plot_metric_2D ('MAE', mae_list, 'val', 'Gradient Boosting Regressor', 'XGB', 'n-estimators', n_min, n_max)
plot_metric_2D ('MSE', mse_list, 'val', 'Gradient Boosting Regressor', 'XGB', 'n-estimators', n_min, n_max)


### Evaluate RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

#### Search interval for n_estimator
n_min = 70
n_max = 100

#### Search interval for max_depth and xGridSearchCV only
d_min = 8
d_max = 12
print()
print(f"Random Forest Regressor pour n_estimator = {n_min}~{n_max} et max_depth = {d_min}~{d_max} :")

#### Other parameters for xGridSearchCV only
gridsearch_params = { 'n_estimators': [i for i in range(n_min, n_max+1)],
                      'max_depth': [i for i in range(d_min, d_max+1)] }
gridsearch_model = RandomForestRegressor(random_state = 14071789)

#### Look for best n_estimators and max_depth using HalvingGridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
print()
print("Optimisation de n_estimators et max_depth avec HalvingGridSearchCV :")

RFR_best_HG = HalvingGridSearchCV(gridsearch_model, gridsearch_params, factor=3, n_jobs=-1)
RFR_best_HG.fit(X_train, y_train)

print("best_params :", RFR_best_HG.best_params_)
predict_and_get_metrics(RFR_best_HG, 'val', X_val, y_val)

#### Look for best n_estimators and max_depth using GridSearchCV
from sklearn.model_selection import GridSearchCV
print()
print("Optimisation de n_estimators et max_depth avec GridSearchCV :")

RFR_best_GS = GridSearchCV(gridsearch_model, gridsearch_params, n_jobs=-1)
RFR_best_GS.fit(X_train, y_train)

print("best_params :", RFR_best_GS.best_params_)
predict_and_get_metrics(RFR_best_GS, 'val', X_val, y_val)

#### Parameters for plotting GridSearchCV results
x_GS = RFR_best_GS.cv_results_['param_max_depth']
y_GS = RFR_best_GS.cv_results_['param_n_estimators']
z_GS = RFR_best_GS.cv_results_['mean_test_score']
X_GS = np.arange(d_min, d_max+1)
Y_GS = np.arange(n_min, n_max+1)
X_surf, Y_surf = np.meshgrid(X_GS, Y_GS)
Z_surf = np.reshape(z_GS, (n_max-n_min+1, d_max-d_min+1))

#### Plot GridSearchCV results in 3D
from matplotlib.ticker import LinearLocator
fig_3D = plt.figure(figsize=(10,10))
ax = fig_3D.add_subplot(111,projection='3d')
surf = ax.plot_surface(X_surf, Y_surf, Z_surf, cmap='coolwarm', linewidth=0, antialiased=False)
ax.xaxis.set_major_locator(LinearLocator(d_max-d_min+1))
ax.xaxis.set_major_formatter('{x:.0f}')
ax.yaxis.set_major_locator(LinearLocator(10))
ax.yaxis.set_major_formatter('{x:.0f}')
ax.zaxis.set_major_formatter('{x:.03f}')
plt.title('Score Random Forest Regressor avec GridSearchCV', fontsize=16)
ax.set_xlabel('max_depth', fontsize=14)
ax.set_ylabel('n_estimators', fontsize=14)
fig_3D.savefig('RFR_score_'+str(n_min)+'-'+str(n_max)+'_d'+str(d_min)+'-'+str(d_max)+'_3D')
plt.clf()

#### Evaluate Random Forest Regressor by iteration over n_estimator and output results
print()
print("Optimisation par simple itération sur n_estimators (max_depth indéterminé) :")
RFR_better, R_better, score_list, mae_list, mse_list = look_for_best_score('Random Forest Regressor', \
    RandomForestRegressor, X_train, y_train, X_val, y_val, 'n-estimators', n_min, n_max)

#### Plot Score, MAE and MSE for Gradient Boosting Regressor
fig = plt.figure(figsize=(7,5))
plot_metric_2D ('Score', score_list, 'val', 'Random Forest Regressor', 'RFR', 'n-estimators', n_min, n_max)
plot_metric_2D ('MAE', mae_list, 'val', 'Random Forest Regressor', 'RFR', 'n-estimators', n_min, n_max)
plot_metric_2D ('MSE', mse_list, 'val', 'Random Forest Regressor', 'RFR', 'n-estimators', n_min, n_max)


# Final evaluation of best models

##  Final evaluation of XGB

### HistGradientBoostingRegressor model

#### Evaluate with test set after training on train+val
print()
print("*** Evaluation finale Histogram-based Gradient Boosting Regressor ***")
print()
HGB.fit(X_train_val, y_train_val)
print("Score, Absolute et Mean Square Error avec le jeu de test :")
predict_and_get_metrics(HGB, 'test', X_test, y_test)

#### Train and evaluate on full selected dataset
print()
print("Score, Absolute et Mean Square Error avec le jeu de données complet :")
HGB.fit(X_sel, y)
predict_and_get_metrics(HGB, 'full', X_sel, y)

### Best GradientBoostingRegressor model selected by iteration

### Evaluate with test set after training on train+val
print()
print("*** Evaluation finale Gradient Boosting Regressor ***")
print()
XGB_better.fit(X_train_val, y_train_val)
print("Score, Absolute et Mean Square Error avec le jeu de test :")
predict_and_get_metrics(XGB_better, 'test', X_test, y_test)

### Train and evaluate on full selected dataset
print()
print("Score, Absolute et Mean Square Error avec le jeu de données complet :")
XGB_full = GradientBoostingRegressor(n_estimators=N_better)
XGB_full.fit(X_sel, y)
predict_and_get_metrics(XGB_full, 'full', X_sel, y)

## Final evaluation of RFR

### Best RFR model selected with GridSearch

#### Evaluate with test set after training on train+val
print()
print("*** Evaluation finale Random Forest Regressor sélectionné avec GridSearch ***")
print()
print("Score, Absolute et Mean Square Error avec le jeu de test :")
RFR_best_GS.fit(X_train_val, y_train_val)
predict_and_get_metrics(RFR_best_GS, 'test', X_test, y_test)

#### Train and evaluate on full selected dataset
print()
print("Score, Absolute et Mean Square Error avec le jeu de données complet :")
RFR_best_GS.fit(X_sel, y)
predict_and_get_metrics(RFR_best_GS, 'full', X_sel, y)

### Best RFR model selected by iteration

#### Evaluate with test set after training on train+val
print()
print("*** Evaluation finale Random Forest Regressor sélectionné par simple itération sur n_estimators ***")
print()
print("Score, Absolute et Mean Square Error avec le jeu de test :")
RFR_better.fit(X_train_val, y_train_val)
predict_and_get_metrics(RFR_better, 'test', X_test, y_test)

#### Train and evaluate on full selected dataset
print()
print("Score, Absolute et Mean Square Error avec le jeu de données complet :")
RFR_better.fit(X_sel, y)
predict_and_get_metrics(RFR_better, 'full', X_sel, y)

### Backup model files trained on full selected dataset
print()
print("*** Sauvegarde des modèles et taille des fichiers correspondants ***")
print()
pickle.dump(HGB, open("Deep_Solar_model_HGB", 'wb'))
print("Taille du modèle HistGradientBoostingRegressor :", round(os.path.getsize("Deep_Solar_model_HGB")/2**10), "ko")
pickle.dump(XGB_full, open("Deep_Solar_model_XGB", 'wb'))
print("Taille du modèle GradientBoostingRegressor :", round(os.path.getsize("Deep_Solar_model_XGB")/2**10), "ko")
pickle.dump(RFR_best_GS, open("Deep_Solar_model_RFR_GS", 'wb'))
print("Taille du modèle RandomForestRegressor sélectionné avec GridSearch :", round(os.path.getsize("Deep_Solar_model_RFR_GS")/2**20), "Mo")
pickle.dump(RFR_better, open("Deep_Solar_model_RFR_iter", 'wb'))
print("Taille du modèle RandomForestRegressor sélectionné par itération :", round(os.path.getsize("Deep_Solar_model_RFR_iter")/2**20), "Mo")


### Unit tests on best regression model: HistGradientBoostingRegressor
print()
print("*** Tests unitaires avec le modèle retenu : HistGradientBoostingRegressor ***")
random_lines = np.random.randint(2, 63845, 3)
for test, line in zip(range(1,4), random_lines):
    raw = X_sel.iloc[line].to_list()
    input_df = pd.DataFrame(raw).transpose()
    input_df.columns = input_features
    pred = HGB.predict(input_df)[0]
    print()
    print("Test unitaire", test, ":")
    print("Input :", raw)
    print("Prédiction :", pred)
    print("Réel :", y.iloc[line])

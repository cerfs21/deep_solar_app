# Deep_Solar_for_Neo4j v1.0
# Deep_Solar_build_model v3.3 reworked to load Deep Solar data into a Neo4j database

'''
For reference:
https://www.kaggle.com/datasets/tunguz/deep-solar-dataset
http://web.stanford.edu/group/deepsolar/home
https://github.com/cerfs21/deep_solar_app
https://en.wikipedia.org/wiki/FIPS_county_code

'''

# IMPORT Python librairies
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso


# EXPLORE the dataset to identify and categorize features

# Import dataset in Pandas DataFrame
df = pd.read_csv('./data/deepsolar_tract.csv', encoding="ISO-8859-1")

# Dataset overview
print("\n*** Exploration du jeu de données d'origine ***\n")
print("Forme :", df.shape, "\nDescription :")
df.info()
print(df['fips'].nunique(), "FIPS codes")
print(df['county'].nunique(), "county names")
# County 5-digit FIPS codes are derived from full FIPS codes by stripping the 6 subarea digits
print(df['fips'].astype(str).str[:-6].nunique(), "counties / county FIPS codes")
print(df['state'].nunique(), "states")

# Identify possible target features to measure the deployment of solar panels
solar_features = ['tile_count', 'tile_count_residential', 'tile_count_nonresidential', 'solar_system_count',
                  'solar_system_count_residential', 'solar_system_count_nonresidential', 'total_panel_area',
                  'total_panel_area_residential', 'total_panel_area_nonresidential', 'heating_fuel_solar_rate',
                  'number_of_solar_system_per_household', 'solar_panel_area_divided_by_area', 'heating_fuel_solar',
                  'solar_panel_area_per_capita']

# Identify and categorize non target features
geographical_features = ['county', 'state', 'fips', 'lon', 'lat', 'elevation']

political_features = ['voting_2016_dem_percentage', 'voting_2016_gop_percentage', 'voting_2012_dem_percentage',
                      'voting_2012_gop_percentage', 'voting_2012_dem_win', 'voting_2016_dem_win']

education_numbers = ['education_bachelor', 'education_college', 'education_doctoral', 'education_high_school_graduate',
                     'education_less_than_high_school', 'education_master', 'education_professional_school',
                     'education_population']

education_features = ['education_less_than_high_school_rate', 'education_high_school_graduate_rate',
                      'education_college_rate', 'education_bachelor_rate', 'education_master_rate',
                      'education_professional_school_rate', 'education_doctoral_rate']

race_features = ['race_asian', 'race_black_africa', 'race_indian_alaska', 'race_islander', 'race_other',
                 'race_two_more', 'race_white', 'race_asian_rate', 'race_other_rate', 'race_two_more_rate']

age_features = ['age_median', 'age_18_24_rate', 'age_25_34_rate', 'age_more_than_85_rate', 'age_75_84_rate',
                'age_35_44_rate', 'age_45_54_rate', 'age_65_74_rate', 'age_55_64_rate', 'age_10_14_rate',
                'age_15_17_rate', 'age_5_9_rate']


# CLEAN the data, SELECT target feature and DISCARD irrelevant features

# Detect non-numerical or non-boolean features (3 object dtype columns listed by df.info) and output their content
print("\n*** Colonnes autres que numériques ou booléennes et aperçu de leur contenu *** \n")
for col in df.columns:
    if df[col].dtype == 'O':
        print(col, df[col].unique(), '\n')

# county and state features have string values as expected. electricity_price_transportation passed to numerical/NaN
df['electricity_price_transportation'] = pd.to_numeric(df['electricity_price_transportation'], errors='coerce')
print("Les colonnes county and state sont conformes.\n\
La colonne electricity_price_transportation est retraitée pour ne conserver que des valeurs numériques.")

# Ignore dataframe index and Gini index columns
index_columns = ['Unnamed: 0', 'gini_index']
df.drop(index_columns, axis=1, inplace=True)

# Create a dataframe copy to be used by Neo4j
df_geo = df.copy()
# Complement it with a county_fips column that will distinguish between counties with same names in different states
df_geo['county_fips']=df_geo['fips'].astype(str).str[:-6]

# Select target feature for feature selection
target = 'solar_panel_area_per_capita'
solar_features.remove(target)
# Discard other solar features which would introduce regression biases
df.drop(solar_features, axis=1, inplace=True)

# Discard data as irrelevant, arguable or redundant with other environment, social and economical features
features_removed = education_numbers + race_features + age_features + geographical_features
df.drop(features_removed, axis=1, inplace=True)

# Remove rows where data is incomplete in both dataframes
df.replace([np.inf, -np.inf], np.nan, inplace=True)  # inf values replaced with NaN
df_geo.replace([np.inf, -np.inf], np.nan, inplace=True)  # inf values replaced with NaN
df.dropna(inplace=True)  # Remove rows where at least one feature value is NaN
df_geo.dropna(inplace=True)  # Remove rows where at least one feature value is NaN

# Present and store the clean dataframe that will be later imported by Neo4j
print("\n*** Jeu de données nettoyé des valeurs indisponibles ***\n")
print("Forme :", df_geo.shape, "\nDescription :")
df_geo.info()
print(df_geo['fips'].nunique(), "FIPS codes")
print(df_geo['county'].nunique(), "county names")
print(df_geo['county_fips'].nunique(), "counties / county FIPS codes")
print(df_geo['state'].nunique(), "states")
df_geo.to_csv('./data/deepsolar_clean.csv', index=False)


# SELECT FEATURES using Linear Regression

# Separate input X and target y
X = df.drop(columns=target)
y = df[target]

# Normalize X with StandardScaler
scaler_X = StandardScaler()
scaler_X.fit(X)
X_norm = pd.DataFrame(scaler_X.transform(X), columns=X.columns)
X_norm.head()

# Normalize y with StandardScaler
scaler_y = StandardScaler()
scaler_y.fit(y.to_frame())
y_norm = scaler_y.transform(y.to_frame())

# Select features with Lasso (alpha parameter chosen to select 10~15 features)
print("\n*** Selection de features avec Lasso ***\n")
select_algo = Lasso(alpha=0.05)
selector = select_algo.fit(X_norm, y_norm)

# Store selected features and associated coefficients into a dictionary
coef_sum = 0
coef_count = 0
features_dict = defaultdict()
for i, feat in enumerate(X_norm.columns):
    if selector.coef_[i] != 0:
        # print(feat, round(selector.coef_[i], 3))
        features_dict[feat] = selector.coef_[i]
        coef_count += 1
        coef_sum += abs(selector.coef_[i])

# List selected features ordered by decreasing absolute values of Lasso coefficients
print("Features sélectionnées avec Lasso par ordre de coefficient décroissant :")
feat_coef_desc = {k: v for k, v in sorted(features_dict.items(), key=lambda x: abs(x[1]), reverse=True)}
for feat, coef in feat_coef_desc.items():
    print(feat, round(coef, 3))
print(f"\nCumul des valeurs absolues des {coef_count} coefficients : {coef_sum}")

# Deep_Solar_prediction_tester v1.1
#	perform 2 unit tests

import pickle
import pandas as pd

model = pickle.load(open("./data/Deep_Solar_model", 'rb'))

input_features = ['median_household_income','electricity_price_commercial','electricity_price_industrial','housing_unit_median_gross_rent','frost_days','relative_humidity','daily_solar_radiation','incentive_count_residential',
'incentive_nonresidential_state_level']

input = [71599.0, 9.44, 7.02, 681.0, 154.0, 0.695, 3.76, 34.0, 13.0]
input_df = pd.DataFrame(input).transpose()
input_df.columns = input_features
pred = model.predict(input_df)[0]
print("Unit test #1:")
print(f"Input: {input}")
print(f"solar_panel_area_per_capita in this area would likely be {pred} m2")

input = [95814.0, 15.31, 6.31, 1486.0, 117.0, 0.663, 3.69, 28.0, 16.0]
input_df = pd.DataFrame(input).transpose()
input_df.columns = input_features
pred = model.predict(input_df)[0]
print("Unit test #2:")
print(f"Input: {input}")
print(f"solar_panel_area_per_capita in this area would likely be {pred} m2")

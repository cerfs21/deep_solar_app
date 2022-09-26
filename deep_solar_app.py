# deep_solar_app v3.0:
#	English comments
#   connection_log.py renamed to connect.py
#	visualization.py renamed to visualize.py
#	logo image converted to webp and renamed with _

#################
# Import & Load #
#################

# Import Python librairies
import numpy as np
import pandas as pd

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dash import no_update
from dash_extensions.enrich import Output, DashProxy, Input, State, MultiplexerTransform
from dash.exceptions import PreventUpdate

from predict import load_model_and_predict
from connection_log import get_greeting_text
from visualize import make_figure_from_prediction


# Load dataset
areas = pd.read_csv('./data/deepsolar_tract.csv', encoding = "ISO-8859-1")


#############################
# Declare dataset variables #
#############################

print("==============================")

# Extract list of 49 states
states = areas["state"].unique()
print(len(states),"états")

# Extract list of 1843 counties
counties = areas["county"].unique()
print(len(counties),"comtés")

# Extract list of FIPS codes
fips = areas["fips"].unique()
print(len(fips),"FIPS")

# Create lists that will be used to manage Dash inputs
# 1) List of model inputs
input_ids =     ['state', 'county', 'fips', 'median_household_income',
                'electricity_price_commercial', 'electricity_price_industrial',
                'housing_unit_median_gross_rent', 'frost_days','relative_humidity',
                'daily_solar_radiation', 'incentive_count_residential', 'incentive_nonresidential_state_level']
# 2) List of model inputs translated in the language of the application interface
input_labels =  ["Etat", "Comté", "FIPS", "Revenu médian d'un ménage (dollars)",
                "Tarif électricité tertiaire (cents/kWh)", "Tarif électricité industrie (cents/kWh)",
                "Loyer médian (dollars)", "Jours de gel", "Humidité relative (%)",
                "Rayonnement solaire (kWh/m²/jour)", "Incitations publiques et privées", "Incitations de l'état"]
# 3) List of short descriptions of model inputs
input_notes =   ["choisissez un état des USA", "choisissez un comté dans cet état", "choisissez un code FIPS dans ce comté", "exprimé en dollars",
                "exprimé en cents/kWh", "exprimé en cents/kWh",
                "exprimé en dollars", "nombre", "exprimée en %",
                "exprimé en kWh/m²/jour", "nombre", "nombre"]


# The code below manages the user interface using Dash tabs and callbacks


########################################
# Declare Dash variables and functions #
########################################

app = DashProxy(
    __name__,
    prevent_initial_callbacks=True,
    transforms=[MultiplexerTransform()],
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

# Application name and logo displayed on top left of all tabs
logo_and_title = dbc.Row(
    [
        dbc.Col(
            html.Img(
                src=dash.get_asset_url("sun_and_solar_panel_logo.webp"),
                className="img-fluid",
            ),
            width=4,
        ),
        dbc.Col(
            html.H1("Deep Solar App."),
            width=8,
            className="text-left",
        ),
    ],
    className="g-0 d-flex align-items-center mb-4",
)

# Home button managed by HIDE_HOME_BUTTON
home_button = dbc.Row(
    [
        dbc.Col(dbc.Button("Revenir à l'accueil", className="btn-secondary", id="home-button"))
    ],
    className="mb-4"
)

# Buttons on first 4 tabs, allowing to get to the next tab up to the 5th and last tab 
buttons = [
    # Button to get from Home tab to Connection tab
    dbc.Button("Commencer", id="start-button"),

    # Button to get from Connection tab to Welcome tab
    dbc.Button("Se connecter", id="connect-button"),

    # Button to get from Welcome tab to Parameters tab
    dbc.Button("Lancer l'application", id="launch-button"),

    # Button to get from Parameters tab to Result tab
    dbc.Button("Obtenir une prédiction", id="predict-button"),

    # Button to get back to Parameters tab from Result tab
    dbc.Button("Modifier les paramètres", id="param-button"),    
]

# Input form for model parameters, other than those selecting an area and its default parameter values
input_form = []
for i in range(3, len(input_ids)):
    input_form.append(
        html.Div(
            [
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(input_labels[i]),
                        dbc.Input(id=input_ids[i], placeholder=input_notes[i], type="number"),
                    ]
                )    
            ],
            className="mb-2",
        )
    )

# Function to increment active_tab_number and get to the next tab
def get_next_tab(active_tab):
    # Extract the 5th character of active_tab_number, which is the current tab number
    active_tab_number = int(active_tab[4:])
    # Increment active_tab_number except when getting back from the 5th (Result) tab to the 4th (Parameters) tab
    if active_tab_number == 4:
        return f"tab-{3}"
    else:
        return f"tab-{active_tab_number + 1}"

# Dash parameters to control displaying tabs and home button
HIDE_TABS = True
HIDE_HOME_BUTTON = False


#####################
# Declare Dash tabs #
#####################

tabs = dbc.Tabs([
    # 1st (Home) tab linked to button_to_next_tab callback
    dbc.Tab(
        [
            dbc.Row(
                [
                    dbc.Col(html.H3("Assistant géomarketing pour le déploiement de panneaux solaires photovoltaïques."))
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card([
                        dbc.CardHeader(html.H4(f"Grâce à cette application vous pourrez..."), className="text-left pt-3"),
                        dbc.CardBody([
                            html.H6(f"Obtenir des informations sur la base installée dans la région de votre choix"),
                            html.H6(f"Prédire son potentiel d'évolution"),
                            html.H6(f"Evaluer l'influence de certains paramètres sur ce potentiel"),
                        ]),
                    ])),    
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    # Bouton 'Commencer'
                    dbc.Col(
                        buttons[0],
                        className="mb-2",
                    ),
                ]
            ),
        ],
        label="Accueil"
    ),

    # 2nd (Connection) tab linked to button_to_next_tab and enter_id callbacks
    dbc.Tab(
        [
            dbc.Row(
                [
                    dbc.Col(dbc.Input(id="name-box", placeholder="Entrez votre nom"), width=8),
                    # Bouton 'Se connecter"'
                    dbc.Col(
                        buttons[1],
                        width=4,
                        className="mb-2",
                    )
                ]
            ),
        ],
        label="Connexion"
    ),

    # 3rd (Welcome) tab linked to button_to_next_tab and enter_id callbacks
    dbc.Tab(
        [
            dbc.Row(
                [
                    dbc.Col(dbc.Card([
                        dbc.CardBody(html.Div(id="welcome-box1")),
                        dbc.CardBody(html.Div(id="welcome-box2"))
                    ])),
                ],
                className="mb-4",
            ),
            dbc.Row([
                # Bouton 'Lancer l'application'
                dbc.Col(
                    buttons[2],
                    className="text-center mb-2",
                )
            ])
        ],
        label="Bienvenue"
    ),

    # 4th (Parameters) tab linked to several callbacks
    # Select an area and display/modify default parameter values
    dbc.Tab(
        [
            dbc.Form(
                [
                    html.H5("Sélectionnez une zone (état + comté + FIPS), puis modifiez les paramètres courants de cette zone si vous le souhaitez.", className="mb-4"),
                    html.Div(
                        [
                            # Saisie de l'état
                            dbc.Label(input_labels[0], html_for="dropdown"),
                            dcc.Dropdown(id=input_ids[0], options=states, placeholder=input_notes[0]),
                            dbc.Alert("Veuillez entrer une valeur.", color="danger", fade=True, is_open=False, id="alert-0"),
                        ],
                        className="mb-2",
                    ),
                    html.Div(
                        [
                            # Saisie du comté
                            dbc.Label(input_labels[1], html_for="dropdown"),
                            dcc.Dropdown(id=input_ids[1], placeholder=input_notes[1]),
                            dbc.Alert("Veuillez entrer une valeur.", color="danger", fade=True, is_open=False, id="alert-1"),
                        ],
                        className="mb-2",
                    ),
                    html.Div(
                        [
                            # Saisie du code FIPS
                            dbc.Label(input_labels[2], html_for="dropdown"),
                            dcc.Dropdown(id=input_ids[2], placeholder=input_notes[2]),
                            dbc.Alert("Veuillez entrer une valeur.", color="danger", fade=True, is_open=False, id="alert-2"),
                        ],
                        className="mb-2",
                    ),
                ]
                # Input form displaying model parameters and their default values for the selected area
                + input_form
                # Button to get model prediction
                + [buttons[3]],
                className="mb-2",
            )
        ],
        label="Paramètres"
    ),
    
    # 5th (Result) tab
    # Display model prediction for the selected area based on selected parameter values
    dbc.Tab(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.H5(f"Valeurs courantes et prédictions"), className="text-center pt-3"),
                            dbc.CardBody(id="prediction-card")
                        ]),
                        className="mb-2",
                    ),
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.H5(f"Comparaison graphique"), className="text-center pt-3"),
                            dbc.CardBody(dcc.Graph(id="prediction-graph")),
                        ]),
                        className="mb-2"
                    ),
                ],
            ),
            dbc.Row(
                [
                    # Button to modify model parameters
                    dbc.Col(
                        buttons[4],
                        className="mb-2",
                    ),
                ]
            ),            
        ],
        label="Résultat"
    ),
], id="tabs", className="mb-4")


################################
# Application interface layout #
################################

if HIDE_TABS:
    tabs.className += " d-none"

if HIDE_HOME_BUTTON:
    home_button.className += " d-none"

app.layout = dbc.Container([
    dbc.Row(
        [
            # Tabs on left half of the screen
            dbc.Col(
                [
                    logo_and_title,
                    home_button,
                    dbc.Row(
                        [
                            dbc.Col(tabs)
                        ]
                    ),
                ],
                sm=12, lg=6,
            ),
            
            # Picture on right half of the screen
            dbc.Col(
                html.Img(
                    src=dash.get_asset_url("solar_panels_on_ground.webp"),
                    style=dict(width="100%"),
                    className="rounded-3 d-sm-none d-lg-block",
                ),
                sm=0, lg=6,
            )
        ],
        className="mt-4"
    ),
])


##########################
# Declare Dash CallBacks #
##########################

# Manage Home button
@app.callback(
    Output(component_id="tabs", component_property="active_tab"),

    Input(component_id="home-button", component_property="n_clicks"),
)
def home_button(n_clicks):
    return "tab-0"


# Manage Start and (application) Launch buttons on Home and Welcome tabs
@app.callback(
    Output(component_id="tabs", component_property="active_tab"),

    State(component_id="tabs", component_property="active_tab"),
    
    Input(component_id="start-button", component_property="n_clicks"),
    Input(component_id="launch-button", component_property="n_clicks"),
    Input(component_id="param-button", component_property="n_clicks"),
)
def button_to_next_tab(active_tab, *args):
    return get_next_tab(active_tab)


# Manage input for user identification
@app.callback(
    Output(component_id="welcome-box1", component_property="children"),
    Output(component_id="welcome-box2", component_property="children"),
    Output(component_id="tabs", component_property="active_tab"),
    Output(component_id="name-box", component_property="invalid"),

    State(component_id="name-box", component_property="value"),
    State(component_id="tabs", component_property="active_tab"),

    Input(component_id="connect-button", component_property="n_clicks"),
)
def enter_id_callback(name, active_tab, n_clicks):
    if name is None:
        return no_update, no_update, True

    text1, text2 = get_greeting_text(name)
    
    return text1, text2, get_next_tab(active_tab), False


# Build input list of counties relevant to selected state
@app.callback(
    Output(component_id=input_ids[1], component_property="options"),

    Input(component_id=input_ids[0], component_property="value"),
)
def county_options_callback(state_selected):
    options=areas[areas["state"]==state_selected]["county"].unique()
    return options


# Build input list of FIPS codes relevant to selected county
@app.callback(
    Output(component_id=input_ids[2], component_property="options"),

    Input(component_id=input_ids[1], component_property="value"),
)
def fips_options_callback(county_selected):
    options=areas[areas["county"]==county_selected]["fips"].unique()
    return options


# Display default model parameter values for selected FIPS code
@app.callback(
    Output(component_id=input_ids[3], component_property="value"),
    Output(component_id=input_ids[4], component_property="value"),
    Output(component_id=input_ids[5], component_property="value"),
    Output(component_id=input_ids[6], component_property="value"),
    Output(component_id=input_ids[7], component_property="value"),
    Output(component_id=input_ids[8], component_property="value"),
    Output(component_id=input_ids[9], component_property="value"),
    Output(component_id=input_ids[10], component_property="value"),
    Output(component_id=input_ids[11], component_property="value"),
    
    Input(component_id=input_ids[2], component_property="value"),
)
def get_default_callback(fips_selected):
    print("******************************")
    print("FIPS sélectionné :", fips_selected)
    default_val = []
    for i in range(3, len(input_ids)):
        default_val.append(areas[areas["fips"]==fips_selected][input_ids[i]].values[0])
    print("Default values for FIPS selected", default_val)
    return default_val

# Gather all parameters, launch model prediction and trigger Result tab
@app.callback(
    Output(component_id="prediction-card", component_property="children"),
    Output(component_id="prediction-graph", component_property="figure"),
    Output(component_id="tabs", component_property="active_tab"),
    Output(component_id="alert-0", component_property="is_open"),
    Output(component_id="alert-1", component_property="is_open"),
    Output(component_id="alert-2", component_property="is_open"),

    # State, county and FIPS parameters
    inputs=[State(component_id=input_ids[0], component_property="value"),
    State(component_id=input_ids[1], component_property="value"),
    State(component_id=input_ids[2], component_property="value"),
    
    # All 9 model parameters
    (State(component_id=input_ids[3], component_property="value"),
    State(component_id=input_ids[4], component_property="value"),
    State(component_id=input_ids[5], component_property="value"),
    State(component_id=input_ids[6], component_property="value"),
    State(component_id=input_ids[7], component_property="value"),
    State(component_id=input_ids[8], component_property="value"),
    State(component_id=input_ids[9], component_property="value"),
    State(component_id=input_ids[10], component_property="value"),
    State(component_id=input_ids[11], component_property="value")),
    
    State(component_id="tabs", component_property="active_tab"),

    Input(component_id="predict-button", component_property="n_clicks")],
)
def get_result_callback(state_selected, county_selected, fips_selected, input_values, active_tab, n_clicks):
    print("******************************")
    print("Etat sélectionné :", state_selected)
    print("Comté sélectionné :", county_selected)
    print("FIPS sélectionné :", fips_selected)
    
    # Trigger a warning for blank state, county or FIPS
    invalid_state = (state_selected == None)
    invalid_county = (county_selected == None)
    invalid_fips = (fips_selected == None)
    if invalid_state or invalid_county or invalid_fips:
        return no_update, no_update, no_update, invalid_state, invalid_county, invalid_fips
    
    print("Valeurs retenues :", input_values)
    current_val = areas[areas["fips"]==fips_selected]["solar_panel_area_per_capita"].values[0]
    population = areas[areas["fips"]==fips_selected]["population"].values[0]
    prediction = load_model_and_predict("./data/Deep_Solar_model", input_values, input_ids)
    installed = int(current_val*population)
    target = int(prediction*population)
        
    print("******************************")
    print("Inputs du CallBack :", dash.callback_context.states)
    print("******************************")
    print("Valeur courante de solar_panel_area_per_capita :", current_val)
    print("Valeur prédite de solar_panel_area_per_capita :", prediction)

    prediction_element = [
            dbc.Row([html.P(f"Population dans cette zone : {population} habitants"),
            html.P(f"Surface totale actuelle (base installée) : {installed} m²"),
            html.P(f"Surface totale modélisée (prédiction) : {target} m²")],
            className="text-center",
            )
    ]
    
    # Build conclusion depending on predicted value < or > installed base
    if target-installed >0:
        conclusion_element = [
            dbc.Row(html.H5(f"La surface à déployer pour atteindre la prédiction est {target-installed} m²."), className="text-center"),
        ]
    else:
        conclusion_element = [
            dbc.Row(html.H5(f"La surface déployée dépasse la valeur prévisible de {installed-target} m²."), className="text-center"),
        ]        

    prediction_text = prediction_element + conclusion_element
    prediction_figure = make_figure_from_prediction(installed, target)

    return prediction_text, prediction_figure, get_next_tab(active_tab), False, False, False


if __name__ == '__main__':
    app.run_server(debug=True)

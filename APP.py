import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
from xgboost import plot_importance
import plotly.express as px
from Modelling import train_df,X_train,y_train,X_test,feature_importances,model_training, xgbr,df_x,df_y


def Header(name, app):
    title = html.H2(name, style={"margin-top": 5})

    return dbc.Row([dbc.Col(title, md=9)])


fig= px.bar(feature_importances[0:20], x='importance', y='index',title='feature importance plot')



# Compute the explanation dataframe, GAM, and scores
mse_mean,mse_std=model_training(X_train,y_train)


# Start the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


# Card components
cards = [
    dbc.Card(
        [
            html.H2(f"{mse_mean:.2f}", className="card-title"),
            html.P("Model Training Average MSE Through CV", className="card-text"),
        ],
        body=True,
        color="light",
    ),
    dbc.Card(
        [
            html.H2(f"{mse_std:.2f}", className="card-title"),
            html.P("Model Training Standard Deviation MSE through CV", className="card-text"),
        ],
        body=True,
        color="dark",
        inverse=True,
    ),
    dbc.Card(
        [
            html.H2(f"{X_train.shape[0]} / {X_test.shape[0]}", className="card-title"),
            html.P("Train / Test Split", className="card-text"),
        ],
        body=True,
        color="primary",
        inverse=True,
    ),
]

prediction = [
    dbc.CardHeader(html.H5("Predict Salary")),
    dbc.CardBody(
        [
            dbc.FormGroup(
            [
                dbc.Label("Company Id"),
                dbc.Input(
                    type='text',
                    id="company",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Job Type"),
                dcc.Dropdown(
                    options=[{"label": col, "value": col} for col in list(set(train_df['jobType']))],
                    id="jobtype",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Degree"),
                dcc.Dropdown(
                    options=[{"label": col, "value": col} for col in list(set(train_df['degree']))],
                    id="degree",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("major"),
                dcc.Dropdown(
                    options=[{"label": col, "value": col} for col in list(set(train_df['major']))],
                    id="major",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("industry"),
                dcc.Dropdown(
                    options=[{"label": col, "value": col} for col in list(set(train_df['industry']))],
                    id="industry",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Required Years of Experience"),
                dbc.Input(
                    type='number',
                    id="yoe",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Miles from Metropolis"),
                dbc.Input(
                    type='number',
                    id="distance",
                ),
            ]
        ),
        dbc.Button("Submit", outline=True, color="primary", className="mr-1", n_clicks=0, id='submit-button'),
        html.Div(id="number-output"),
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

app.layout = dbc.Container(
    [
        Header("Salary Prediction Automation", app),
        html.Hr(),
        dbc.Row([dbc.Col(card) for card in cards]),
        html.Br(),
        dbc.Row(dcc.Graph(id='importance',figure=fig)),
        dbc.Row([dbc.Col(dbc.Card(prediction)),], style={"marginTop": 30}),
    ],
    fluid=False,
)

@app.callback (Output('number-output', 'children'),
              [Input('company', 'value'),
               Input('jobtype', 'value'),
               Input('degree', 'value'),
               Input('major', 'value'),
               Input('industry', 'value'),
               Input('yoe', 'value'),
               Input('distance', 'value'),
               Input('submit-button', 'n_clicks'),
               ])
def get_salary(company,jobtype,degree,major,industry,yoe,distance,n_clicks):
    if n_clicks:
        a = np.zeros(shape=(1, 94),dtype=int)
        test_df = pd.DataFrame(a, columns=df_x.columns)
        for i in test_df.columns:
            if jobtype in i:
                test_df[i] = 1
            if degree in i:
                test_df[i] = 1
            if major in i:
                test_df[i] = 1
            if industry in i:
                test_df[i] = 1
            if company in i:
                test_df[i] = 1
        test_df['yearsExperience']=yoe
        test_df['milesFromMetropolis']=distance
        xgbr.fit(df_x, df_y)
        pred=xgbr.predict(test_df)[0]
        return u'''
                    The predicted yearly salary is approximately ${}k.
                    '''.format(pred)
    else:
        return None


if __name__ == "__main__":
    app.run_server(debug=True)
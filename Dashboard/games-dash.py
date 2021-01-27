import dash
import dash_table
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import dash_extensions as de
import plotly.express as px
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd

# Preprocessed data from metacritic web-scrapping
df_games = pd.read_csv('data/games-data-cleaned.csv')
games_per_platform = df_games.groupby(['platform']).size().reset_index(name='counts')
by_month = df_games.groupby('month')
by_dow = df_games.groupby('dow')
df_peryear = pd.read_csv('data/games-data-per-year.csv')
df_cut = pd.read_csv('data/df-simple-recommender.csv')
df_cut_cr = pd.read_csv('data/df-simple-recommender-critics.csv')
df_cb = pd.read_csv('data/games-recommender.csv')
df_genre_plat = pd.read_csv('data/games-genre-platform.csv')

# Creating genre_list for each game
df_games['genre_list'] = df_games['genre'].apply(lambda s: list(set(s.split(','))) )
unique_genres = []
for i in df_games['genre_list']:
    unique_genres += i
unique_genres = list(set(unique_genres))

# Preparing platform list of dictionaries for dropdown options
platform_options = []
for plat in df_games['platform'].unique():
    my_dict = {}
    my_dict['label'] = str(plat)
    my_dict['value'] = str(plat)
    platform_options.append(my_dict)
platform_options = sorted(platform_options, key = lambda k: k['label'])

# Preparing years list of dictionaries for dropdown options
year_options = []
for year in sorted(df_games['year'].unique()):
    my_dict = {}
    my_dict['label'] = str(year)
    my_dict['value'] = str(year)
    year_options.append(my_dict)

### Function to make generic histogram
def make_dist(df, feature, plat):
    '''
    Function to plot a histogram of the distribution of 'feature' within 'df'.
    '''
    # Check distribution of feature:
    mean_feat = df[feature].mean()
    num_of_values = len(df)

    fig = px.histogram(df, x=feature,
                       title=plat+' --- Mean value of ' +feature+ ': ' +str(round(mean_feat,2))+
                           '<br> - With '+str(num_of_values)+' games',
                       color_discrete_sequence=['black'],
                       opacity = 0.6)

    return fig

##################################################### Preparation for content based recommender
indices = pd.Series(df_cb.index, index=df_cb['name'])
cosine_sim = np.loadtxt('data/cosine_sim.csv', delimiter=',')
print(cosine_sim.shape)

# Function that takes in game name as input and outputs most similar games
def get_recommendations(name, num_of_recs = 10, cosine_sim=cosine_sim):
    # Get the index of the game that matches the title
    idx = indices[name]

    # Get the pairwsie similarity scores of all games with that games
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the games based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar games
    sim_scores = sim_scores[1:num_of_recs+1]

    # Get the game indices
    game_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar games
    return df_cb[['name','platform','year']].iloc[game_indices]

game_options = []
for game in df_cb['name'].unique():
    my_dict = {}
    my_dict['label'] = str(game)
    my_dict['value'] = str(game)
    game_options.append(my_dict)
game_options = sorted(game_options, key = lambda k: k['label'])

#####################################################

# Lotties
lottie_url = "https://assets8.lottiefiles.com/packages/lf20_e4n4pfjf.json"
lottie_options = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))

##################### START OF APP
# mathjax = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML'
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY])

# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

## Creating the sidebar for multipage handling
sidebar = html.Div(
    [
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.Div(de.Lottie(options=lottie_options,
                                           width="70%",
                                           height="70%",
                                           url=lottie_url
                                           )),
                        html.H4("Metacritic Game Data explorer ",
                            style = {'textAlign':'center'}),
                        html.H6(["",
                            html.A('Data Source',
                                href = 'https://www.metacritic.com/browse/games/score/metascore/all/all/filtered?sort=desc',
                                target = '_blank)')
                            ], style={'textAlign':'right'}),

                        html.H6(["Made with ",
                            html.A('Dash', href = 'https://dash.plotly.com/',
                                target = '_blank)'),
                            html.Br(),
                            html.Br(),
                            "by ",
                            html.A('Bruno V. Ribeiro', href = 'https://github.com/BrunoBVR',
                                target = '_blank)'),
                            ], style={'textAlign':'right'}),
                    ]
                ),
            ],
            color="success",
            inverse=False,
            outline=False
        ),
        # html.H2("Sidebar", className="display-4"),
        html.Hr(),

        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Release date info", href="/rd-info", active = "exact"),
                dbc.NavLink("Genre info", href="/genre-info", active = "exact"),
                dbc.NavLink("Recommended games", href="/rec", active = "exact"),
                dbc.NavLink("Get recommendation", href="/get-rec", active = "exact"),
            ],
            vertical = True,
            pills = True,
        ),
        dbc.Card(
            [
                dbc.CardBody([
                    html.H6(["",
                        html.A('GitHub for scraper',
                            href = 'https://github.com/BrunoBVR/projectGames',
                            target = '_blank)')
                        ], style={'textAlign':'left'})
                ])
            ],
            color='warning'),
    ],
    style = SIDEBAR_STYLE,
)

## Creating content for each page
content = html.Div(id="page-content", children=[], style = CONTENT_STYLE)

## Main body of app layout
app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])

## Callback for each different page
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
################################################## MAIN PAGE
    if pathname == "/":
        return[
            html.H1("General info on the data",
                style = {'textAlign':'center'}),
            html.Hr(),
            html.Br(),

            html.H2(
                    children="Top 5 games rated by meta-score:",
                    id = 'top',
                style = {'textAlign':'left'}
            ),

            dcc.Slider(id='top-slider',
                min = 3,
                max = 20,
                step = 1,
                marks = {i: str(i) for i in range(1,21)},
                value = 5
            ),

            html.Div(id='main-table'),
            html.Hr(),
            html.H2('Search data for specific game:'),
            dcc.Input(id='game-input',
                type='text',
                value='Zelda',
                placeholder='Insert game to search for info.'
            ),
            dbc.Button('Submit game', id='submit-game', color='success', n_clicks=0),

            # Div for the alert with number of games
            html.Div(id="num-of-games"),

            # Div for the table of searched game
            html.Div(id='game-table'),

            html.Hr(),

            ## Insert number of games per platform
            dcc.Graph(id='games-plat',
                figure=(px.bar(games_per_platform,
                                color_discrete_sequence =['gray']*len(games_per_platform),
                                x='platform', y='counts',
                                labels={'platform':'Platform','counts':'Number of games'},
                                title='Number of games per platform').update_xaxes(categoryorder="total descending"))
            ),

            ## Distribution of scores by platform
            html.Hr(),

            dbc.Row([
                dbc.Col([
                    html.H3('Choose the platform to display distribution of meta-score:'),
                    dcc.Dropdown(id='plat-choice',
                        options = platform_options,
                        style={'color': '#000000'},
                        value = 'Nintendo64',
                        placeholder = 'Select platform.'
                        )
                ]), # end of column
                dbc.Col(
                    html.Br(),
                )
            ]),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='score-plat',
                        figure=(make_dist(df_games[df_games['platform'] == 'Nintendo64'], 'score', 'Nintendo64'))
                    ),
                ]), # end of column
                dbc.Col([
                    dcc.Graph(id='user-score-plat',
                        figure=(make_dist(df_games[df_games['platform'] == 'Nintendo64'], 'user score', 'Nintendo64'))
                    ),
                ]), # end of column
            ]),



        ]

################################################## R-date page
    elif pathname == "/rd-info":
        return[
            html.H1("Release Date information",
                style = {'textAlign':'center'}),
            html.Hr(),

            html.H2('Data on year of release'),
            dcc.Graph(
                figure = px.bar(df_peryear, x = 'year', y = '#_of_games',
                                height = 300,
                                color_discrete_sequence =['black'],
                                title = 'Number of games released by year')
            ),

            dcc.Graph(
                figure = px.bar(df_peryear, x = 'year', y = 'avg_score',
                                height = 300,
                                color_discrete_sequence =['black'],
                                title = 'Average meta score of games released by year')
            ),

            dcc.Graph(
                figure = px.bar(df_peryear, x = 'year', y = 'avg_user_score',
                                height = 300,
                                color_discrete_sequence =['black'],
                                title = 'Average user score of games released by year')
            ),

            html.Hr(),
            html.H2('Games released by month'),

            dcc.Graph(
                figure = px.bar(by_month.size().reset_index(name='counts'), x='month', y='counts',
                                height = 400,
                                color_discrete_sequence = ['gray'],
                                title = 'Number of games released by month').update_layout(
                                    xaxis= dict(
                                        tickmode = 'array',
                                        tickvals = [1,2,3,4,5,6,7,8,9,10,11,12],
                                        ticktext = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
                                    )
                                )
            ),
            html.Hr(),
            html.H2('Games released by day of the week'),

            dcc.Graph(
                figure = px.bar(by_dow.size().reset_index(name='counts'), x='dow', y='counts',
                                height = 400,
                                color_discrete_sequence = ['gray'],
                                title = 'Number of games released by day of the week',
                                category_orders = {
                                    'dow': ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']
                                },
                                labels = {
                                    'dow': 'Day of the Week'
                                })
            ),
        ]

################################################## Genre page
    elif pathname == "/genre-info":
        return[
            html.H1("Genre information",
                style = {'textAlign':'center'}),
            html.Hr(),

            dbc.Row([
                dbc.Col([
                    html.H3('Choose the platform to display distribution of genres:'),
                    dcc.Dropdown(id='plat-choice-genre',
                        options = platform_options,
                        style={'color': '#000000'},
                        value = 'Nintendo64',
                        placeholder = 'Select platform.'
                        )
                ]), # end of column
                dbc.Col(
                    html.Br(),
                )
            ]),

            dcc.Graph(id='genre-plat',
                figure = px.bar(df_genre_plat[df_genre_plat['Nintendo64']!=0],
                                x = 'genre', y ='Nintendo64',
                                title = 'Genres for the Nintendo64',
                                color_discrete_sequence = ['gray'],
                                labels={'Nintendo64':'Number of games'}).update_xaxes(categoryorder="total descending")
            ),

            html.Hr(),
            html.H3("Distribution of meta-score by genre"),

            dcc.Input(id='genre-input',
                type='text',
                value='Action',
                placeholder='Type a genre.'
            ),
            dbc.Button('Submit genre', id='submit-val',color='success', n_clicks=0),

            dcc.Graph(id='genre-score',
                figure = make_dist(df_games.loc[df_games['genre'].str.contains('action')], 'score', 'Action')
            ),

            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.H5(id='p-genre'),
                        ]
                    ),
                ],
                color="warning",
                inverse=False,
                outline=False
            )
        ]

################################################## Recommender page
    elif pathname == "/rec":
        return[
            html.H1("Game recommendation",
                style = {'textAlign':'center'}),
            html.Hr(),
            html.H2("Simple Recommender - Top games"),
            dbc.Jumbotron([
                dcc.Markdown(
                    """We will create a simple recommendation list to give the top games based on a metric
                    similar to the **IMDB TOP 250 Movies** score.
                    Simply using the score of the game is not a great ideia, given it does not consider the
                    popularity of the game (games with very few but passionate users will have a huge user score,
                    but not necessarily a good critic score).
                    Let's use the IMDB weighted rating formula as a metric."""),
                html.A('IMDB rating', href = 'https://help.imdb.com/article/imdb/track-movies-tv/ratings-faq/G67Y87TFYYP6TWAV#',
                       style={'color':'#000000'},
                       target = '_blank)'),
            ]),
            ### Recommended by user reviews
            html.H3(id = 'top-sr'),
            dcc.Slider(id='sr-slider',
                min = 5,
                max = 100,
                step = 5,
                marks = {i: str(i) for i in range(5,101,5)},
                value = 5
            ),
            html.Div(id='sr-table'),

            ### Recommended by critics reviews
            html.H3(id = 'top-cr'),
            html.Div(id='cr-table'),
        ]

################################################## Get recommendation page
    elif pathname == "/get-rec":
        return[
            html.H1("Game recommender",
                style = {'textAlign':'center'}),
            html.Hr(),

            dbc.Alert([
                html.H2("This is a content-based recommender system based on the top 1000 games sorted by meta-score.",
                        style= {'textAlign':'center'})
            ],color="secondary",
            ),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.H3('Choose game to display top recommendations:'),
                    dcc.Dropdown(id='game-choice',
                        options = game_options,
                        style={'color': '#000000'},
                        value = 'The Legend of Zelda: Ocarina of Time',
                        placeholder = 'Select game.'
                        )
                ]), # end of column
                dbc.Col(
                    html.Br(),
                )
            ]),
            html.Br(),

            html.Div(id='rec-table'),


        ]

    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )
##############################################################
############################### START OF INTERACTIVE CALLBACKS
##############################################################

### Main page
@app.callback(
    Output("top", "children"),
    Output("main-table", "children"),
    [Input("top-slider", "value")]
)
def text_top(value):
    text = f"Top {value} games rated by meta-score:"

    t = dbc.Table.from_dataframe(df_games[['name','platform','r-date','score','user score']].head(value),
                            striped = True,
                            bordered=True,
                            hover=True,
                            dark=True)
    return text, t

@app.callback(
    Output("game-table", "children"),
    [Input('submit-game', 'n_clicks'),
    State("game-input", "value")]
)
def game_table(n_clicks, value):
    val = value.lower()
    dff = df_games.loc[df_games['name'].str.contains(val, case = False)][['name','platform','r-date','score','user score', 'critics', 'users']]

    return dbc.Table.from_dataframe(df=dff,
                            id='game-table',
                            striped = True,
                            bordered=True,
                            hover=True,
                            dark=True)

@app.callback(
    Output("num-of-games", "children"),
    [Input("submit-game", 'n_clicks'),
    State("game-input", "value")]
)
def games_alert(n_clicks, value):
    val = value.lower()
    dff = df_games.loc[df_games['name'].str.contains(val, case = False)][['name','platform','r-date','score','user score']]
    return dbc.Alert(html.H4('There are ' +str( len(dff) )+ ' games matching your search!'),
                     dismissable = True,
                     fade = False)

@app.callback(
    Output("score-plat", "figure"),
    [Input("plat-choice", "value")]
)
def score_dist(value):
    return make_dist(df_games[df_games['platform'] == value], 'score', value)

@app.callback(
    Output("user-score-plat", "figure"),
    [Input("plat-choice", "value")]
)
def score_dist(value):
    return make_dist(df_games[df_games['platform'] == value], 'user score', value)

### Genre page
@app.callback(
    Output("genre-plat", "figure"),
    [Input("plat-choice-genre", "value")]
)
def genre_dist(value):
    return px.bar(df_genre_plat[df_genre_plat[value]!=0],
                    x = 'genre', y =value,
                    title = 'Genres for the '+ value,
                    color_discrete_sequence = ['gray'],
                    labels={value:'Number of games'}).update_xaxes(categoryorder="total descending")

@app.callback(
    Output("genre-score", "figure"),
    [Input('submit-val', 'n_clicks'),
    State("genre-input", "value")]
)
def genre_score(n_clicks, value):
    val = value.lower()
    return make_dist(df_games.loc[df_games['genre'].str.contains(val)], 'score', val)

@app.callback(
    Output("p-genre", "children"),
    [Input('submit-val', 'n_clicks'),
    State("genre-input", "value")]
)
def genre_text(n_clicks, value):
    val = value.lower()
    ret = "  ".join(s.upper()+"/ " for s in unique_genres if val in s.lower())
    return "Genres containing "+val.capitalize()+" : "+ret

### Recommender page
@app.callback(
    Output("top-sr", "children"),
    Output("sr-table", "children"),
    Output("top-cr", "children"),
    Output("cr-table", "children"),
    [Input("sr-slider", "value")]
)
def text_sr(value):
    text = f"Top {value} recommended games (weighted on user reviews):"
    text_cr = f"Top {value} recommended games (weighted on critic reviews):"

    t = dbc.Table.from_dataframe(df_cut.head(value),
                            striped = True,
                            bordered=True,
                            hover=True,
                            dark=True)

    t_cr = dbc.Table.from_dataframe(df_cut_cr.head(value),
                            striped = True,
                            bordered=True,
                            hover=True,
                            dark=True)
    return text, t, text_cr, t_cr

### Get recommendation page
@app.callback(
    Output("rec-table", "children"),
    [Input("game-choice", "value")]
)
def recommend(value):
    dff = get_recommendations(value,10, cosine_sim)
    t = dbc.Table.from_dataframe(df=dff,
                                striped = True,
                                bordered=True,
                                hover=True,
                                dark=True)
    return t

#####################################
if __name__ == '__main__':
    app.run_server()

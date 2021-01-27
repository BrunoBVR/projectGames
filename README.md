# Scraping games info from metacritic

The scraper is found on [scraper.ipynb](https://github.com/BrunoBVR/projectGames/blob/main/scraper.ipynb).
Requires `requests`, `bs4.BeautifulSoup`, `re` and `pandas`.

**The final dataset is on file [games-data.csv](https://github.com/BrunoBVR/projectGames/blob/main/games-data.csv)**.

We will use [Metacritic](https://www.metacritic.com/browse/games/score/metascore/all/all/filtered?sort=desc) data to create a dataframe with data on games across all platforms and all time.

## The dataset

We want to get a dataframe with all games with columns:
* **name**: The name of the game
* **platform**: Platform it was released
* **r-date**: date it was released
* **score**: average score given by critics (metascore)
* **user score**: average score given by users in the website
* **developer**: game developer
* **genre**: genre of the game (can be multiple)
* **players**: Number of players (some games don't have this information)
* **critics**: number of critics reviewing the game
* **users**: Number of metacritic users that reviewed the game

**All data was collected on November 10th, 2020.**

## Steps used in scraper:

* Create a dictionary `pages` that will contain the DataFrame objects from all pages. Each entry is a pandas DataFrame with data from the games in each site page. There are, currently, 180 pages of rated games.
* For each page, create a dictionary `data_page` of empty lists to be filled with the data from each game. As each page displays 100 games, each of this lists should contain 100 elements (except for the last page).
* Use `requests` to get into the url of each page and `BEautifulSoup` to parse the html file.
* Loop through all games in each page and scrap the relevant data. Note that 'developer', 'genre', 'players', 'critics' and 'users' are found on different URLs, so we need to fetch these for each game. This URL for each game is inside a `a` tag with a `title` class.
* There are a couple of if's in the scraper to ensure None objects get dealt with (some games don't have a number of players information, for example; some games have no user reviews, given it is not yet released, and a few others).
* After all data is collected (and it **takes a few hours** - a bit more than 15 in my laptop), all dataframes in the `pages` dictionary are concatenated to create a single one with all game data.
* The dataframe is export to a csv file.
* I enjoy the awesome new dataset and all I can do with it!

# Basic EDA and data cleaning

This can be found on the notebook `project-Games-Cleaning-and-EDA.ipynb`.

# Content-based Recommender system

This can be found on the notebook `project-Games-Recommender.ipynb`

# Dashboard for data exploration and recommender

Inside the `Dashboard` folder, there is a script named `games-dash.py` file for the creation of a multi-page dashboard using **Dash**. This dashboard uses most of the information in the EDA notebook and a simplified version of the content-based recommender (using only the top 1000 games ranked by meta score).

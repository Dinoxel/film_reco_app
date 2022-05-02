import pandas as pd
from sklearn.neighbors import NearestNeighbors
from unidecode import unidecode
import streamlit as st
from collections import defaultdict
import itertools

pd.set_option('display.width', 7000000)
pd.set_option('display.max_columns', 100)
st.set_page_config(
    page_title="Film Recommendation App",
    page_icon="🎞️",
    layout="wide"
)

l10n_fr = {
    'lang': 'Français (French)',
    'polished_search': 'Affiner votre recherche pour trouver moins de films.',
    'knn_selector': "Nombres de films à afficher :",
    'selector': 'Sélectionnez une langue :',
    'title': 'App de recommendation de films',
    'define_movie': "Écrivez le nom d'un film pour obtenir des recommendations",
    'warning_num_letters': "Le titre du film que vous recherchez doit comporter au moins 3 lettres",
    'warning_no_movie': "Aucun film ne correspond à votre recherche, veuillez en choisir un autre.",
    'saga_lotr_fullname': 'Seigneur des anneaux',
    'found_movies': "Les films suivants ressortent d'après votre recherche :",
    'relevant_movie': "Le film le plus pertinent semble être « {} » de {}.",
    'movie_selector': 'Sélectionner le film voulue dans la liste suivante. Défaut : {} ({})',
    "genre_title": "Titre",
    "genre_startYear": "Année",
    "genre_multigenres": "Genres",
    "genre_runtimeMinutes": "Durée (minutes)",
    "genre_averageRating": "Note (moyenne)",
    "genre_numVotes": "Nombre de votes",
    "genre_nconst": "Acteurs"
}

l10n_en = {
    'lang': 'English',
    'polished_search': 'Refine your search to find fewer movies.',
    'knn_selector': "Number of movies to display:",
    'selector': 'Select a language:',
    'title': 'Movies Recommendation App',
    'define_movie': 'Write the name of a movie to get recommendations (Input must be a French movie name)',
    'warning_num_letters': "The title of the movie you're looking for must contain at least 3 letters",
    'warning_no_movie': "No movies was found, please choose another one.",
    'saga_lotr_fullname': 'Lord of the Rings',
    'found_movies': "The following movies result from your search:",
    'relevant_movie': "The most relevant seems to be '{}' from {}.",
    'movie_selector': 'Select the desired movie in the following list. Default: {} ({})',
    "genre_title": "Title",
    "genre_startYear": "Year",
    "genre_multigenres": "Genres",
    "genre_runtimeMinutes": "Time (minutes)",
    "genre_averageRating": "Rating (average)",
    "genre_numVotes": "Num. of Votes",
    "genre_nconst": "Actors"
}

lang_selector = st.sidebar.selectbox('', (l10n_fr['lang'], l10n_en['lang']))

# Système de localization et l10n implémentés après fin du projet par Axel Simond
if lang_selector == l10n_fr['lang']:
    l10n = l10n_fr
else:
    st.warning('''The database only has French names for the moment; English names are planned to be added soon enough.''')
    l10n = l10n_en

def df_prettifier(df, final=False):
    df["multigenres"] = df["multigenres"].apply(lambda row: row.replace(",", ", "))
    df = df.rename(
        columns={
            'startYear': l10n['genre_startYear'],
            'title': l10n['genre_title'],
            'multigenres': l10n['genre_multigenres']
        }
    )

    if final:
        df = df.rename(
            columns={
                'runtimeMinutes': l10n['genre_runtimeMinutes'],
                'averageRating': l10n['genre_averageRating'],
                'numVotes': l10n['genre_numVotes'],
                'nconst': l10n['genre_nconst']
            }
        )

    return df

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++ THE APP STARTS HERE ++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
st.title(l10n['title'])

# "https://www.imdb.com/title/" + title_id + "/"

# Chargement de la base principale
@st.cache
def loading_dataframe():
    # Cache la base de base
    df_full_final_x = pd.read_csv(
        'https://media.githubusercontent.com/media/Dinoxel/film_reco_app/master/Desktop/projets/projet_2/database_imdb/df_full_final_X.csv',
        index_col=0
    )

    # Store la base d'affichage
    df_display_final_def = df_full_final_x.copy()[
        ['titleId', 'title', 'multigenres', 'startYear', 'runtimeMinutes', 'averageRating', 'numVotes', 'nconst']
    ]
    df_display_final_def['nconst'] = df_display_final_def['nconst'].astype(str)

    # Store la base de knn
    df_knn_final_def = df_full_final_x.copy().drop(
        columns=['averageRating', 'numVotes', 'startYear', 'runtimeMinutes', 'multigenres', 'years', 'nconst']
    )
    return df_display_final_def, df_knn_final_def

# Assignation de la DB principale aux bases d'affichage et de machine learning
df_display_final_x, df_knn_final_x = loading_dataframe()

# df_posters = pd.read_pickle(gen_link('df_posters'))

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++ WEIGHTS ++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Sélectionne toutes les colonnes booléennes
X = df_knn_final_x.iloc[:, 2:].columns

# Définit le poids de base pour toutes les colonnes sur 1
df_weights = pd.DataFrame([[1 for x in X]], columns=X)

# Définit les poids par type de colonnes
weight_popular_genres = 0.65
weight_genres_comedy = 1.1
weight_genres_western = 2
weight_rating_low = 0.75
weight_all_reals = 1.25
weight_all_actors = 0.6
weight_years = 0.85
weight_years_low = 0.75
weight_num_votes_low = 0.5
weight_num_votes_med = 0.65

# ======================== Genres ========================
# Créé un dictionnaire avec le nombre d'occurences de chaque genre à partir de la colonne "multigenres"
genres_count = defaultdict(int)
for k in itertools.chain.from_iterable(df_display_final_x["multigenres"].str.split(',')):
    genres_count[k] += 1

# Ne ressort que les genres étant présent plus de 1500 fois
popular_genres = [genre for genre, count in genres_count.items() if count > 1500]

# Applique le filtre sur tous les genres populaires
for genre in popular_genres:
    df_weights[genre] = weight_popular_genres

df_weights['Western'] = weight_genres_western
df_weights['Comedy'] = weight_genres_comedy

# ======================== Rating inférieur ou égal à 7.5 ========================
df_weights['rating <= 7.5'] = weight_rating_low

# ======================== Réalisateurs ========================
for real in df_weights.loc[:, 'nm0000229':'year <= 1960']:
    df_weights[real] = weight_all_reals

# ======================== Acteurs ========================
for actor in df_weights.loc[:, 'year >= 1990':'nm9654612']:
    df_weights[actor] = weight_all_actors

# ======================== Années ========================
for year in df_weights.loc[:, 'year <= 1960':'year >= 1990']:
    df_weights[year] = weight_years

df_weights['year <= 1960'] = weight_years_low
df_weights['numvotes <= 3.6k'] = weight_num_votes_low
df_weights['3.6k < numvotes > 16k'] = weight_num_votes_med
weights = df_weights.iloc[0].to_list()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++ INPUT +++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Définit la base à utiliser pour la recherche
df_display_titles = df_display_final_x[['titleId', 'title', 'numVotes', 'startYear', 'multigenres']]

# Demande un film à chercher
movie_title = unidecode(st.text_input(l10n['define_movie'])).lower()

# Condition si la demande fait moins de 3 lettres, repose la question
if not movie_title:
    st.write("")
elif len(movie_title) <= 2:
    st.warning(l10n['warning_num_letters'])
else:
    is_custom_word = False
    custom_words_dict = {'lotr': l10n['saga_lotr_fullname'],
                         'star wars': 'Star Wars',
                         'harry potter': 'Harry Potter',
                         'indiana jones': 'Indiana Jones'}

    # Condition pour les noms de saga, modifie 'movie_title' pour la recherche
    for acronym, saga_name in custom_words_dict.items():
        if movie_title == acronym:
            cleaned_name = saga_name  # Nécessaire pour afficher le nom de la saga plus loin
            movie_title = unidecode(saga_name).lower()
            is_custom_word = True
            break

    # Recherche le film demandé dans la base de données
    df_display_titles = df_display_titles[df_display_titles['title'].apply(lambda x: unidecode(x.lower())).str.contains(movie_title)]

    # Si au moins un film correspond à la recherche
    if not len(df_display_titles) > 0:
        st.warning(l10n['warning_no_movie'])
    else:
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++++++++++++++++++++++++++++ INPUT INDEX ++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        found_movie = False
        # condition si un seul film est présent après recherche
        if len(df_display_titles) == 1:
            movie_index = 0
            found_movie = True

            first_movie = df_display_titles.iloc[0]
            st.write(l10n['relevant_movie'].format(first_movie.title, first_movie.startYear))

        # condition si plusieurs films ont le même nom
        else:
            if len(df_display_titles) > 13:
                st.info(l10n["polished_search"])
            st.write(l10n['found_movies'])

            # condition pour la recherche par saga
            if is_custom_word:
                df_display_titles = df_display_titles.sort_values(by=['startYear', 'title']).reset_index()
                st.dataframe(df_prettifier(df_display_titles[['startYear', 'title', 'multigenres']]))

                first_movie = df_display_titles.iloc[0]

            # condition pour la recherche standard
            else:
                df_display_titles = df_display_titles.sort_values(by='numVotes', ascending=False).reset_index()
                st.dataframe(df_prettifier(df_display_titles[['startYear', 'title', 'multigenres']]))

                first_movie = df_display_titles.iloc[0]
                st.write(l10n['relevant_movie'].format(first_movie.title, first_movie.startYear))

            df_titles_selector = df_display_titles["title"] + " (" + df_display_titles["startYear"].astype(str) + ")"

            movie_selection = st.selectbox(
                l10n["movie_selector"].format(first_movie.title, first_movie.startYear),
                df_titles_selector.to_list()
            )

            movie_index = df_titles_selector[df_titles_selector == movie_selection].index[0]
            found_movie = True

        if found_movie:
            max_movie_length = st.sidebar.radio(l10n["knn_selector"], (5, 10, 15, 20))
            max_movie_length += 1
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # ++++++++++++++++++++++++++++++++++++++++++++++++++ MACHINE LEARNING ++++++++++++++++++++++++++++++++++++++++++++++++++
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            movie_id = df_display_titles.iloc[movie_index, :].titleId
            selected_movie = df_display_final_x[df_display_final_x['titleId'] == movie_id].iloc[:1]

            # Définit des infos pour le ML
            X = df_knn_final_x.iloc[:, 2:]
            weights = df_weights.iloc[0].to_list()
            n_neighbors_num = 6

            # MACHINE LEARNING
            while True:
                model_nn = NearestNeighbors(
                    n_neighbors=n_neighbors_num,
                    metric_params={"w": weights},
                    metric="wminkowski"
                ).fit(X)

                selected_movies_index = \
                    model_nn.kneighbors(df_knn_final_x[df_knn_final_x['titleId'] == movie_id].iloc[:1, 2:])[1][0][1:]

                # Augmente les voisins n si le film est présent dans la liste de recommendation
                # Bidouillage
                if len([df_knn_final_x.iloc[x, 0] for x in selected_movies_index if x != movie_id]) != max_movie_length:
                    n_neighbors_num += 1
                else:
                    break

            # Si le film est présent, le supprime des films à afficher
            selected_movies_index = selected_movies_index.tolist()
            if selected_movie.index.to_list()[0] in selected_movies_index:
                selected_movies_index.remove(selected_movie.index.to_list()[0])
            else:
                selected_movies_index = selected_movies_index[:-1]

            # Concatène les films prédits
            predicted_movies = pd.DataFrame()
            for movie_index in selected_movies_index:
                predicted_movies = pd.concat([predicted_movies, df_display_final_x.iloc[[movie_index]]], ignore_index=True)

            # Affiche le filmé sélectionné
            st.dataframe(df_prettifier(selected_movie[selected_movie.columns[1:]], final=True))

            # Affiche la recommendation de films
            st.dataframe(df_prettifier(predicted_movies[predicted_movies.columns[1:]], final=True))

            # get_html_title_page('0110912')
            # print(predicted_films)


def display_files():
    logo_image = "https://www.themoviedb.org/t/p/original/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg"

    st.markdown(
        """
        <style>
        .container {
            display: flex;
        }
        .logo-text {
            font-weight:700 !important;
            font-size:50px !important;
            color: #f9a01b !important;
            padding-top: 75px !important;
        }
        .logo-img {
            margin-right:1%;
            float:right;
            width: 19%;
        }

        .container .logo-img:last-child {
        margin-right:0;
        }
        .container {
        justify-content: space-between;
        display:flex
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    a = """<div class="container">
            <img class="logo-img" src="{LOGO_IMAGE}">
            <p class="logo-text">{name}</p>
        </div>"""

    st.markdown(f"""
        <div class="container">
        {[print(f'<img class="logo-img" src="{logo_image}">') for x in range(5)]}
        </div>
        """,
                unsafe_allow_html=True
                )

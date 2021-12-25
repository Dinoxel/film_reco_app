import pandas as pd
from sklearn.neighbors import NearestNeighbors
from unidecode import unidecode
import streamlit as st
from collections import defaultdict
import itertools

pd.set_option('display.width', 7000000)
pd.set_option('display.max_columns', 100)

l10n_fr = {'l10n_lang': 'Français (French)',
           'l10n_selector': 'Sélectionnez une langue :',
           'l10n_title': 'App de recommendation de films',
           'l10n_define_film': "Écrivez le nom d'un film pour obtenir des recommendations",
           'l10n_warning_num_letters': "Le titre du film que vous recherchez doit comporter au moins 3 lettres",
           'l10n_warning_no_film': "Aucun film ne correspond à votre recherche, veuillez en choisir un autre.",
           'l10n_saga_lotr_fullname': 'Seigneur des anneaux',
           'l10n_found_films': "Les films suivants ressortent d'après votre recherche :",
           'l10n_choose_index_saga': "Choissisez l'index du film souhaité pour la saga « {} » :",
           'l10n_choose_index_normal': "Pour le sélectionner écrivez 0, sinon écrivez l'index du film souhaité :",
           'l10n_relevant_film': "Le film le plus pertinent semble être « {} » de {}.",
           'l10n_warning_index_not_present': "Il semblerait que l'index {} ne soit pas dans la liste, veuillez sélectionner un index valide",
           'l10n_selected_film': "Vous avez sélectionné le film : « {} »"}

l10n_en = {'l10n_lang': 'English',
           'l10n_selector': 'Select a language:',
           'l10n_title': 'Films Recommendation App',
           'l10n_define_film': 'Write the name of a film to get recommendations (Input must be a French Film name)',
           'l10n_warning_num_letters': "The title of the film you're looking for must contain at least 3 letters",
           'l10n_warning_no_film': "No films was found, please choose another one.",
           'l10n_saga_lotr_fullname': 'Lord of the Rings',
           'l10n_found_films': "The following films result from your search:",
           'l10n_choose_index_saga': "Choose the wished film's index for the '{}' saga:",
           'l10n_choose_index_normal': "In order to select it, write 0, otherwise write the number of the desired film:",
           'l10n_relevant_film': "The most relevant seems to be '{}' from {}.",
           'l10n_warning_index_not_present': "It seems that the index {} isn't in the list, please select a valid index",
           'l10n_selected_film': "You have selected the film: '{}'"}


lang_selector = st.sidebar.selectbox('', (l10n_fr['l10n_lang'], l10n_en['l10n_lang']))

# Système de localization et l10n implémentés après fin du projet par Axel Simond
if lang_selector == l10n_fr['l10n_lang']:
    l10n = l10n_fr
else:
    st.warning('''The database only has French names for the moment; English names are planned to be added soon enough.''')
    l10n = l10n_en

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++ THE APP STARTS HERE ++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
st.title(l10n['l10n_title'])

# "https://www.imdb.com/title/" + title_id + "/"

# Chargement de la base principale
@st.cache
def loading_dataframe():
    # Cache la base de base
    df_full_final_X = pd.read_csv(
        'https://media.githubusercontent.com/media/Dinoxel/film_reco_app/master/Desktop/projets/projet_2/database_imdb/df_full_final_X.csv',
        index_col=0)

    # Store la base d'affichage
    df_display_final_def = df_full_final_X.copy()[
        ['titleId', 'title', 'multigenres', 'startYear', 'runtimeMinutes', 'averageRating', 'numVotes', 'nconst']]
    df_display_final_def['nconst'] = df_display_final_def['nconst'].astype(str)

    # Store la base de knn
    df_knn_final_def = df_full_final_X.copy().drop(
        columns=['averageRating', 'numVotes', 'startYear', 'runtimeMinutes', 'multigenres', 'years', 'nconst'])
    return df_display_final_def, df_knn_final_def

# Assignation de la DB principale aux bases d'affichage et de machine learning
df_display_final_X, df_knn_final_X = loading_dataframe()

# df_posters = pd.read_pickle(gen_link('df_posters'))

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++ WEIGHTS ++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Selectionne toutes les colonnes booléennes
X = df_knn_final_X.iloc[:, 2:].columns

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
weight_numvotes_low = 0.5
weight_numvotes_med = 0.65

# ======================== Genres ========================
# Créé un dictionnaire avec le nombre d'occurences de chaque genre à partir de la colonne "multigenres"
genres_count = defaultdict(int)
for k in itertools.chain.from_iterable(df_display_final_X["multigenres"].str.split(',')):
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
df_weights['numvotes <= 3.6k'] = weight_numvotes_low
df_weights['3.6k < numvotes > 16k'] = weight_numvotes_med
weights = df_weights.iloc[0].to_list()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++ INPUT +++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Définit la base à utiliser pour la recherche
df_display_titles = df_display_final_X[['titleId', 'title', 'numVotes', 'startYear', 'multigenres']]

# Demande un film à chercher
film_title = unidecode(st.text_input(l10n['l10n_define_film'])).lower()

# Condition si la demande fait moins de 3 lettres, repose la question
if not film_title:
    st.write("")
elif len(film_title) <= 2:
    st.warning(l10n['l10n_warning_num_letters'])
else:
    is_custom_word = False
    custom_words_dict = {'lotr': l10n['l10n_saga_lotr_fullname'],
                         'star wars': 'Star Wars',
                         'harry potter': 'Harry Potter',
                         'indiana jones': 'Indiana Jones'}

    # Condition pour les noms de saga, modifie 'film_title' pour la recherche
    for acronym, saga_name in custom_words_dict.items():
        if film_title == acronym:
            cleaned_name = saga_name  # Nécessaire pour afficher le nom de la saga plus loin
            film_title = unidecode(saga_name).lower()
            is_custom_word = True
            break

    # Recherche le film demandé dans la base de données
    df_display_titles = df_display_titles[
        df_display_titles['title'].apply(lambda x: unidecode(x.lower())).str.contains(film_title)]

    # Si au moins un film correspond à la recherche

    if not len(df_display_titles) > 0:
        st.warning(l10n['l10n_warning_no_film'])
    else:

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++++++++++++++++++++++++++++ INPUT INDEX ++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        film_index = 123456789
        # condition si un seul film est présent après recherche
        if len(df_display_titles) == 1:
            film_index = 0

        # condition si plusieurs films ont le même nom
        else:
            st.write(l10n['l10n_found_films'])

            # condition pour la recherche par saga
            if is_custom_word:
                df_display_titles = df_display_titles.sort_values(by=['startYear', 'title']).reset_index()
                st.dataframe(df_display_titles[['startYear', 'title']])
                text_index_input = l10n['l10n_choose_index_saga'].format(cleaned_name)

            # condition pour la recherche standard
            else:
                df_display_titles = df_display_titles.sort_values(by='numVotes', ascending=False).reset_index()
                st.dataframe(df_display_titles[['startYear', 'title', 'multigenres']])
                first_film = df_display_titles.iloc[0]
                text_index_input = l10n['l10n_choose_index_normal']
                st.write(l10n['l10n_relevant_film'].format(first_film.title, first_film.startYear))

            selected_film = st.text_input(text_index_input)

            if selected_film:
                # condition si l'index n'est pas dans la liste
                if int(selected_film) not in list(range(len(df_display_titles))):
                    st.warning(l10n['l10n_warning_index_not_present'].format(str(selected_film)))
                else:
                    film_index = df_display_titles.index[int(selected_film)]
            else:
                st.write("")

        # condition pour éviter au code de fonctionner si aucun paramètre n'a été rentré
        # bidouillage
        if film_index == 123456789:
            st.write('')
        else:

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # ++++++++++++++++++++++++++++++++++++++++++++++++++ MACHINE LEARNING ++++++++++++++++++++++++++++++++++++++++++++++++++
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            st.write(l10n['l10n_selected_film'].format(df_display_titles.iloc[film_index, :].title))
            film_id = df_display_titles.iloc[film_index, :].titleId
            selected_film = df_display_final_X[df_display_final_X['titleId'] == film_id].iloc[:1]

            # Définit des infos pour le ML
            X = df_knn_final_X.iloc[:, 2:]
            weights = df_weights.iloc[0].to_list()
            n_neighbors_num = max_film_length = 6

            # MACHINE LEARNING
            while True:
                model_nn = NearestNeighbors(n_neighbors=n_neighbors_num, metric_params={"w": weights},
                                            metric='wminkowski').fit(X)
                selected_films_index = \
                model_nn.kneighbors(df_knn_final_X[df_knn_final_X['titleId'] == film_id].iloc[:1, 2:])[1][0][1:]

                # Augmente les voisins n si le film est présent dans la liste de recommendation
                # Bidouillage
                if len([df_knn_final_X.iloc[x, 0] for x in selected_films_index if x != film_id]) != max_film_length:
                    n_neighbors_num += 1
                else:
                    break

            # Si le film est présent, le supprime des films à afficher
            selected_films_index = selected_films_index.tolist()
            if selected_film.index.to_list()[0] in selected_films_index:
                selected_films_index.remove(selected_film.index.to_list()[0])
            else:
                selected_films_index = selected_films_index[:-1]

            # Concatène les films prédits
            predicted_films = pd.DataFrame()
            for film_index in selected_films_index:
                predicted_films = pd.concat([predicted_films, df_display_final_X.iloc[[film_index]]], ignore_index=True)

            # Affiche le filmé sélectionné
            st.dataframe(selected_film)

            # Affiche la recommendation de films
            st.dataframe(predicted_films)

            # get_html_title_page('0110912')
            # print(predicted_films)


def display_files():
    LOGO_IMAGE = "https://www.themoviedb.org/t/p/original/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg"

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
        {[print(f'<img class="logo-img" src="{LOGO_IMAGE}">') for x in range(5)]}
        </div>
        """,
                unsafe_allow_html=True
                )

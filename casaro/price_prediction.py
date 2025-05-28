import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


sklearn.set_config(transform_output='pandas')


#dataset e dati
df = pd.read_csv('data/steam.csv')
df.dropna(axis=0, inplace=True)
df = df[df["owners"] != "0-20000"]
df['release_date'] = (df['release_date'].str.replace('-', '').astype(int) // 10000)
df['positive_percentage'] = df['positive_ratings'] / (df['positive_ratings'] + df['negative_ratings'])


# 1. Separare i generi multipli in una lista
df['genres_split'] = df['genres'].str.split(';')

# 2. Creare il One-Hot Encoding per ciascun genere
# Esplodiamo la lista di generi per ottenere un singolo genere per riga
genres_dummies = df['genres_split'].explode().str.get_dummies()

# 3. Rinominare le colonne come "Genre:'nomeGenere'"
genres_dummies.columns = ['Genre: ' + col for col in genres_dummies.columns]

# 4. Ripristinare gli indici per entrambi i DataFrame
df = df.reset_index(drop=True)
genres_dummies = genres_dummies.reset_index(drop=True)

# 5. Unire i dummies con il dataframe originale (aggiungendo nuove colonne per i generi)
df = pd.concat([df, genres_dummies], axis=1)

# 6. Rimuovere le colonne con meno di 400 occorrenze
# Calcolare la somma di ogni colonna (quante volte appare il genere)
genre_counts = genres_dummies.sum()

# Filtrare le colonne che appaiono più di 400 volte
columns_to_keep = genre_counts[genre_counts >= 250].index

# Filtrare il dataframe mantenendo solo le colonne di generi desiderate
df = df[columns_to_keep.tolist() + [col for col in df.columns if col not in genres_dummies.columns]]

#drop rows with Nan
df.dropna(axis=0, inplace=True)

#Conta il numero di colonne
print(f"Numero di colonne dopo il filtro: {len(df.columns)}")
#print columns names
print(f"Colonne rimanenti: {df.columns.tolist()}")

df


x = df[['release_date', 'publisher', 'median_playtime', 'price', 'Genre: Action', 'Genre: Adventure', 'Genre: Casual', 'Genre: Early Access', 'Genre: Free to Play', 'Genre: Indie', 'Genre: Massively Multiplayer', 'Genre: RPG', 'Genre: Racing', 'Genre: Simulation', 'Genre: Sports', 'Genre: Strategy']]
y = df['positive_percentage']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


encoder = ColumnTransformer(
    [
        ('onehot', OneHotEncoder(sparse_output=False, min_frequency=5, handle_unknown='infrequent_if_exist'), ['publisher'])
    ],
    remainder='passthrough',
    verbose_feature_names_out=False,
    force_int_remainder_cols=False
)


pipe = Pipeline([
    ('encoder', encoder),
    ('standardization', StandardScaler()),
    ('regressor', RandomForestRegressor())
])


params = {
    'regressor__n_estimators' : [1, 10, 30],
    'regressor__criterion': ['squared_error'],
    'encoder__onehot__min_frequency': [1, 4, 7]
}


from sklearn.model_selection import GridSearchCV

grid_seach = GridSearchCV(
    estimator=pipe,
    param_grid=params,
    scoring=make_scorer(mean_absolute_error, greater_is_better=False),
    #n_jobs=-1,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    refit=True,
    verbose=4
)


grid_seach.fit(x_train, y_train)


mae = mean_absolute_error(y_test, grid_seach.predict(x_test))
mape = mean_absolute_percentage_error(y_test, grid_seach.predict(x_test))

print(mae)
print(f"errore medio assoluto: {(mae*100).round(1)}%")

print(mape)
print(f"errore medio percentuale assoluto: {(mape*100).round(1)}%")


df_prediction = pd.read_csv('data/predict.csv')
x_prediction = df_prediction[['release_date', 'publisher', 'median_playtime', 'price', 'Genre: Action', 'Genre: Adventure', 'Genre: Casual', 'Genre: Early Access', 'Genre: Free to Play', 'Genre: Indie', 'Genre: Massively Multiplayer', 'Genre: RPG', 'Genre: Racing', 'Genre: Simulation', 'Genre: Sports', 'Genre: Strategy']]
y_prediction = grid_seach.predict(x_prediction)

df_prediction['pos_perc'] = (df_prediction['pos_perc'].astype(float) * 100).round(2)
df_prediction['predicted_pos_perc'] = (y_prediction*100).round(2)
print(df_prediction[['name', 'predicted_pos_perc', 'pos_perc']])
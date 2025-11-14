from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, split, explode, trim, count, desc,
    regexp_extract, avg
)

# -----------------------------------------------------------------------------
# 1) Criar sessão Spark
# -----------------------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("NetflixAnalysis")
    .getOrCreate()
)

# ↓↓↓ MENOS LIXO NO TERMINAL ↓↓↓
spark.sparkContext.setLogLevel("WARN")

# -----------------------------------------------------------------------------
# 2) Ler dataset
# -----------------------------------------------------------------------------
file_path = "/opt/spark/data/netflix_titles.csv"

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(file_path)
)

print("\n====================== ESQUEMA DO DATASET ======================")
df.printSchema()

print("\n================== CONTAGEM POR TIPO (Movie x TV Show) ==================")
df.groupBy("type").count().show(truncate=False)

# -----------------------------------------------------------------------------
# 3) Explodir gêneros (listed_in) e países (country)
# -----------------------------------------------------------------------------
genre_df = (
    df
    .withColumn("genre", explode(split(col("listed_in"), ",")))
    .withColumn("genre", trim(col("genre")))
)

country_df = (
    df
    .withColumn("country_exploded", explode(split(col("country"), ",")))
    .withColumn("country_exploded", trim(col("country_exploded")))
)

# -----------------------------------------------------------------------------
# 4) Estatísticas para FILMES
# -----------------------------------------------------------------------------
movies_df = df.filter(col("type") == "Movie")

print("\n================== GÊNEROS MAIS FREQUENTES EM FILMES ==================")
movie_genres = (
    genre_df
    .filter(col("type") == "Movie")
    .groupBy("genre")
    .agg(count("*").alias("count"))
    .orderBy(desc("count"))
)
movie_genres.show(10, truncate=False)

top_movie_genre_row = movie_genres.first()
top_movie_genre = top_movie_genre_row["genre"] if top_movie_genre_row else "Drama"

print("\n================== PAÍSES COM MAIS FILMES ==================")
movie_countries = (
    country_df
    .filter(col("type") == "Movie")
    .groupBy("country_exploded")
    .agg(count("*").alias("count"))
    .orderBy(desc("count"))
)
movie_countries.show(10, truncate=False)

top_movie_country_row = movie_countries.first()
top_movie_country = (
    top_movie_country_row["country_exploded"]
    if top_movie_country_row else "United States"
)

print("\n================== RATINGS MAIS COMUNS EM FILMES ==================")
movie_ratings = (
    movies_df
    .groupBy("rating")
    .agg(count("*").alias("count"))
    .orderBy(desc("count"))
)
movie_ratings.show(10, truncate=False)

top_movie_rating_row = movie_ratings.first()
top_movie_rating = top_movie_rating_row["rating"] if top_movie_rating_row else "PG-13"

# ---- DURAÇÃO MÉDIA DOS FILMES (tratando campos vazios) ---------------------
movies_with_minutes = (
    movies_df
    # extrai apenas o número de "90 min"
    .withColumn("duration_min_str", regexp_extract(col("duration"), r"(\\d+)", 1))
    # ignora linhas onde não achou número
    .filter(col("duration_min_str") != "")
    .withColumn("duration_min", col("duration_min_str").cast("int"))
)

avg_movie_duration_row = (
    movies_with_minutes
    .agg(avg("duration_min").alias("avg_min"))
    .first()
)

avg_movie_duration = (
    int(avg_movie_duration_row["avg_min"])
    if avg_movie_duration_row and avg_movie_duration_row["avg_min"] is not None
    else 100
)

print("\n================== DURAÇÃO MÉDIA DOS FILMES (min) ==================")
print(f"Duração média aproximada dos filmes: {avg_movie_duration} minutos")

# -----------------------------------------------------------------------------
# 5) Estatísticas para SÉRIES (TV Shows)
# -----------------------------------------------------------------------------
shows_df = df.filter(col("type") == "TV Show")

print("\n================== GÊNEROS MAIS FREQUENTES EM SÉRIES ==================")
tv_genres = (
    genre_df
    .filter(col("type") == "TV Show")
    .groupBy("genre")
    .agg(count("*").alias("count"))
    .orderBy(desc("count"))
)
tv_genres.show(10, truncate=False)

top_tv_genre_row = tv_genres.first()
top_tv_genre = top_tv_genre_row["genre"] if top_tv_genre_row else "TV Dramas"

print("\n================== PAÍSES COM MAIS SÉRIES ==================")
tv_countries = (
    country_df
    .filter(col("type") == "TV Show")
    .groupBy("country_exploded")
    .agg(count("*").alias("count"))
    .orderBy(desc("count"))
)
tv_countries.show(10, truncate=False)

top_tv_country_row = tv_countries.first()
top_tv_country = (
    top_tv_country_row["country_exploded"]
    if top_tv_country_row else "United States"
)

print("\n================== RATINGS MAIS COMUNS EM SÉRIES ==================")
tv_ratings = (
    shows_df
    .groupBy("rating")
    .agg(count("*").alias("count"))
    .orderBy(desc("count"))
)
tv_ratings.show(10, truncate=False)

top_tv_rating_row = tv_ratings.first()
top_tv_rating = top_tv_rating_row["rating"] if top_tv_rating_row else "TV-14"

# ---- NÚMERO MÉDIO DE TEMPORADAS (tratando campos vazios) -------------------
shows_with_seasons = (
    shows_df
    .withColumn("seasons_str", regexp_extract(col("duration"), r"(\\d+)", 1))
    .filter(col("seasons_str") != "")
    .withColumn("seasons", col("seasons_str").cast("int"))
)

avg_seasons_row = shows_with_seasons.agg(avg("seasons").alias("avg_seasons")).first()
avg_seasons = (
    int(avg_seasons_row["avg_seasons"])
    if avg_seasons_row and avg_seasons_row["avg_seasons"] is not None
    else 2
)

print("\n================== NÚMERO MÉDIO DE TEMPORADAS ==================")
print(f"Número médio aproximado de temporadas: {avg_seasons}")

# -----------------------------------------------------------------------------
# 6) Criar FILME hipotético com base nas análises
# -----------------------------------------------------------------------------
hyp_movie = {
    "title": "Sombras em " + top_movie_country,
    "type": "Movie",
    "main_genre": top_movie_genre,
    "country": top_movie_country,
    "average_duration_min": avg_movie_duration,
    "rating": top_movie_rating,
    "year": 2025,
}

hyp_movie_description = f"""
================== FILME HIPOTÉTICO (BASEADO NO DATASET) ==================

Título: {hyp_movie['title']}
Tipo: {hyp_movie['type']}
Gênero principal: {hyp_movie['main_genre']}
País de produção: {hyp_movie['country']}
Duração aproximada: {hyp_movie['average_duration_min']} minutos
Classificação indicativa: {hyp_movie['rating']}
Ano de lançamento: {hyp_movie['year']}

Justificativa:
- Gênero escolhido: '{hyp_movie['main_genre']}' é o gênero de filme mais frequente no dataset.
- País: '{hyp_movie['country']}' aparece entre os que mais produzem filmes na base.
- Duração: usamos a média de duração dos filmes calculada via Spark (coluna 'duration').
- Rating: '{hyp_movie['rating']}' está entre as classificações mais comuns nos filmes.

Conteúdo (ideia):
Um filme voltado ao público que mais consome esse tipo de produção na plataforma, 
aproveitando o gênero e o país que já têm alta aceitação, mas com uma narrativa original.
"""

# -----------------------------------------------------------------------------
# 7) Criar SÉRIE hipotética com base nas análises
# -----------------------------------------------------------------------------
hyp_series = {
    "title": "Conexões em " + top_tv_country,
    "type": "TV Show",
    "main_genre": top_tv_genre,
    "country": top_tv_country,
    "seasons": avg_seasons,
    "rating": top_tv_rating,
    "year": 2025,
}

hyp_series_description = f"""
================== SÉRIE HIPOTÉTICA (BASEADA NO DATASET) ==================

Título: {hyp_series['title']}
Tipo: {hyp_series['type']}
Gênero principal: {hyp_series['main_genre']}
País de produção: {hyp_series['country']}
Número previsto de temporadas: {hyp_series['seasons']}
Classificação indicativa: {hyp_series['rating']}
Ano de lançamento: {hyp_series['year']}

Justificativa:
- Gênero: '{hyp_series['main_genre']}' está entre os gêneros mais recorrentes em séries na base.
- País: '{hyp_series['country']}' é um dos principais produtores de séries na Netflix.
- Temporadas: '{hyp_series['seasons']}' foi definido a partir da média de temporadas observada 
  na coluna 'duration' (ex.: "3 Seasons").
- Rating: '{hyp_series['rating']}' segue a classificação com maior frequência em séries.

Conteúdo (ideia):
Uma série dramática contemporânea em {hyp_series['country']}, com arcos planejados para 
cerca de {hyp_series['seasons']} temporadas, explorando temas típicos das produções mais populares 
do catálogo nesse gênero.
"""

# -----------------------------------------------------------------------------
# 8) Exibir resultados finais organizados
# -----------------------------------------------------------------------------
print(hyp_movie_description)
print(hyp_series_description)

spark.stop()

# analyze_netflix.py
from pyspark.sql import SparkSession, functions as F

spark = (
    SparkSession.builder
    .appName("NetflixAnalysis")
    .getOrCreate()
)

# Silencia os logs verbosos do Spark (deixa só erro)
spark.sparkContext.setLogLevel("ERROR")

# ================== CARREGAR DATASET ==================
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", False)
    .csv("/opt/spark/data/netflix_titles.csv")
    .filter(F.length(F.trim(F.col("show_id"))) > 0)
)

print("\n====================== ESQUEMA DO DATASET ======================")
df.printSchema()

# ================== CONTAGEM POR TIPO ==================
print("\n\n================== CONTAGEM POR TIPO (Movie x TV Show) ==================")
type_counts = (
    df.groupBy("type")
      .count()
      .orderBy(F.desc("count"))
)
type_counts.show(truncate=False)

# ================== SEPARAR FILMES E SÉRIES ==================
movies = df.filter(F.col("type") == "Movie")
series = df.filter(F.col("type") == "TV Show")

# ================== VALORES VÁLIDOS DE RATING ==================
valid_ratings = [
    "TV-MA", "TV-14", "TV-PG", "TV-Y", "TV-Y7", "TV-Y7-FV",
    "TV-G", "G", "PG", "PG-13", "R", "NC-17", "NR", "UR"
]

movies_for_ratings = movies.filter(F.col("rating").isin(valid_ratings))
series_for_ratings = series.filter(F.col("rating").isin(valid_ratings))

# --------------------------------------------------------------------
# 1) ANÁLISE DE FILMES
# --------------------------------------------------------------------

# --------- Gêneros mais frequentes em filmes ---------
movie_genres = (
    movies
    .withColumn("genre", F.explode(F.split(F.col("listed_in"), ",")))
    .withColumn("genre", F.trim("genre"))
    .filter(F.col("genre") != "")
    .groupBy("genre")
    .count()
    .orderBy(F.desc("count"))
)

print("\n\n================== GÊNEROS MAIS FREQUENTES EM FILMES ==================")
movie_genres.show(10, truncate=False)

# --------- Países com mais filmes ---------
movie_countries = (
    movies
    .withColumn("country_exploded", F.explode(F.split(F.col("country"), ",")))
    .withColumn("country_exploded", F.trim("country_exploded"))
    .filter(F.col("country_exploded") != "")
    .groupBy("country_exploded")
    .count()
    .orderBy(F.desc("count"))
)

print("\n\n================== PAÍSES COM MAIS FILMES ==================")
movie_countries.show(10, truncate=False)

# --------- Ratings mais comuns em filmes (APENAS VÁLIDOS) ---------
movie_ratings = (
    movies_for_ratings
    .groupBy("rating")
    .count()
    .orderBy(F.desc("count"))
)

print("\n\n================== RATINGS MAIS COMUNS EM FILMES ==================")
movie_ratings.show(10, truncate=False)

# --------- Duração média dos filmes (minutos) ---------
movies_with_duration = (
    movies
    .withColumn("duration_str", F.regexp_extract("duration", r"(\d+)", 1))
    .withColumn(
        "duration_num",
        F.when(
            F.col("duration_str").rlike("^[0-9]+$"),
            F.col("duration_str").cast("int")
        ).otherwise(F.lit(None).cast("int"))
    )
)

avg_movie_duration_row = (
    movies_with_duration
    .select(F.avg("duration_num").alias("avg_duration"))
    .first()
)

avg_movie_duration_exact = avg_movie_duration_row["avg_duration"]
avg_movie_duration = int(round(avg_movie_duration_exact)) if avg_movie_duration_exact is not None else 0

print("\n\n================== DURAÇÃO MÉDIA DOS FILMES (min) ==================")
if avg_movie_duration_exact is not None:
    print(f"Duração média exata dos filmes: {avg_movie_duration_exact:.2f} minutos")
    print(f"Duração média aproximada (arredondada): {avg_movie_duration} minutos\n")
else:
    print("Não foi possível calcular a duração média (dados ausentes).\n")

# --------------------------------------------------------------------
# 2) ANÁLISE DE SÉRIES
# --------------------------------------------------------------------

# --------- Gêneros mais frequentes em séries ---------
series_genres = (
    series
    .withColumn("genre", F.explode(F.split(F.col("listed_in"), ",")))
    .withColumn("genre", F.trim("genre"))
    .filter(F.col("genre") != "")
    .groupBy("genre")
    .count()
    .orderBy(F.desc("count"))
)

print("\n\n================== GÊNEROS MAIS FREQUENTES EM SÉRIES ==================")
series_genres.show(10, truncate=False)

# --------- Países com mais séries ---------
series_countries = (
    series
    .withColumn("country_exploded", F.explode(F.split(F.col("country"), ",")))
    .withColumn("country_exploded", F.trim("country_exploded"))
    .filter(F.col("country_exploded") != "")
    .groupBy("country_exploded")
    .count()
    .orderBy(F.desc("count"))
)

print("\n\n================== PAÍSES COM MAIS SÉRIES ==================")
series_countries.show(10, truncate=False)

# --------- Ratings mais comuns em séries (APENAS VÁLIDOS) ---------
series_ratings = (
    series_for_ratings
    .groupBy("rating")
    .count()
    .orderBy(F.desc("count"))
)

print("\n\n================== RATINGS MAIS COMUNS EM SÉRIES ==================")
series_ratings.show(10, truncate=False)

# --------- Número médio de temporadas ---------
series_with_seasons = (
    series
    .withColumn("seasons_str", F.regexp_extract("duration", r"(\d+)", 1))
    .withColumn(
        "seasons_num",
        F.when(F.col("seasons_str").rlike("^[0-9]+$"),
               F.col("seasons_str").cast("int"))
         .otherwise(F.lit(None).cast("int"))
    )
)

avg_seasons_row = (
    series_with_seasons
    .select(F.avg("seasons_num").alias("avg_seasons"))
    .first()
)

avg_seasons_exact = avg_seasons_row["avg_seasons"]
avg_seasons = int(round(avg_seasons_exact)) if avg_seasons_exact is not None else 0

print("\n\n================== NÚMERO MÉDIO DE TEMPORADAS ==================")
if avg_seasons_exact is not None:
    print(f"Número médio exato de temporadas: {avg_seasons_exact:.2f}")
    print(f"Número médio aproximado de temporadas: {avg_seasons}\n")
else:
    print("Não foi possível calcular o número médio de temporadas.\n")

# --------------------------------------------------------------------
# 3) CRIAÇÃO DE FILME E SÉRIE HIPOTÉTICOS
# --------------------------------------------------------------------

# Top 1 de cada agrupamento para guiar a sugestão
top_movie_genre_row = movie_genres.first()
top_movie_country_row = movie_countries.first()
top_movie_rating_row = movie_ratings.first()

top_movie_genre = top_movie_genre_row["genre"] if top_movie_genre_row else "International Movies"
top_movie_country = top_movie_country_row["country_exploded"] if top_movie_country_row else "United States"
top_movie_rating = top_movie_rating_row["rating"] if top_movie_rating_row else "TV-MA"

top_series_genre_row = series_genres.first()
top_series_country_row = series_countries.first()
top_series_rating_row = series_ratings.first()

top_series_genre = top_series_genre_row["genre"] if top_series_genre_row else "International TV Shows"
top_series_country = top_series_country_row["country_exploded"] if top_series_country_row else "United States"
top_series_rating = top_series_rating_row["rating"] if top_series_rating_row else "TV-MA"

# --------- FILME HIPOTÉTICO ---------
print("\n\n================== FILME HIPOTÉTICO (BASEADO NO DATASET) ==================\n")

hyp_movie_title = f"Sombras em {top_movie_country}"

print(f"Título: {hyp_movie_title}")
print("Tipo: Movie")
print(f"Gênero principal: {top_movie_genre}")
print(f"País de produção: {top_movie_country}")
print(f"Duração aproximada: {avg_movie_duration} minutos")
print(f"Classificação indicativa: {top_movie_rating}")
print("Ano de lançamento: 2025\n")

print("Justificativa:")
print(f"- Gênero escolhido: '{top_movie_genre}' está entre os gêneros de filme mais frequentes no dataset.")
print(f"- País: '{top_movie_country}' aparece entre os países que mais produzem filmes na base.")
print("- Duração: usamos a média de duração dos filmes calculada via Spark (coluna 'duration').")
print(f"- Rating: '{top_movie_rating}' está entre as classificações mais comuns nos filmes.\n")

print("Conteúdo (ideia):")
print("Um filme voltado ao público que mais consome esse tipo de produção na plataforma,")
print("aproveitando o gênero e o país que já têm alta aceitação, mas com uma narrativa original.\n")

# --------- SÉRIE HIPOTÉTICA ---------
print("\n================== SÉRIE HIPOTÉTICA (BASEADA NO DATASET) ==================\n")

hyp_series_title = f"Conexões em {top_series_country}"

print(f"Título: {hyp_series_title}")
print("Tipo: TV Show")
print(f"Gênero principal: {top_series_genre}")
print(f"País de produção: {top_series_country}")
print(f"Número previsto de temporadas: {avg_seasons}")
print(f"Classificação indicativa: {top_series_rating}")
print("Ano de lançamento: 2025\n")

print("Justificativa:")
print(f"- Gênero: '{top_series_genre}' aparece entre os gêneros mais recorrentes em séries na base.")
print(f"- País: '{top_series_country}' é um dos principais produtores de séries na Netflix.")
print(f"- Temporadas: '{avg_seasons}' foi definido a partir da média de temporadas observada na coluna 'duration'.")
print(f"- Rating: '{top_series_rating}' segue a classificação com maior frequência em séries.\n")

print("Conteúdo (ideia):")
print(f"Uma série dramática contemporânea em {top_series_country}, com arcos planejados para")
print(f"cerca de {avg_seasons} temporadas, explorando temas típicos das produções mais populares")
print("do catálogo nesse gênero.\n")

# --------------------------------------------------------------------
# 4) SUGESTÃO DE FILME REAL E SÉRIE REAL A PARTIR DO DATASET
# --------------------------------------------------------------------

print("\n================== FILME REAL SUGERIDO (BASEADO NO DATASET) ==================\n")

real_movie_candidates = movies

# tentar aproximar do perfil encontrado (gênero, país, rating)
real_movie_candidates = real_movie_candidates.filter(
    F.col("listed_in").contains(top_movie_genre)
)

real_movie_candidates = real_movie_candidates.filter(
    F.col("country").contains(top_movie_country)
)

real_movie_candidates = real_movie_candidates.filter(
    F.col("rating") == top_movie_rating
)

real_movie = (
    real_movie_candidates
    .withColumn(
        "release_year_int",
        F.when(F.col("release_year").rlike("^[0-9]+$"),
               F.col("release_year").cast("int"))
         .otherwise(F.lit(None).cast("int"))
    )
    .orderBy(F.desc("release_year_int"))
    .select("title", "country", "listed_in", "rating", "duration", "release_year_int")
    .first()
)

if real_movie:
    print(f"Título: {real_movie['title']}")
    print("Tipo: Movie")
    print(f"País(es): {real_movie['country']}")
    print(f"Gênero(s): {real_movie['listed_in']}")
    print(f"Duração: {real_movie['duration']}")
    print(f"Classificação indicativa: {real_movie['rating']}")
    print(f"Ano de lançamento: {real_movie['release_year_int']}\n")
else:
    print("Não foi possível encontrar um filme com todos os critérios (gênero, país e rating).")
    print("Ainda assim, os padrões extraídos podem ser usados para guiar decisões de catálogo.\n")

print("\n================== SÉRIE REAL SUGERIDA (BASEADO NO DATASET) ==================\n")

real_series_candidates = series

real_series_candidates = real_series_candidates.filter(
    F.col("listed_in").contains(top_series_genre)
)

real_series_candidates = real_series_candidates.filter(
    F.col("country").contains(top_series_country)
)

real_series_candidates = real_series_candidates.filter(
    F.col("rating") == top_series_rating
)

real_series = (
    real_series_candidates
    .withColumn(
        "release_year_int",
        F.when(F.col("release_year").rlike("^[0-9]+$"),
               F.col("release_year").cast("int"))
         .otherwise(F.lit(None).cast("int"))
    )
    .orderBy(F.desc("release_year_int"))
    .select("title", "country", "listed_in", "rating", "duration", "release_year_int")
    .first()
)

if real_series:
    print(f"Título: {real_series['title']}")
    print("Tipo: TV Show")
    print(f"País(es): {real_series['country']}")
    print(f"Gênero(s): {real_series['listed_in']}")
    print(f"Duração (texto original): {real_series['duration']}")
    print(f"Classificação indicativa: {real_series['rating']}")
    print(f"Ano de lançamento: {real_series['release_year_int']}\n")
else:
    print("Não foi possível encontrar uma série com todos os critérios (gênero, país e rating).")
    print("Ainda assim, os padrões extraídos podem ser usados para guiar decisões de catálogo.\n")

spark.stop()

from pyspark.sql import SparkSession, SQLContext

spark = SparkSession.builder \
    .appName("hw2_q5") \
    .config("spark.jars", "/Users/wuyiman/Downloads/postgresql-42.7.7.jar") \
    .getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

df = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/14763hw2") \
    .option("dbtable", "news.articles") \
    .option("user", "wuyiman") \
    .option("password", "") \
    .option("driver", "org.postgresql.Driver") \
    .load()

df.createOrReplaceTempView("articles")

to_delete = sqlContext.sql("""
    SELECT COUNT(*) AS cnt
    FROM articles
    WHERE lower(title) LIKE '%nfl%'
""").collect()[0].cnt

print(f"Deleted rows: {to_delete}")

remaining = sqlContext.sql("""
    SELECT *
    FROM articles
    WHERE lower(title) NOT LIKE '%nfl%'
""")

remaining.write \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/14763hw2") \
    .option("dbtable", "news.articles") \
    .option("user", "wuyiman") \
    .option("password", "") \
    .option("driver", "org.postgresql.Driver") \
    .mode("overwrite") \
    .save()
check = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/14763hw2") \
    .option("dbtable", "news.articles") \
    .option("user", "wuyiman") \
    .option("password", "") \
    .option("driver", "org.postgresql.Driver") \
    .load()

check.createOrReplaceTempView("articles2")

df_check = sqlContext.sql("""
    SELECT COUNT(*) AS cnt
    FROM articles2
    WHERE lower(title) LIKE '%nfl%'
""")
df_check.show()
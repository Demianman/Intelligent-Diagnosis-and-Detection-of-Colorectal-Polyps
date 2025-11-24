from urllib.request import Request, urlopen
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
import feedparser
from dateutil import parser
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone

spark = SparkSession.builder \
    .appName("hw3_q3") \
    .config("spark.jars", "/Users/wuyiman/Downloads/postgresql-42.7.7.jar") \
    .getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

url = "https://news.google.com/rss/search?q=technology&hl=en-US&gl=US&ceid=US:en"
req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
data = urlopen(req, timeout=20).read()
feed = feedparser.parse(data)

rows = []
last_build_date = feed.feed.get("updated") or feed.feed.get("published")

for entry in feed.entries:
    title = entry.get("title")
    link = entry.get("link")
    pub_date = parser.parse(entry.get("published")) if entry.get("published") else None
    desc_raw = entry.get("description", "")
    description = BeautifulSoup(desc_raw, "html.parser").get_text()
    source = (entry.get("source") or {}).get("title")
    rows.append((last_build_date, title, link, pub_date, description, source, "technology"))

schema = StructType([
    StructField("lastBuildDate", StringType(), True),
    StructField("title",        StringType(), True),
    StructField("link",         StringType(), True),
    StructField("pubDate",      TimestampType(), True),
    StructField("description",  StringType(), True),
    StructField("source",       StringType(), True),
    StructField("category",     StringType(), True),
])

if rows:
    df = spark.createDataFrame(rows, schema=schema)
else:
    df = spark.createDataFrame(sc.emptyRDD(), schema)

df.createOrReplaceTempView("articles")

now = datetime.now(timezone.utc)
yesterday = now - timedelta(days=1)

now_str = now.strftime("%Y-%m-%d %H:%M:%S")
yesterday_str = yesterday.strftime("%Y-%m-%d %H:%M:%S")

query = f"""
    SELECT pubDate, title, source
    FROM articles
    WHERE pubDate BETWEEN TIMESTAMP('{yesterday_str}') AND TIMESTAMP('{now_str}')
"""
df_sql = sqlContext.sql(query)

results = df_sql.collect()
for row in results:
    print(f"{row['pubDate']}")
    print(f"Title: {row['title']}")
    print(f"Source: {row['source']}\n")

df_sql.write \
  .format("jdbc") \
  .option("url", "jdbc:postgresql://localhost:5432/14763hw2") \
  .option("dbtable", "news.articles_last24h") \
  .option("user", "wuyiman") \
  .option("password", "") \
  .option("driver", "org.postgresql.Driver") \
  .mode("overwrite") \
  .save()

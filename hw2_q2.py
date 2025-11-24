from urllib.request import Request, urlopen
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
import feedparser
from dateutil import parser
from bs4 import BeautifulSoup

spark = SparkSession.builder \
    .appName("hw2_q2") \
    .config("spark.jars", "/Users/wuyiman/Downloads/postgresql-42.7.7.jar") \
    .getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

schema = StructType([
    StructField("lastBuildDate", TimestampType(), True),
    StructField("title",        StringType(), True),
    StructField("link",         StringType(), True),
    StructField("pubDate",      TimestampType(), True),
    StructField("description",  StringType(), True),
    StructField("source",       StringType(), True),
    StructField("category",     StringType(), True),
])

def to_naive_datetime(date_str):
    if not date_str:
        return None
    try:
        dt = parser.parse(date_str)
        return dt.replace(tzinfo=None)
    except Exception:
        return None

def fetch_feed(url, category):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = urlopen(req, timeout=20).read()
    feed = feedparser.parse(data)

    rows = []
    raw_last_build = feed.feed.get("updated") or feed.feed.get("published")
    last_build_date = to_naive_datetime(raw_last_build)

    for entry in feed.entries:
        title = entry.get("title")
        link = entry.get("link")
        pub_date = to_naive_datetime(entry.get("published"))
        desc_raw = entry.get("description", "")
        description = BeautifulSoup(desc_raw, "html.parser").get_text()
        source = (entry.get("source") or {}).get("title")
        rows.append((last_build_date, title, link, pub_date, description, source, category))

    if rows:
        return spark.createDataFrame(rows, schema=schema)
    else:
        return spark.createDataFrame(sc.emptyRDD(), schema)

url = "https://news.google.com/rss/search?q=technology&hl=en-US&gl=US&ceid=US:en"
df = fetch_feed(url, "technology")
df.createOrReplaceTempView("articles")

df_sql = sqlContext.sql("SELECT * FROM articles")

df_sql.write \
  .format("jdbc") \
  .option("url", "jdbc:postgresql://localhost:5432/14763hw2") \
  .option("dbtable", "news.articles") \
  .option("user", "wuyiman") \
  .option("password", "") \
  .option("driver", "org.postgresql.Driver") \
  .mode("append") \
  .save()

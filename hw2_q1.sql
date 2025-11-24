CREATE TABLE news.articles (
    identifier SERIAL PRIMARY KEY,
    lastBuildDate TIMESTAMP WITH TIME ZONE,
    Title TEXT,
    Link TEXT,
    PubDate TIMESTAMP WITH TIME ZONE,
    Description TEXT,
    Source TEXT
);

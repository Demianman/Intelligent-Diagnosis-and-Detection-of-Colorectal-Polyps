ALTER TABLE news.articles
ADD COLUMN IF NOT EXISTS category text DEFAULT 'technology';
UPDATE news.articles
SET category = 'technology'
WHERE category IS NULL;
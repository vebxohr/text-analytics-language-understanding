from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup


driver = webdriver.Chrome(ChromeDriverManager().install())

driver.get('https://old.reddit.com')
content = driver.page_source

soup = BeautifulSoup(content, features="html.parser")

posts = soup.select('div[class*="thing"]')

for post in posts:
    # Skip promotions (no timestamp, subreddit or votes)
    if "promoted" in post.attrs.get("class"):
        continue
    title = post.select('p[class="title"] > a')[0].contents[0]
    score = post.select('div[class="score likes"]')[0].contents[0]
    tagline = post.select('p[class="tagline"]')[0]
    time = tagline.select('time')[0].attrs.get("datetime")
    subreddit = tagline.select('a[class*="subreddit"]')[0].contents[0]

    print(title)
    print(f'Upvotes: {score}')
    print(time)
    print(subreddit)
    print("\n")



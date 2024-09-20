from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

class MetacriticReviewScraper:
    def __init__(self):
        # Set up Chrome options for headless mode
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")  # Run browser in headless mode
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration
        self.chrome_options.add_argument("--window-size=1920x1080")  # Set window size for headless mode
        self.driver = webdriver.Chrome(options=self.chrome_options)

    def fetch_reviews(self, game_name, platforms, review_limit):
        reviews = []

        for platform in platforms:
            game_url = self.format_game_url(game_name, platform)
            self.load_page(game_url)
            platform_reviews = self.extract_reviews(platform)

            # Append formatted reviews to the main list
            reviews.extend(platform_reviews[:review_limit])

        self.driver.quit()
        return reviews

    def format_game_url(self, game_name, platform):
        game_name_url = game_name.lower().replace(" ", "-")
        return f"https://www.metacritic.com/game/{game_name_url}/user-reviews/?platform={platform}"

    def load_page(self, url):
        self.driver.get(url)
        time.sleep(2)  # Wait for the page to load

    def extract_reviews(self, platform):
        reviews = []
        try:
            review_elements = self.driver.find_elements(By.CLASS_NAME, "c-siteReview_quote.g-outer-spacing-bottom-small")
            for review_element in review_elements:
                review_span = review_element.find_element(By.TAG_NAME, "span")
                review_text = review_span.text.strip()
                if review_text:
                    reviews.append(f"{review_text}")  # Format the review text as requested
        except Exception as e:
            print(f"Error while scraping {platform} reviews: {e}")
        return reviews
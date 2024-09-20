import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

class IMDBReviewScraper:
    def __init__(self, api_key):
        self.api_key = api_key
        self.driver = None

    def get_imdb_id(self, movie_name):
        """
        Fetch the IMDb ID for a movie based on its name.
        """
        url = f"http://www.omdbapi.com/?t={movie_name}&apikey={self.api_key}"
        response = requests.get(url)
        data = response.json()

        if data['Response'] == 'True':
            return data['imdbID']
        else:
            return None

    def setup_driver(self):
        """
        Set up the Selenium WebDriver with headless mode.
        """
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(options=chrome_options)

    def get_imdb_reviews(self, movie_id, review_count):
        """
        Fetch reviews for a given IMDb movie ID, returning a list of unique reviews.
        """
        base_url = f"https://www.imdb.com/title/{movie_id}/reviews/?ref_=tt_urv"
        reviews = set()  # Use a set to store unique reviews
        
        try:
            self.setup_driver()
            self.driver.get(base_url)
            

            # Wait for the reviews to load initially
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "text.show-more__control"))
            )

            # Scroll down and click "Load More" to gather reviews
            while len(reviews) < review_count:
                review_elements = self.driver.find_elements(By.CLASS_NAME, "text.show-more__control")
                

                for review_element in review_elements:
                    review_text = review_element.text
                    if review_text:
                        reviews.add(review_text)

                if len(reviews) >= review_count:
                    break
                
                # Click the load more button
                try:
                    load_more_button = self.driver.find_element(By.CLASS_NAME, "ipl-load-more__button")
                    self.driver.execute_script("arguments[0].scrollIntoView();", load_more_button)
                    time.sleep(1)
                    load_more_button.click()
                    
                    time.sleep(2)
                except Exception as e:
                    print(f"No more reviews to load or an error occurred: {e}")
                    break

        except Exception as e:
            print(f"An error occurred during the scraping process: {e}")
        finally:
            self.driver.quit()

        return list(reviews)  # Convert set to list and return
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

class RottenTomatoesReviewScraper:
    def __init__(self):
        self.driver = None

    def setup_driver(self):
        """
        Set up the Selenium WebDriver with headless mode.
        """
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run browser in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(options=chrome_options)

    def get_rotten_tomatoes_reviews(self, movie_slug, review_count):
        """
        Fetch reviews from Rotten Tomatoes for a given movie slug.
        Returns a list of unique review texts.
        """
        movie_name_formatted = movie_slug.lower().replace(' ', '_')
        base_url = f"https://www.rottentomatoes.com/m/{movie_name_formatted}/reviews"
        reviews = set()  # Use a set to store unique reviews
        
        try:
            self.setup_driver()
            self.driver.get(base_url)
        

            # Wait for the reviews to load initially
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "review-text-container"))
            )

            # Scroll down and click "Load More" to gather reviews
            while len(reviews) < review_count:
                self.load_more_reviews()
                
                # Find all review elements after loading more
                review_elements = self.driver.find_elements(By.CLASS_NAME, "review-text-container")
                

                for review_element in review_elements:
                    try:
                        # Extract the text from the <p> tag with class 'review-text'
                        review_text = review_element.find_element(By.CLASS_NAME, "review-text").text
                        reviews.add(review_text)  # Add review text to the set for uniqueness
                    except Exception as e:
                        print(f"Error extracting review text: {e}")

                    # Stop if we've reached the desired number of unique reviews
                    if len(reviews) >= review_count:
                        break

        except Exception as e:
            print(f"An error occurred during the scraping process: {e}")
        finally:
            # Close the Selenium browser session
            self.driver.quit()

        return list(reviews)  # Convert set to list and return

    def load_more_reviews(self):
        """
        Clicks the "Load More" button to load additional reviews.
        """
        try:
            load_more_button = self.driver.find_element(By.CLASS_NAME, "load-more-container")
            self.driver.execute_script("arguments[0].scrollIntoView();", load_more_button)
            time.sleep(1)  # Wait for the scroll to complete
            
            # Click the load more button
            load_more_button.click()
            time.sleep(2)  # Wait for more reviews to load
        except Exception as e:
            print(f"No more reviews to load or an error occurred: {e}")

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

class LetterboxdReviewScraper:
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

    def get_letterboxd_reviews(self, movie_name, review_count):
        """
        Fetch reviews from Letterboxd for a given movie name.
        Returns a list of review texts.
        """
        movie_name_formatted = movie_name.lower().replace(' ', '-')
        base_url = f"https://letterboxd.com/film/{movie_name_formatted}/reviews/by/activity/"
        reviews = []
        page = 1
        
        try:
            self.setup_driver()
            while len(reviews) < review_count:
                # Navigate to the page (with pagination if needed)
                url = base_url if page == 1 else f"{base_url}page/{page}/"
                self.driver.get(url)
                
                try:
                    # Wait for the reviews to load
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "body-text"))
                    )

                    # Find all review elements
                    review_elements = self.driver.find_elements(By.CLASS_NAME, 'body-text')

                    if review_elements:
                        for review_element in review_elements:
                            # Extract text within the <p> tags
                            review_paragraphs = review_element.find_elements(By.TAG_NAME, 'p')
                            review_text = " ".join([p.text for p in review_paragraphs]).strip()
                            reviews.append(review_text[:300])  # Store the review (limited to 300 characters)
                            
                            # Check if we've collected enough reviews
                            if len(reviews) >= review_count:
                                break
                    else:
                        print("No reviews found on this page.")
                        
                except Exception as e:
                    print(f"Error occurred: {e}")

                # Add a short delay between page requests
                time.sleep(2)

                # Move to the next page
                page += 1

        finally:
            # Close the Selenium browser session
            if self.driver:
                self.driver.quit()

        return reviews  # Return the list of reviews
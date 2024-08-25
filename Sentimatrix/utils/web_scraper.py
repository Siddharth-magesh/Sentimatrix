import requests
from bs4 import BeautifulSoup
from .review_patterns import review_patterns

def Fetch_Reviews(url : str , Use_Local_Scraper : bool , Use_Scraper_API : bool , Scraper_api_key : str , Local_api_key : str):
    """
    Fetches the reviews from the provided URL using either the local scraper or ScraperAPI,
    depending on the initialized settings.
    
    Args:
        url (str): The URL of the product page to scrape.
    
    Returns:
        list: A list of review strings extracted from the webpage.
    """
    if Use_Local_Scraper == True:
        return fetch_reviews_locally(url,Local_api_key)
    elif Use_Scraper_API == True:
        return fetch_reviews_with_scraperAPI(url,Scraper_api_key)
    else:
        raise ValueError("You must enable either local scraper or ScraperAPI , Use_Local_Scraper = True")
    
def fetch_reviews_locally(url : str, Local_api_key : str) -> list:
    """
    Fetches the reviews from the provided URL using direct local scraping.
    
    Args:
        url (str): The URL of the product page to scrape.
    
    Returns:
        list: A list of review strings extracted from the webpage.
    """
    HEADERS = ({
        'User-Agent': Local_api_key,
        'Accept-Language': 'en-US,en;q=0.5'
    })
    response = requests.get(
        url=url,
        headers=HEADERS
    )
    reviews_list = []
    if response.status_code == 200:
        soup = BeautifulSoup(response.content , "html.parser")

        for pattern in review_patterns:
            reviews = soup.find_all(pattern['tag'],pattern['attrs'])
            for review in reviews:
                review_text = review.text.split()
                if review_text:
                    reviews_list.append(review_text)
    else :
        print(f"Failed to retrieve page: Status code {response.status_code}")

    list_of_sentences = [' '.join(sublist) for sublist in reviews_list]

    return list_of_sentences

def fetch_reviews_with_scraperAPI( url : str , Scraper_api_key : str) -> list:
    """
    Fetches the reviews from the provided URL using ScraperAPI.
    
    Args:
        url (str): The URL of the product page to scrape.
    
    Returns:
        list: A list of review strings extracted from the webpage.
    """
    if not Scraper_api_key:
        raise ValueError("API key is required for ScraperAPI.")
    
    payload = {'api_key': Scraper_api_key, 'url': url}
    response = requests.get('https://api.scraperapi.com/', params=payload)

    reviews_list = []

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        for pattern in review_patterns:
            reviews = soup.find_all(pattern['tag'], pattern['attrs'])
            for review in reviews:
                review_text = review.text.strip()
                if review_text:
                    reviews_list.append(review_text)
    else:
        print(f"Failed to retrieve page: Status code {response.status_code}")
    
    return reviews_list

def add_review_pattern(tag : str, attrs : dict):
    """
    Adds a new review pattern to the scraper.
    
    Args:
        tag (str): The HTML tag name (e.g., 'div', 'span').
        attrs (dict): A dictionary of HTML attributes to match (e.g., {'class': 'review-body'}).
    """
    new_pattern = {'tag': tag, 'attrs': attrs}
    review_patterns.append(new_pattern)
    print(f"Added new pattern: {new_pattern}")


def get_review_patterns() -> list:
    """
    Returns the current list of review patterns.
    
    Returns:
        list: The list of review patterns.
    """
    return review_patterns
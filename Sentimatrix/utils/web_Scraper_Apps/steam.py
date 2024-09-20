import requests

class SteamReviewScraper:
    def __init__(self):
        self.base_search_url = "https://store.steampowered.com/api/storesearch/"
        self.base_reviews_url = "https://store.steampowered.com/appreviews/"

    def get_steam_app_id(self, game_name):
        """
        Fetch the Steam app ID for a game based on its name.
        """
        params = {
            'term': game_name,
            'l': 'english',  # Language (optional)
            'cc': 'us'       # Country (optional)
        }
        response = requests.get(self.base_search_url, params=params)

        if response.status_code != 200:
            print(f"Error fetching app search: {response.status_code}")
            return None
        
        search_data = response.json()

        if search_data.get('total', 0) == 0:
            print("No games found with that name.")
            return None
        
        app_id = search_data['items'][0]['id']
        game_title = search_data['items'][0]['name']
        
        return app_id, game_title

    def get_steam_reviews(self, app_id, review_limit):
        """
        Fetch the reviews for a given Steam app ID, returning only the comments as a list of strings.
        """
        params = {
            'json': 1,
            'num_per_page': review_limit,
            'language': 'english',  # Set language (optional)
            'filter': 'all',        # Filter type: all, recent, etc.
            'purchase_type': 'all'  # All reviews, verified, etc.
        }
        
        reviews_response = requests.get(f"{self.base_reviews_url}{app_id}", params=params)

        if reviews_response.status_code != 200:
            print(f"Error fetching reviews: {reviews_response.status_code}")
            return []
        
        reviews_data = reviews_response.json()

        if reviews_data.get('success', 0) != 1:
            print("Failed to fetch reviews.")
            return []

        reviews = reviews_data['reviews']
        if not reviews:
            print("No reviews found.")
            return []

        # Extract and return the review comments as a list of strings
        review_comments = [review['review'] for review in reviews[:review_limit]]
        
        return review_comments

    def fetch_reviews_for_game(self, game_name, review_limit):
        """
        Fetch reviews for a game, returning only the comments as a list of strings.
        """
        app_id, game_title = self.get_steam_app_id(game_name)
        if app_id:
            return self.get_steam_reviews(app_id, review_limit)
        else:
            print("Could not find the game. Please try again.")
            return []


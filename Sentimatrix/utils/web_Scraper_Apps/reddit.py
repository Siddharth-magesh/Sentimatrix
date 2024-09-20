import praw
import time

class RedditScraper:
    def __init__(self, reddit_client_id, reddit_client_secret, reddit_user_agent):
        self.reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent,
        )

    def search_reddit_for_product(self, product_name, limit=1000):
        search_query = product_name
        posts_data = []
        after = None  # Pagination key

        while len(posts_data) < limit:
            try:
                search_results = self.reddit.subreddit('all').search(search_query, limit=100, params={'after': after})
                batch_fetched = False

                for submission in search_results:
                    post_info = submission.selftext  # Extract the post text
                    if post_info:  # Add only non-empty posts
                        posts_data.append(post_info)
                    after = submission.name
                    batch_fetched = True

                if not batch_fetched:
                    break

            except Exception as e:
                print(f"An error occurred: {e}")
                time.sleep(2)
                continue

        return posts_data[:limit]  # Return a flattened list of post texts


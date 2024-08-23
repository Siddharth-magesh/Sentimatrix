#Quick Sentiment Unit Test
'''from Sentimatrix.sentiment_generation import SentConfig

Sent = SentConfig(
    Use_Local_Sentiment_LLM = True,
)
sentiments = ["i am very happy","i am very sad","i am alright"]
sentiment_result = Sent.get_Quick_sentiment('J. Wert5.0 out of 5 stars\nI like it\nReviewed in the United States on February 10, 2024Size: TKLVerified Purchase\nDecent keyboard. Came from a huntsman v2 tkl tournament edition. Did alot of research on keyboards, biggest complaints about this was Razers software, but hey I was already using it.It is a bit loud. I am looking into possible ways to deaden the sound. The rapid fire does work. I have not had any issues with micro shudders in game. (Others have mention this being an issue) mainly play gta v and apex legends. Very responsive key presses. Feels solid and decent built. Overall nice keyboard, specially if looking for rapid trigger and customization. Yes, a wooting or something like may be better. But, I wam happy with this purchase.\nRead more\n9 people found this helpful\n\n\n              Helpful\n\n\nReport',device_map="cpu")

print(sentiment_result)'''

#Initial Test For Web Scraper
'''from Sentimatrix.utils.web_scraper import ReviewScraper

scraper = ReviewScraper(Use_Local_Scraper=True)

# Example Amazon product URL (you can use any valid e-commerce page URL)
url = "https://www.amazon.com/Razer-Huntsman-Esports-Gaming-Keyboard/dp/B0CG7FQML2"

# Fetch reviews using the local scraper
reviews_local = scraper.fetch_reviews(url)
#print("Reviews fetched locally:", reviews_local)

list_of_sentences = [' '.join(sublist) for sublist in reviews_local]

# Print the result
for sentence in list_of_sentences:
    print(sentence)'''
'''# Add a new review pattern dynamically
scraper.add_review_pattern('div', {'class': 'new-review-class'})

# Check the current review patterns
current_patterns = scraper.get_review_patterns()
print("Current review patterns:", current_patterns)'''

#Final WebScraper Test
'''from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Scraper_API = True,
    Scraper_api_key= ""
)
Sent.get_sentiment_from_website_each_feedback_sentiment('https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1')
'''
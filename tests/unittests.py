# Quick Sentiment Unit Test -->Worked
'''from Sentimatrix.sentiment_generation import SentConfig

Sent = SentConfig(
    Use_Local_Sentiment_LLM = True,
)
sentiments = ["i am very happy","i am very sad","i am alright"]
sentiment_result = Sent.get_Quick_sentiment(text_message=sentiments,device_map="auto")

print(sentiment_result)'''

# Initial Test For Web Scraper -->Worked
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

# Final WebScraper Test -->Worked
'''from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Scraper_API = True,
    Scraper_api_key= "7ebf4f26faa024ef86d97279c16c2a0c"
)
result = Sent.get_sentiment_from_website_each_feedback_sentiment('https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1')
print(result)'''

# Initial Website Sentiment Test without groq --> Worked Without using Scraper API also
'''from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_Sentiment_LLM= True,
    device_map = "auto" 
)
target = 'https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1'
result = Sent.get_sentiment_from_website_each_feedback_sentiment(
    target_website=target,
    Use_Local_Scraper= True,
    get_Groq_Review= False
)

print(result)'''

# Initial Website Sentiment Test without groq --> Worked With using Scraper API also
'''from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_Sentiment_LLM= True,
    Use_Gemini_API=True,
    device_map = "auto" 
)
target = 'https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1'
result = Sent.get_sentiment_from_website_each_feedback_sentiment(
    target_website=target,
    Use_Local_Scraper=True,
    Use_Scraper_API=False,
    get_Gemini_Review=True,
    Google_API=""
)

print(result)'''

# Multi Site Scraper
'''from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_Sentiment_LLM= True,
    device_map = "auto" 
)
target = ['https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1','https://www.amazon.com/Legendary-Whitetails-Journeyman-Jacket-Tarmac/dp/B013KW38RQ/ref=cm_cr_arp_d_product_top?ie=UTF8']
result = Sent.get_sentiment_from_website_each_feedback_sentiment(
    target_website=target,
    Use_Local_Scraper= True,
    get_Groq_Review= False
)

print(result)'''

# Final UnitTest For get_sentiment_from_website_each_feedback_sentiment -->Works For Single Site and Multisite
'''from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Groq_API=True,
    Groq_API="",
    Use_Local_Sentiment_LLM=True,
    device_map="auto",
    Use_Local_Scraper=True
)
target = ['https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1','https://www.amazon.com/Legendary-Whitetails-Journeyman-Jacket-Tarmac/dp/B013KW38RQ/ref=cm_cr_arp_d_product_top?ie=UTF8']
result = Sent.get_sentiment_from_website_each_feedback_sentiment(
    target_website=target,
    get_Groq_Review=True
)

print(result)'''

# UnitTest For get_sentiment_from_website_each_feedback_sentiment Using Local LLM --> Done ,Infernce result is bad
'''from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Groq_API=True,
    Use_Local_Sentiment_LLM=True,
    device_map="auto",
    Use_Local_Scraper=True
)
target = 'https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1'
result = Sent.get_sentiment_from_website_each_feedback_sentiment(
    target_website=target,
    get_localLLM_review=True
)

print(result)'''

# UnitTest for get_sentiment_from_website_overall_summary using Groq
'''from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Groq_API=True,
    Use_Local_Scraper=False,
    Use_Scraper_API=True,
    Scraper_api_key="",
    Groq_API=""
)
target = 'https://www.amazon.com/dp/B0B9MYQ1W1/ref=sspa_dk_detail_4?psc=1&pd_rd_i=B0B9MYQ1W1&pd_rd_w=uR0Qm&content-id=amzn1.sym.7446a9d1-25fe-4460-b135-a60336bad2c9&pf_rd_p=7446a9d1-25fe-4460-b135-a60336bad2c9&pf_rd_r=FT61ZXKDX7WCHVB8BQ26&pd_rd_wg=dRCyO&pd_rd_r=d8984c14-110a-4ff7-ba32-2ba58196ef03&s=apparel&sp_csd=d2lkZ2V0TmFtZT1zcF9kZXRhaWw'
result = Sent.get_sentiment_from_website_overall_summary(
    target_website=target,
    Groq_LLM_Max_Tokens=500,
    Groq_LLM_Max_Input_Tokens=1000
)
print(result)'''

# UnitTest for get_sentiment_from_website_overall_summary using Local LLM Ollama
'''from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_General_LLM=True,
    Use_Local_Scraper=False,
    Use_Scraper_API=True,
    Scraper_api_key=''
)
target = 'https://www.amazon.com/Legendary-Whitetails-Journeyman-Jacket-Tarmac/dp/B013KW38RQ/ref=cm_cr_arp_d_product_top?ie=UTF8'
result = Sent.get_sentiment_from_website_overall_summary(
    target_website=target
)
print(result)'''

# UnitTest for get_sentiment_from_website_overall_summary using Gemini
'''from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_General_LLM=True,
    Use_Local_Scraper=True,
)
target = 'https://www.amazon.com/Legendary-Whitetails-Journeyman-Jacket-Tarmac/dp/B013KW38RQ/ref=cm_cr_arp_d_product_top?ie=UTF8'
result = Sent.get_sentiment_from_website_overall_summary(
    target_website=target,
    Use_Gemini_API=True,
    Google_API="AIzaSyCBqcqJGCeDc1Tk63g9MsfQQXNdITdihmI"
)
print(result)'''

# UnitTest for get_analytical_customer_sentiments -->Done
'''from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_Scraper=True,
    Use_Local_Sentiment_LLM=True
)
target = 'https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1'
result = Sent.get_analytical_customer_sentiments(
    target_website=target,
    Use_Bar_chart_visualize=True,
    Use_box_plot_visualize=True,
    Use_histogram_visualize=True,
    Use_pie_chart_visualize=True,
    Use_violin_plot_visualize=True
)'''

# UnitTest for get_Sentiment_Audio_file --->Done
'''from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_Scraper=True,
    Use_Local_Sentiment_LLM=True
)
audio_path = r'D:\Sentimatrix\tests\voice_datasets-wav\review_1.wav'
result = Sent.get_Sentiment_Audio_file(audio_path)

print(result)'''

# UnitTest for compare_product_on_reviews --->Done Value Fetch was good , but llm didnt perform well
'''from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_Scraper=False,
    Use_Scraper_API=True,
    Scraper_api_key='',
    Use_Groq_API=True,
    Use_Local_Sentiment_LLM=True,
    Groq_API=''
)


targetsite1 = 'https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1'
targetsite2 = 'https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1'
result = Sent.compare_product_on_reviews(
    target_website1=targetsite1,
    target_website2=targetsite2
)

print(result)
'''

# UnitTest for compare_product_on_reviews --->Using Gemini LLM
'''from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_Scraper=True,
    Use_Scraper_API=False,
    Use_Gemini_API=True,
    Use_Local_Sentiment_LLM=True,
    Google_API='AIzaSyCBqcqJGCeDc1Tk63g9MsfQQXNdITdihmI'
)


targetsite1 = 'https://www.amazon.com/Legendary-Whitetails-Journeyman-Jacket-Tarmac/dp/B013KW38RQ/ref=cm_cr_arp_d_product_top?ie=UTF8'
targetsite2 = 'https://www.amazon.com/dp/B075ZD9L6X/ref=sspa_dk_detail_2?psc=1&pd_rd_i=B075ZD9L6X&pd_rd_w=1hd1P&content-id=amzn1.sym.10c716f0-b18a-473a-9d6a-5d87bc47ef1e&pf_rd_p=10c716f0-b18a-473a-9d6a-5d87bc47ef1e&pf_rd_r=QYMFNXHMV8Q39KPKSKMJ&pd_rd_wg=YBNW5&pd_rd_r=0aba1fc4-298b-4065-a1b3-05b3580b6673&s=apparel&sp_csd=d2lkZ2V0TmFtZT1zcF9kZXRhaWxfdGhlbWF0aWM'
result = Sent.compare_product_on_reviews(
    target_website1=targetsite1,
    target_website2=targetsite2
)

print(result)'''


# UnitTest for get_Sentiment_Image_file --->Done , But Need to be Executed and Tested
'''from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_Scraper=True,
    Use_Local_Sentiment_LLM=True
)
image_path = r"D:\Hackathon\KCG\stock images\Screenshot 2024-08-25 074810.png"
result = Sent.get_Sentiment_Image_file(
    Image_File_path=image_path
)

print(result)'''

# unitTest for Multi_language_Sentiment --->Done , can be integrated to various tasks
'''from Sentimatrix.sentiment_generation import SentConfig
SENT = SentConfig(
    Use_Local_Sentiment_LLM=True
)
message = 'நான் இந்த தயாரிப்பை வெறுக்கிறேன்'
result = SENT.Multi_language_Sentiment(message)

print(result)'''

# unitTest for Config_Local_Scraper --->Done
'''from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig()
result = Sent.Config_Local_Scraper(action='get')
print(result)'''
'''from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig()
Sent.Config_Local_Scraper(action='add',tag='div',attrs={'class': 'a-section celwidget review'})'''

# unitTest for Save_reviews_to_CSV ---> Done , Need to be Fixed
'''from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_Scraper=True,
    Use_Local_Sentiment_LLM=True
)
target = 'https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1'
Sent.Save_reviews_to_CSV(
    target_site=target,
    output_dir=r'',
    file_name='review.csv'
)'''

# Initial Website Emotion Test without groq --> Worked With using Scraper API also
'''from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_Emotion_LLM = True,
    device_map = "auto" 
)
target = 'https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1'
result = Sent.get_emotion_from_website_each_feedback(
    target_website=target,
    Use_Scraper_API=True,
    Scraper_api_key=""
)

print(result)'''


# Emotion Summarization using Groq

'''from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_Emotion_LLM=True,
    Use_Scraper_API=True,
    Scraper_api_key="",
    Use_Groq_API=True,
    Groq_API=""
)
target = 'https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1'
result = Sent.get_Emotion_from_website_overall_summary(
    target_website=target
)

print(result)'''


# Emotion Summarization using Ollama

'''from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_Emotion_LLM=True,
    Use_Scraper_API=True,
    Scraper_api_key="",
    Use_Local_General_LLM=True
)
target = 'https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1'
result = Sent.get_Emotion_from_website_overall_summary(
    target_website=target
)

print(result)'''

# Emotion Summarization using Gemini

'''from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_Emotion_LLM=True,
    Use_Scraper_API=True,
    Scraper_api_key="",
    Use_Local_General_LLM=True,
    Use_Gemini_API=True,
    Google_API=""
)
target = 'https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1'
result = Sent.get_Emotion_from_website_overall_summary(
    target_website=target
)

print(result)'''
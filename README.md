Here’s a detailed documentation for your `Sentimatrix` project, including the functionalities and sample code for accessing them.

---

# Sentimatrix Documentation

## Overview

`Sentimatrix` is a sentiment analysis and web scraping toolkit designed to analyze and visualize sentiments from various sources, including text, audio, and images. It offers integration with local and remote sentiment analysis models and web scrapers.

**Please Note:** This is a beta version of the project, and it is in the initial stages of development. Updates will be patched frequently. Ensure you review the latest documentation and updates regularly.

## Requirements

Before using this product, make sure to:

1. **Get API Keys:**

   - **Groq API**
   - **Hugging Face API**
   - **Scraper API**
   - **Browser API** from [What Is My Browser](https://www.whatismybrowser.com/detect/what-is-my-user-agent/)

2. **API Notes:**

   - **OpenAI API**: Not advised for use as it has not been tested yet.
   - **Local LLM**: Performance depends on your system configuration. Some APIs may have limited free usage and could incur costs in the future.

3. **Audio Files:**
   - Ensure audio files are converted to `.wav` format before processing.

## Features

1. **Quick Sentiment Analysis**

   - Analyze the sentiment of text messages quickly using predefined models.

2. **Feedback Sentiment from Websites**

   - Extract and analyze sentiments from customer feedback on e-commerce websites.

3. **Overall Summary Sentiment Analysis**

   - Generate an overall sentiment summary for a product based on its reviews.

4. **Analytical Visualization**

   - Visualize sentiment data using various chart types, including bar charts, box plots, histograms, pie charts, and violin plots.

5. **Sentiment Analysis from Audio Files**

   - Analyze the sentiment of spoken words from audio files.

6. **Product Comparison**

   - Compare sentiments between different products based on their reviews.

7. **Sentiment Analysis from Image Files**

   - Extract and analyze sentiment from images containing text.

8. **Multi-Language Sentiment Analysis**

   - Analyze sentiments in different languages using translation models.

9. **Local Scraper Configuration**

   - Configure and use a local scraper for extracting reviews from websites.

10. **Save Reviews to CSV**
    - Save extracted reviews from websites to a CSV file for further analysis.

## Installation

You can install `Sentimatrix` using pip:

```bash
pip install sentimatrix
```

## Usage

### Importing the Library

```python
from Sentimatrix.sentiment_generation import SentConfig
```

### Creating an Instance

```python
sent = SentConfig(
    Use_Local_Sentiment_LLM=True,
    Use_Local_Scraper=True,
    device_map="auto"
)
```

Here's a detailed documentation for your `Sentimatrix` project, including descriptions of the functionalities and sample code for usage. This documentation also addresses parameters like `Use_Scraper_API` and `Scraper_api_key`.

---

# Sentimatrix Documentation

## Overview

`Sentimatrix` is a sentiment analysis and web scraping toolkit designed to analyze and visualize sentiments from various sources, including text, audio, and images. It offers integration with local and remote sentiment analysis models and web scrapers.

## Functionalities

### 1. **Quick Sentiment Analysis**

**Description:** Analyze the sentiment of short text messages quickly using local sentiment models.

**Usage:**

```python
from Sentimatrix.sentiment_generation import SentConfig

Sent = SentConfig(
    Use_Local_Sentiment_LLM=True,
)

sentiments = ["I am very happy", "I am very sad", "I am alright"]
sentiment_result = Sent.get_Quick_sentiment(text_message=sentiments, device_map="auto")

print(sentiment_result)
```

### 2. **Web Scraper**

**Description:** Scrape reviews from e-commerce websites and analyze their sentiments.

**Usage:**

**Initialization and Scraping:**

```python
from Sentimatrix.utils.web_scraper import ReviewScraper

scraper = ReviewScraper(Use_Local_Scraper=True)
url = "https://www.amazon.com/Razer-Huntsman-Esports-Gaming-Keyboard/dp/B0CG7FQML2"
reviews_local = scraper.fetch_reviews(url)

list_of_sentences = [' '.join(sublist) for sublist in reviews_local]
for sentence in list_of_sentences:
    print(sentence)
```

**Adding and Checking Review Patterns:**

```python
scraper.add_review_pattern('div', {'class': 'new-review-class'})
current_patterns = scraper.get_review_patterns()
print("Current review patterns:", current_patterns)
```

### 3. **Sentiment Analysis from Websites**

**Description:** Analyze sentiments from reviews on a given website. Supports both local and API-based scraping.

**Usage:**

**Without Scraper API:**

```python
from Sentimatrix.sentiment_generation import SentConfig

Sent = SentConfig(
    Use_Local_Sentiment_LLM=True,
    device_map="auto"
)
target = 'https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1'
result = Sent.get_sentiment_from_website_each_feedback_sentiment(
    target_website=target,
    Use_Local_Scraper=True,
    get_Groq_Review=False
)

print(result)
```

**With Scraper API:**

```python
from Sentimatrix.sentiment_generation import SentConfig

Sent = SentConfig(
    Use_Local_Sentiment_LLM=True,
    device_map="auto",
    Use_Scraper_API=True,
    Scraper_api_key=""
)
target = 'https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1'
result = Sent.get_sentiment_from_website_each_feedback_sentiment(
    target_website=target,
    get_Groq_Review=False
)

print(result)
```

### 4. **Multi-Site Scraper**

**Description:** Scrape and analyze sentiments from multiple sites simultaneously.

**Usage:**

```python
from Sentimatrix.sentiment_generation import SentConfig

Sent = SentConfig(
    Use_Local_Sentiment_LLM=True,
    device_map="auto"
)
targets = [
    'https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1',
    'https://www.amazon.com/Legendary-Whitetails-Journeyman-Jacket-Tarmac/dp/B013KW38RQ/ref=cm_cr_arp_d_product_top?ie=UTF8'
]
result = Sent.get_sentiment_from_website_each_feedback_sentiment(
    target_website=targets,
    Use_Local_Scraper=True,
    get_Groq_Review=False
)

print(result)
```

### 5. **Sentiment Analysis from Audio Files**

**Description:** Analyze sentiment from audio files.

**Usage:**

```python
from Sentimatrix.sentiment_generation import SentConfig

Sent = SentConfig(
    Use_Local_Scraper=True,
    Use_Local_Sentiment_LLM=True
)
audio_path = r'D:\Sentimatrix\tests\voice_datasets-wav\review_1.wav'
result = Sent.get_Sentiment_Audio_file(audio_path)

print(result)
```

### 6. **Comparing Products Based on Reviews**

**Description:** Compare sentiments of reviews for two different products.

**Usage:**

```python
from Sentimatrix.sentiment_generation import SentConfig

Sent = SentConfig(
    Use_Local_Scraper=True,
    Use_Groq_API=True,
    Use_Local_Sentiment_LLM=True,
    Groq_API=''
)
targetsite1 = 'https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1'
targetsite2 = 'https://www.amazon.in/dp/B0CV9S7ZV6/ref=sspa_dk_detail_0?pd_rd_i=B0CV9S7ZV6'
result = Sent.compare_product_on_reviews(
    target_website1=targetsite1,
    target_website2=targetsite2
)

print(result)
```

### 7. **Sentiment Analysis from Images**

**Description:** Analyze sentiment from images containing text.

**Usage:**

```python
from Sentimatrix.sentiment_generation import SentConfig

Sent = SentConfig(
    Use_Local_Scraper=True,
    Use_Local_Sentiment_LLM=True
)
image_path = ''
result = Sent.get_Sentiment_Image_file(Image_File_path=image_path, Image_to_Text_Model='microsoft/Florence-2-large')

print(result)
```

### 8. **Multi-Language Sentiment Analysis**

**Description:** Perform sentiment analysis on text in multiple languages.

**Usage:**

```python
from Sentimatrix.sentiment_generation import SentConfig

SENT = SentConfig(
    Use_Local_Sentiment_LLM=True
)
message = 'நான் இந்த தயாரிப்பை வெறுக்கிறேன்'
result = SENT.Multi_language_Sentiment(message)

print(result)
```

### 9. **Configuration and Review Management**

**Description:** Manage local scraper configurations and save reviews to CSV.

**Usage:**

**Configuring Local Scraper:**

```python
from Sentimatrix.sentiment_generation import SentConfig

Sent = SentConfig()
result = Sent.Config_Local_Scraper(action='get')
print(result)
```

**Saving Reviews to CSV:**

```python
from Sentimatrix.sentiment_generation import SentConfig

Sent = SentConfig(
    Use_Local_Scraper=True,
    Use_Local_Sentiment_LLM=True
)
target = 'https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1'
Sent.Save_reviews_to_CSV(
    target_site=target,
    output_dir=r'',
    file_name='review.csv'
)
```

## Additional Notes

- **Function `get_sentiment_from_website_overall_summary`**: This function is still under development. It will be updated in future releases.
- **Function `compare_product_on_reviews`**: Features for this function will be updated soon, including additional mathematical comparisons.

## Parameters

- `Use_Local_Sentiment_LLM` (bool): Whether to use a local sentiment analysis model.
- `Use_Scraper_API` (bool): Whether to use an external scraper API.
- `Scraper_api_key` (str): API key for accessing the external scraper.
- `Use_Local_Scraper` (bool): Whether to use a local web scraper.
- `Use_Groq_API` (bool): Whether to use the Groq API for sentiment analysis.
- `Groq_API` (str): API key for accessing the Groq API.
- `Use_Local_General_LLM` (bool): Whether to use a general local LLM for analysis.
- `device_map` (str): Device configuration for model inference (e.g., "auto").

## Conclusion

This documentation provides an overview of `Sentimatrix` functionalities and usage. For more detailed configurations and advanced features, refer to the specific function implementations or the project's source code.

---

Feel free to modify any details or add additional sections based on specific project needs.

## Testing

To ensure the correctness of your implementation, you can run the unit tests included in the `tests/test_sent_config.py` file. Use the following command to run the tests:

```bash
pytest
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

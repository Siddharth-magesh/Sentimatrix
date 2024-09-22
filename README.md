Here’s a detailed documentation for your `Sentimatrix` project, including the functionalities and sample code for accessing them.

---

# Sentimatrix Documentation

## Overview

`Sentimatrix` is a sentiment analysis and web scraping toolkit designed to analyze and visualize sentiments from various sources, including text, audio, and images. It offers integration with local and remote sentiment analysis models and web scrapers.

**Please Note:** This is a beta version of the project, and it is in the initial stages of development. Updates will be patched frequently. Ensure you review the latest documentation and updates regularly.

## Table of Contents

1. [Requirements](#Requirements)
2. [Installation](#Installation)
3. [Usage](#usage)
   - [Importing the Library](#importing-the-library)
   - [Creating an Instance](#creating-an-instance)
4. [Functionalities](#functionalities)
   - [1. Quick Sentiment Analysis From Text](#1-quick-sentiment-analysis-from-text)
   - [2. Quick Sentiment Analysis From Audio](#2-quick-sentiment-analysis-from-audio)
   - [3. Quick Sentiment Analysis From Image](#3-quick-sentiment-analysis-from-image)
   - [4. Multimodal Sentiment Analysis](#4-multimodal-sentiment-analysis)
   - [5. Emotion Detection](#5-emotion-detection)
   - [6. Sentiment Analysis from Web Scraping](#6-sentiment-analysis-from-web-scraping)
   - [7. Customizable Sentiment Models](#7-customizable-sentiment-models)
   - [8. Storing Results in Database](#8-storing-results-in-database)
   - [9. Exporting Sentiment Reports](#9-exporting-sentiment-reports)
   - [10. Visualizing Sentiments](#10-visualizing-sentiments)
5. [Additional Notes](#additional-notes)
6. [Parameters](#parameters)
7. [Conclusion](#conclusion)
8. [License](#license)

## Requirements

Before using this product, make sure to:

1. **Get API Keys:**

   - **Groq API**
   - **Hugging Face API**
   - **Scraper API**
   - **Gemini API**
   - **Steam API**
   - **OpenAI API**
   - **IMDB API**
   - **Google Youtube API**
   - **Reddit API**
   - **Browser API** from [What Is My Browser](https://www.whatismybrowser.com/detect/what-is-my-user-agent/)
   - **Install Ollama for Local Inference and Pull llama3.1 and llava**

2. **API Notes:**

   - **OpenAI API**: Not advised for use as it has not been tested yet.
   - **Local LLM**: Performance depends on your system configuration. Make Sure Your System Supports Ollama.
   - **API Warnings**: Some APIs may have limited free usage and could incur costs in the future.

3. **Audio Files:**
   - Ensure audio files are converted to `.wav` format before processing.

## Features

1. **Quick Sentiment Analysis from Text**

   - Analyze the sentiment of short text messages rapidly using local sentiment models.

2. **Quick Sentiment Analysis from Audio**

   - Analyze the sentiment of spoken words from audio files (in `.wav` format) for emotional insights.

3. **Quick Sentiment Analysis from Image**

   - Extract and analyze sentiment from images containing text using local image captioning models.

4. **Feedback Sentiment from Websites**

   - Extract and analyze sentiments from customer feedback on e-commerce websites.

5. **Overall Summary Sentiment Analysis**

   - Generate a comprehensive sentiment summary for a product based on its reviews.

6. **Analytical Visualization**

   - Visualize sentiment data using various chart types, including bar charts, box plots, histograms, pie charts, and violin plots.

7. **Product Comparison**

   - Compare sentiments between different products based on their reviews to help users make informed decisions.

8. **Multi-Language Sentiment Analysis**

   - Perform sentiment analysis on text in multiple languages using translation models.

9. **Local Scraper Configuration**

   - Configure and use a local scraper for extracting reviews from specified websites, with customizable settings.

10. **Save Reviews to CSV**

    - Save extracted reviews from websites to a CSV file for further analysis.

11. **Sentiment and Emotion Analysis from Websites**

    - Analyze sentiments and emotions from reviews on specified websites, supporting both local and API-based scraping.

12. **Summarizing Products Based on Reviews and Sentiments**

    - Summarize reviews for products using various local or API-based LLMs to provide an overview of sentiments.

13. **Summarizing Products Based on Reviews and Emotions**

    - Summarize reviews and associated emotions for products using local or API-based LLMs.

14. **Suggestions Generation**

    - Generate product suggestions based on sentiments and reviews.

15. **Visual Comparison**

    - Perform graphical visualizations for comparing sentiments of two different products.

16. **Configuration and Review Management**

    - Manage configurations for local scrapers and review patterns for targeted scraping.

17. **Web Scraper for Multiple Sites**

    - Scrape and analyze sentiments from multiple sites simultaneously for broader insights.

18. **Sentiment Analysis from Multiple Sources**
    - Analyze sentiments from various platforms like YouTube, IMDB, LetterBoxD, MetaCritic, Reddit, and RottenTomatoes.

### System Requirements

- **Operating System**: Windows/Linux/macOS
- **Memory**: Minimum 16GB RAM recommended for local inference
- **Storage**: Sufficient space to install Ollama and models
- **Dependencies**:
  - Python 3.8 or higher
  - Ollama installed and configured
  - Models: `llama3.1`, `llava`

## Installation

You can install `Sentimatrix` using pip:

```bash
pip install Sentimatrix
```

Make Sure to install the Following Torch Version to have Inference on GPU

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
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

## Functionalities

### 1. **Quick Sentiment Analysis From Text**

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

### 2. **Quick Sentiment Analysis From Audio**

**Description:** Analyze the sentiment of Audio Files quickly using local sentiment models. Inputs are .wav files.

**Usage:**

```python
from Sentimatrix.sentiment_generation import SentConfig

Sent = SentConfig(
    Use_Local_Scraper=True,
    Use_Local_Sentiment_LLM=True
)
audio_path = r'<Audio Path>.wav'
result = Sent.get_Sentiment_Audio_file(audio_path)

print(result)
```

### 3. **Quick Sentiment Analysis From Image**

**Description:** Analyze the sentiment of Image Files quickly using local llava model for image captioning and Sentiment models.

**Usage:**

```python
from Sentimatrix.sentiment_generation import SentConfig

Sent = SentConfig(
    Use_Local_General_LLM=True,
    Use_Local_Sentiment_LLM=True
)
imagepath = r"<Image Path>.png"
result = Sent.get_Sentiment_Image_file(
    Image_File_path=imagepath
)
print(result)
```

### 4. **Web Scraper**

**Configurations and General Fetch**

**Scraping E-commerce:**

**Description:** Scrape reviews from e-commerce websites , Mostly Amazon. Might Work on Other Sites.

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

**Description:** Adding Classes to be Scrapped.

```python
scraper.add_review_pattern('div', {'class': 'new-review-class'})
current_patterns = scraper.get_review_patterns()
print("Current review patterns:", current_patterns)
```

**Configuring Local Scraper:**

**Description:** Manage local Amazon scraper configurations.

```python
from Sentimatrix.sentiment_generation import SentConfig

Sent = SentConfig()
result = Sent.Config_Local_Scraper(action='get')
print(result)
```

### 5. **Sentiment and Emotion Analysis from Websites**

**Sentiment From Amazon Site**

**Description:** Analyze sentiments from reviews on a given website. Supports both local and API-based scraping.

```python
from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_Emotion_LLM = True,
    device_map = "auto"
)
target = 'target site'
result = Sent.get_emotion_from_website_each_feedback(
    target_website=target,
    Use_Scraper_API=True,
    Scraper_api_key=""
)

print(result)
```

**Emotion From Amazon Site**

**Description:** Analyze Emotions from reviews on a given website. Supports both local and API-based scraping.

```python
from Sentimatrix.sentiment_generation import SentConfig

Sent = SentConfig(
    Use_Local_Sentiment_LLM=True,
    device_map="auto"
)
target = 'target-site'
result = Sent.get_sentiment_from_website_each_feedback_sentiment(
    target_website=target,
    Use_Local_Scraper=False,
    Use_Scraper_API=True,
    Scraper_api_key="Your Scraper API Key",
    get_Groq_Review=False # Set True to Get Groq Reviews
)

print(result)
```

**From Amazon Multiple Site**

**Description:** Scrape and analyze sentiments from multiple sites simultaneously.

```python
from Sentimatrix.sentiment_generation import SentConfig

Sent = SentConfig(
    Use_Local_Sentiment_LLM=True,
    device_map="auto"
)
targets = [
    '',
    ''
]
result = Sent.get_sentiment_from_website_each_feedback_sentiment(
    target_website=targets,
    Use_Local_Scraper=True,
    get_Groq_Review=False
)

print(result)
```

**From Youtube**

**Description:** Scrape and analyze sentiments from Youtube.

```python
from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_General_LLM=True,
    Use_Local_Sentiment_LLM=True
)
target = 'Product Name'
youtube_api_key = 'Google API KEY'
result = Sent.get_analysis_report_from_youtube(
    Product_Name=target,
    Youtube_API=youtube_api_key,
    Use_Local_API=True,
    Get_Suggestions=True
)

print(result)
```

**From IMDB**

**Description:** Scrape and analyze sentiments from IMDB.

```python
from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_General_LLM=True,
    Use_Local_Sentiment_LLM=True
)
target = "<Movie Name"
Reviews_Count = 50
IMDB_API = ""
result = Sent.get_analysis_report_from_imdb(
    Product_Name=target,
    Reviews_Count=Reviews_Count,
    IMDB_API=IMDB_API,
    Use_Gemini_API=True,
    Google_API="",
    Get_Suggestions=False
)

print(result)
```

**From LetterBoxD**

**Description:** Scrape and analyze sentiments from LetterBoxD.

```python
from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_General_LLM=True,
    Use_Local_Sentiment_LLM=True
)
target = "Movie Name"
Reviews_Count = 20
result = Sent.get_analysis_report_from_LetterBoxD(
    Product_Name=target,
    Reviews_Count=Reviews_Count,
    Use_Groq_API=True,
    Groq_API="",
    Get_Suggestions=True
)

print(result)
```

**From MetaCritic**

**Description:** Scrape and analyze sentiments from MetaCritic.

```python
from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_General_LLM=True,
    Use_Local_Sentiment_LLM=True
)
target = "Game Name"
Reviews_Count = 10
result = Sent.get_analysis_report_from_metacritic(
    Product_Name=target,
    Reviews_Count=Reviews_Count,
    Use_Local_API=True,
    Get_Suggestions=True
)

print(result)
```

**From Reddit**

**Description:** Scrape and analyze sentiments from Reddit.

```python
from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_General_LLM=True,
    Use_Local_Sentiment_LLM=True
)
target = ""
Reviews_Count = 10
result = Sent.get_analysis_report_from_reddit(
    Product_Name=target,
    Reviews_Count=Reviews_Count,
    Use_Local_API=True,
    Get_Suggestions=True
)

print(result)
```

**From RottenTomatoes**

**Description:** Scrape and analyze sentiments from RottenTomatoes.

```python
from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_General_LLM=True,
    Use_Local_Sentiment_LLM=True
)
target = "Movie Name"
Reviews_Count = 100
result = Sent.get_analysis_report_from_rottentomatoes(
    Product_Name=target,
    Reviews_Count=Reviews_Count,
    Use_Local_API=True,
    Get_Suggestions=True
)

print(result)
```

**From Steam**

**Description:** Scrape and analyze sentiments from Steam.

```python
from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_General_LLM=True,
    Use_Local_Sentiment_LLM=True
)
target = "Game Name"
Reviews_Count = 50
result = Sent.get_analysis_report_from_steam(
    Product_Name=target,
    Reviews_Count=Reviews_Count,
    Use_Local_API=True,
    Get_Suggestions=True
)

print(result)
```

### 6. **Summarzing Products Based on Reviews and Sentiments**

**Description:** Summarize the Reviews of Product Reviews. Can Use Local LLM (Ollama : llama3.1) , Groq LLM's , Gemini LLM's , OpenAI LLM's

**Usage:**

```python
from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Groq_API=True,
    Use_Local_Scraper=False,
    Use_Scraper_API=True,
    Scraper_api_key="",
    Groq_API=""
)
target = 'target Site'
result = Sent.get_sentiment_from_website_overall_summary(
    target_website=target,
    Groq_LLM_Max_Tokens=500,
    Groq_LLM_Max_Input_Tokens=850,
    Groq_LLM="llama-3.1-70b-versatile"
)
print(result)
```

### 7. **Summarzing Products Based on Reviews and Emotions**

**Description:** Summarize the Reviews of Product Reviews and Emotions. Can Use Local LLM (Ollama : llama3.1) , Groq LLM's , Gemini LLM's , OpenAI LLM's

**Usage:**

```python
from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_Emotion_LLM=True,
    Use_Scraper_API=True,
    Scraper_api_key="",
    Use_Groq_API=True,
    Groq_API=""
)
target = 'target site'
result = Sent.get_Emotion_from_website_overall_summary(
    target_website=target,
    Groq_LLM_Max_Tokens=800,
    Groq_LLM_Max_Input_Tokens=500
)

print(result)
```

### 8. **Comparing Products Based on Reviews**

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
targetsite1 = ''
targetsite2 = ''
result = Sent.compare_product_on_reviews(
    target_website1=targetsite1,
    target_website2=targetsite2
)

print(result)
```

### 9. **Multi-Language Sentiment Analysis**

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

### 9. **Visualizations**

**Description:** Performs Graphical Visualizations for Sentiments for the target Product.

**Usage:**

```python
from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_Scraper=True,
    Use_Local_Sentiment_LLM=True
)
target = ''
result = Sent.get_analytical_customer_sentiments(
    target_website=target,
    Use_Bar_chart_visualize=True,
    Use_box_plot_visualize=True,
    Use_histogram_visualize=True,
    Use_pie_chart_visualize=True,
    Use_violin_plot_visualize=True,
    Use_Card_Emotion_Visulize=True
)
```

### 10. **Visual Comparison**

**Description:** Performs Graphical Visualizations for Comparing Two Prodcuts.

**Usage:**

```python
from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_Scraper=False,
    Use_Scraper_API=True,
    Scraper_api_key="",
    Use_Local_Sentiment_LLM=False
)
target1 = ''
target2 = ''
result = Sent.compare_product_on_reviews(
    target_website1=target1,
    target_website2=target2,
    Get_Graphical_View=True
)

print(result)
```

### 11. **Suggestions Generations**

**Description:** Generates Suggestions for the given product.

**Usage:**

```python
from Sentimatrix.sentiment_generation import SentConfig
Sent = SentConfig(
    Use_Local_General_LLM=True,
    Use_Local_Sentiment_LLM=True
)
target = "product Name"
Reviews_Count = 20
result = Sent.get_analysis_report_from_LetterBoxD(
    Product_Name=target,
    Reviews_Count=Reviews_Count,
    Use_Groq_API=True,
    Groq_API="",
    Get_Suggestions=False # ACTIVATING THIS PARAM WILL GENEARTE YOU A SUGGESTION USING LOCAL LLM
)

print(result)
```

### 12. **Configuration and Review Management**

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

- **Function `Use_OpenAI_API`**: This Function is Still Under Development.
- **Web Scraping**: The Local Scraper Might not work occasionally and Some sites might Take Multiple tries and long time Duration. The Next update is to let the LLM choose what site to Scrape and Do scraping By the LLM.

## Parameters

- `Use_Local_Sentiment_LLM` (bool): Whether to use a local sentiment analysis model.
- `Use_Scraper_API` (bool): Whether to use an external scraper API.
- `Scraper_api_key` (str): API key for accessing the external scraper.
- `Use_Local_Scraper` (bool): Whether to use a local web scraper.
- `Use_Groq_API` (bool): Whether to use the Groq API for sentiment analysis.
- `Groq_API` (str): API key for accessing the Groq API.
- `Use_Local_General_LLM` (bool): Whether to use a general local LLM for analysis.
- `device_map` (str): Device configuration for model inference (e.g., "auto").
- `Audio_File_Path` (str): Path to the audio file for sentiment analysis.
- `Image_File_Path` (str): Path to the image file for sentiment analysis.
- `Target_Website` (str): URL of the website to scrape for reviews.
- `Output_Directory` (str): Directory path for saving output files, such as CSVs.
- `Review_Count` (int): Number of reviews to scrape or analyze from a given source.
- `Use_Visualization` (bool): Whether to generate visual representations of sentiment data.
- `Get_Suggestions` (bool): Whether to generate suggestions based on analyzed sentiment.
- `Multi_Language` (bool): Whether to enable sentiment analysis in multiple languages.
- `Local_Scraper_Config` (dict): Configuration settings for the local scraper, including parameters like user-agent and timeouts.

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.

## Conclusion

`Sentimatrix` offers a robust toolkit for sentiment analysis and web scraping, empowering users to gain valuable insights from various data sources. As we continue to develop and refine the project, your feedback and contributions are invaluable. Stay tuned for more features and updates!

---

## Support

If you encounter any issues or have questions, please open an issue on the [GitHub repository](https://github.com/Siddharth-magesh/Sentimatrix) or contact the maintainer at [siddharthmagesh007@gmail.com](mailto:siddharthmagesh007@gmail.com).

## Creators

| Role               | Name               | Contact                        | GitHub                                |
| ------------------ | ------------------ | ------------------------------ | ------------------------------------- |
| **Lead Developer** | Siddharth Magesh | [siddharthmagesh007@gmail.com](siddharthmagesh007@gmail.com) | [Siddharth-magesh]([Siddharth-Magesh](https://github.com/Siddharth-magesh)) |
| **Contributor**    | Pranesh Kumar V  | [praneshvaradharaj@gmail.com](praneshvaradharaj@gmail.com)  | [PraneshPK2005]([Pranesh-pk](https://github.com/PraneshPK2005))    |

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

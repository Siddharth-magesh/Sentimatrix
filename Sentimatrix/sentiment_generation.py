from .utils.Quick_Sentiment import Generation_Pipeline
from .utils.web_scraper import Fetch_Reviews, add_review_pattern, get_review_patterns
from .utils.Structured_Sentiment import Struct_Generation_Pipeline, Struct_Generation_Pipeline_Visual
from .utils.Structured_Emotion import Struct_Emotion
from .utils.llm_inference.groq_inference import (
    Groq_inference_list,
    summarize_reviews,
    compare_reviews_local,
    summarize_Emotions_reviews,
    suggest_reviews
)
from .utils.llm_inference.gemini_inference import (
    Gemini_inference_list,
    summarize_reviews_gemini,
    compare_reviews_gemini,
    summarize_Emotion_reviews_gemini,
    suggest_reviews_gemini
)
from .utils.llm_inference.openai_inference import (
    OpenAI_inference_list,
    summarize_reviews_openai,
)
from .utils.llm_inference.localLLM_inference import (
    LocalLLM_inference_list,
    summarize_reviews_local,
    Ollama_Local_Summarize,
    Ollama_Local_Emotion_Summarize,
    Ollama_Local_Sentiment_Comparsion,
    Ollama_Local_Suggestions
)
from .utils.visualization import (
    plot_sentiment_box_plot,
    plot_sentiment_distribution,
    plot_sentiment_histograms,
    plot_sentiment_pie_chart,
    plot_sentiment_violin_plot,
    calculate_top_emotions_percentages,
    plot_sentiment_comparison
)
from .utils.wav_to_text import audio_to_text
from .utils.text_translation import Translate_text
from .utils.save_to_csv import save_reviews_to_csv
from .utils.llm_inference.Image_to_text import Generate_Summary_From_Image
from .utils.web_Scraper_Apps.imdb import IMDBReviewScraper
from .utils.web_Scraper_Apps.letterboxd import LetterboxdReviewScraper
from .utils.web_Scraper_Apps.metacritic import MetacriticReviewScraper
from .utils.web_Scraper_Apps.reddit import RedditScraper
from .utils.web_Scraper_Apps.rottentomatoes import RottenTomatoesReviewScraper
from .utils.web_Scraper_Apps.steam import SteamReviewScraper
from .utils.web_Scraper_Apps.youtube import YouTubeDataFetcher

class SentConfig:
    """
    Main Class for Configuring and Performing Sentiment Analysis.

    This class provides functionality for sentiment analysis using various methods and models,
    including local sentiment models, general models, APIs, and web scraping tools.

    Constructor:
    - Initializes the class with the following configuration options:
        - Use_Local_Sentiment_LLM (bool): Flag to use a local sentiment analysis model.
        - Use_Local_General_LLM (bool): Flag to use a local general-purpose language model.
        - Use_Groq_API (bool): Flag to use the Groq API.
        - Use_Open_API (bool): Flag to use the OpenAI API.
        - Use_Local_Scraper (bool): Flag to use a local web scraper.
        - Use_Scraper_API (bool): Flag to use a web scraper API.
        - Local_Sentiment_LLM (str): Identifier for the local sentiment analysis model.
        - Local_General_LLM (str): Identifier for the local general-purpose language model.
        - Local_General_LLM_kwargs (dict): Configuration parameters for the local general-purpose model.
        - Groq_API (str): API key or URL for the Groq API.
        - OpenAi_API (str): API key or URL for the OpenAI API.
        - HuggingFace_API (str): API key or URL for the Hugging Face API.
        - Local_api_key (str): API key for local services.
        - Scraper_api_key (str): API key for the web scraper.
        - Groq_LLM (str): Identifier for the Groq LLM model.
        - OpenAI_LLM (str): Identifier for the OpenAI LLM model.
        - device_map (str): Device configuration for model inference.

    Methods:
    - get_Quick_sentiment: Analyzes sentiment for a given text message quickly.
    - get_sentiment_from_website_each_feedback_sentiment: Analyzes sentiment of individual feedback from a website.
    - get_sentiment_from_website_overall_summary: Provides an overall sentiment summary from a website.
    - get_analytical_customer_sentiments: Analyzes customer sentiments from various sources for deeper insights.
    - get_Sentiment_Audio_file: Extracts sentiment from an audio file by converting it to text and analyzing it.
    - compare_product_on_reviews: Compares sentiment of reviews between two products or services.
    - get_Sentiment_Image_file: Extracts sentiment from an image by converting it to text and analyzing it.
    - Multi_language_Sentiment: Translates a text message to a default language and analyzes its sentiment.
    - Config_Local_Scraper: Configures the local web scraper by adding or retrieving review patterns.
    - Save_reviews_to_CSV: Saves the fetched reviews to a CSV file.
    """

    def __init__(
            self,
            Use_Local_Sentiment_LLM=True,
            Use_Local_Emotion_LLM = True,
            Use_Local_General_LLM=False,
            Use_Groq_API=False,
            Use_Gemini_API=False,
            Use_Open_API=False,
            Use_Local_Scraper=False,
            Use_Scraper_API=False,
            Local_Sentiment_LLM="cardiffnlp/twitter-roberta-base-sentiment-latest",
            Local_General_LLM="llama3.1",
            Local_Emotion_LLM="SamLowe/roberta-base-go_emotions",
            Local_General_LLM_kwargs={
                'temperature': 0.1,
                'top_p': 1
            },
            Groq_API="",
            OpenAi_API="",
            HuggingFace_API="",
            Google_API="",
            Local_api_key='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            Scraper_api_key=None,
            Groq_LLM="llama3-8b-8192",
            OpenAI_LLM="GPT-3.5",
            Gemini_LLM="gemini-1.5-pro",
            device_map="auto",
            Ollama_Model_EndPoint = "http://localhost:11434/api/generate" 
    ):
        """
        Initializes the SentConfig class with configuration options for sentiment analysis
        and web scraping. Sets various flags and parameters for using local LLMs, APIs, and scrapers.
        """

        self.Use_Local_Sentiment_LLM = Use_Local_Sentiment_LLM
        self.Use_Local_Emotion_LLM = Use_Local_Emotion_LLM
        self.Use_Local_General_LLM = Use_Local_General_LLM
        self.Use_Groq_API = Use_Groq_API
        self.Use_Open_API = Use_Open_API
        self.Use_Gemini_API = Use_Gemini_API
        self.Use_Local_Scraper = Use_Local_Scraper
        self.Use_Scraper_API = Use_Scraper_API
        self.Local_Sentiment_LLM = Local_Sentiment_LLM
        self.Local_General_LLM = Local_General_LLM
        self.Local_General_LLM_kwargs = Local_General_LLM_kwargs
        self.Groq_API = Groq_API
        self.Google_API = Google_API
        self.OpenAi_API = OpenAi_API
        self.Local_api_key = Local_api_key
        self.Scraper_api_key = Scraper_api_key
        self.HuggingFace_API = HuggingFace_API
        self.Groq_LLM = Groq_LLM
        self.OpenAI_LLM = OpenAI_LLM
        self.Gemini_LLM = Gemini_LLM
        self.device_map = device_map
        self.Local_Emotion_LLM = Local_Emotion_LLM
        self.Ollama_Model_EndPoint = Ollama_Model_EndPoint

    def get_Quick_sentiment(
            self,
            text_message,
            Use_Local_Sentiment_LLM=None,
            Local_Sentiment_LLM=None,
            device_map=None
    ):
        """
        Function Description : Gives a Sentiment analysis on Text Messages . Runs On local machine with HF LLM.

        input params : 
        text_message -> Str or List of Str containing Text messages
        Use_Local_Sentiment_LLM : Bool , must be Set to True
        Local_Sentiment_LLM : Sentiment LLM
        device_map : "cuda" or "cpu" or "cuda"

        return : 
        {'label': str, 'score': float} or [{'label': str, 'score': float},{'label': str, 'score': float},...]
        """
        Use_Local_Sentiment_LLM = Use_Local_Sentiment_LLM if Use_Local_Sentiment_LLM is not None else self.Use_Local_Sentiment_LLM
        Local_Sentiment_LLM = Local_Sentiment_LLM if Local_Sentiment_LLM is not None else self.Local_Sentiment_LLM
        device_map = device_map if device_map is not None else self.device_map
        final_result = Generation_Pipeline(
            text_message=text_message,
            Use_Local_Sentiment_LLM=Use_Local_Sentiment_LLM,
            model_id=Local_Sentiment_LLM,
            device_map=device_map
        )
        return final_result
    
    def get_emotion_from_website_each_feedback(
        self,
        target_website,
        Use_Local_Emotion_LLM=None,
        Use_Local_General_LLM=None,
        Use_Local_Scraper=None,
        Use_Scraper_API=None,
        Use_Groq_API=None,
        Use_Gemini_API=False,
        Use_Open_API=None,
        Local_Emotion_LLM="SamLowe/roberta-base-go_emotions",
        Local_General_LLM="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        Local_General_LLM_kwargs={
            'temperature': 0.1,
            'top_p': 1
        },
        Groq_API=None,
        OpenAi_API=None,
        Google_API=None,
        HuggingFace_API=None,
        Scraper_api_key=None,
        Local_api_key=None,
        Groq_LLM="llama3-8b-8192",
        OpenAI_LLM="GPT-3.5",
        Gemini_LLM="gemini-1.5-pro",
        device_map="auto",
        get_Groq_Review=False, 
        get_Gemini_Review=False,
        get_OpenAI_review=False,
        get_localLLM_review=False,
        Groq_LLM_Temperature=0.1,
        Groq_LLM_Max_Tokens=100,
        Groq_LLM_Max_Input_Tokens=300,
        Groq_LLM_top_p=1,
        Groq_LLM_stream=False,
        OpenAI_LLM_Temperature=0.1,
        OpenAI_LLM_Max_Tokens=100,
        OpenAI_LLM_stream=False,
        OpenAI_LLM_Max_Input_Tokens=300,
        Local_LLM_Max_Input_Tokens=300,
        Gemini_LLM_Temperature=0.1,
        Gemini_LLM_Max_Tokens=100,
        Gemini_LLM_Max_Input_Tokens=300
    ):
        """
        Fetches reviews from specified websites and performs sentiment analysis on each review.
        Supports different methods for fetching and analyzing reviews, including local LLMs and APIs.
        Returns sentiment analysis results for the fetched reviews, optionally including additional comments from other LLMs.

        Parameters:
        - target_website: A string or list of strings representing the websites to fetch reviews from.
        - Use_Local_Sentiment_LLM: Boolean indicating whether to use the local sentiment LLM.
        - Use_Local_General_LLM: Boolean indicating whether to use the local general LLM.
        - Use_Local_Scraper: Boolean indicating whether to use a local scraper.
        - Use_Scraper_API: Boolean indicating whether to use a scraper API.
        - Use_Groq_API: Boolean indicating whether to use the Groq API.
        - Use_Open_API: Boolean indicating whether to use the OpenAI API.
        - Local_Sentiment_LLM: The specific local sentiment model to use.
        - Local_General_LLM: The specific local general model to use.
        - Local_General_LLM_kwargs: Additional arguments for the local general LLM.
        - Groq_API: API key for the Groq API.
        - OpenAi_API: API key for OpenAI.
        - HuggingFace_API: API key for Hugging Face.
        - Scraper_api_key: API key for the scraper service.
        - Local_api_key: API key for the local API.
        - Groq_LLM: Model identifier for Groq LLM.
        - OpenAI_LLM: Model identifier for OpenAI LLM.
        - device_map: Specifies the device to run the model on (e.g., "cpu" or "cuda").
        - get_Groq_Review: Boolean indicating whether to get additional comments from Groq LLM.
        - get_OpenAI_review: Boolean indicating whether to get additional comments from OpenAI.
        - get_localLLM_review: Boolean indicating whether to get additional comments from the local LLM.
        - Groq_LLM_Temperature: Temperature setting for the Groq LLM.
        - Groq_LLM_Max_Tokens: Maximum tokens for the Groq LLM.
        - Groq_LLM_Max_Input_Tokens: Maximum input tokens for the Groq LLM.
        - Groq_LLM_top_p: Top-p value for the Groq LLM.
        - Groq_LLM_stream: Boolean indicating whether to use streaming for Groq LLM.
        - OpenAI_LLM_Temperature: Temperature setting for the OpenAI LLM.
        - OpenAI_LLM_Max_Tokens: Maximum tokens for the OpenAI LLM.
        - OpenAI_LLM_stream: Boolean indicating whether to use streaming for OpenAI LLM.
        - OpenAI_LLM_Max_Input_Tokens: Maximum input tokens for the OpenAI LLM.
        - Local_LLM_Max_Input_Tokens: Maximum input tokens for the local LLM.

        Returns:
        - Sentiment analysis results for the fetched reviews, optionally with additional comments from Groq, OpenAI, or local LLMs.
        """
        Use_Local_Scraper = Use_Local_Scraper if Use_Local_Scraper is not None else self.Use_Local_Scraper
        Use_Scraper_API = Use_Scraper_API if Use_Scraper_API is not None else self.Use_Scraper_API
        Scraper_api_key = Scraper_api_key if Scraper_api_key is not None else self.Scraper_api_key
        Local_api_key = Local_api_key if Local_api_key is not None else self.Local_api_key
        Local_Emotion_LLM = Local_Emotion_LLM if Local_Emotion_LLM is not None else self.Local_Emotion_LLM
        device_map = device_map if device_map is not None else self.device_map

        Groq_API = Groq_API if Groq_API is not None else self.Groq_API
        Use_Groq_API = Use_Groq_API if Use_Groq_API is not None else self.Use_Groq_API
        Groq_LLM = Groq_LLM if Groq_LLM is not None else self.Groq_LLM

        Use_Open_API = Use_Open_API if Use_Open_API is not None else self.Use_Open_API
        OpenAi_API = OpenAi_API if OpenAi_API is not None else self.OpenAi_API
        OpenAI_LLM = OpenAI_LLM if OpenAI_LLM is not None else self.OpenAI_LLM

        Google_API = Google_API if Google_API is not None else self.Google_API
        Use_Gemini_API = Use_Gemini_API if Use_Gemini_API is not None else self.Use_Gemini_API
        Gemini_LLM = Gemini_LLM if Gemini_LLM is not None else self.Gemini_LLM

        Use_Local_General_LLM = Use_Local_General_LLM if Use_Local_General_LLM is not None else self.Use_Local_General_LLM
        Local_General_LLM = Local_General_LLM if Local_General_LLM is not None else self.Local_General_LLM
        Local_General_LLM_kwargs = Local_General_LLM_kwargs if Local_General_LLM_kwargs is not None else self.Local_General_LLM_kwargs
        HuggingFace_API = HuggingFace_API if HuggingFace_API is not None else self.HuggingFace_API

        fetched_review = []
        fetched_review_array = []
        final_resulted_output = None
        final_resulted_output_of_all_sites = []

        # Getting the Reviews From Site
        if isinstance(target_website, str):
            fetched_review = Fetch_Reviews(
                target_website,
                Use_Local_Scraper,
                Use_Scraper_API,
                Scraper_api_key,
                Local_api_key
            )
        elif isinstance(target_website, list) and all(isinstance(item, str) for item in target_website):
            for site in target_website:
                fetched_review_temp = Fetch_Reviews(
                    site,
                    Use_Local_Scraper,
                    Use_Scraper_API,
                    Scraper_api_key,
                    Local_api_key
                )
                fetched_review_array.append(fetched_review_temp)
        else:
            return "Error: The accepted format is either a list of strings or a single string."
        
        # Genrating Emotion For Those Reviews
        if fetched_review:
            final_resulted_output = Struct_Emotion(
                text_message=fetched_review,
                Use_Local_Emotion_LLM=True,
                model_id=Local_Emotion_LLM,
                device_map=device_map
            )
        elif fetched_review_array:
            webcount = 1
            for each_website_reviews in fetched_review_array:
                final_resulted_output_of_each_site = Struct_Emotion(
                    text_message=each_website_reviews,
                    Use_Local_Emotion_LLM=True,
                    model_id=Local_Emotion_LLM,
                    device_map=device_map
                )
                final_resulted_output_of_each_site.insert(0, webcount)
                webcount = webcount + 1
                final_resulted_output_of_all_sites.append(
                    final_resulted_output_of_each_site)
        else:
            return "Error: Didnt Find Any Inputs"

        # Adding Additional Comments To those Reviews
        if get_Groq_Review == True or get_OpenAI_review == True or get_localLLM_review == True or get_Gemini_Review == True:
            if get_localLLM_review:  # Local LLM Inference
                if final_resulted_output:  # Single Website Given by the user
                    updated_reviews = LocalLLM_inference_list(
                        final_resulted_output,
                        model_name=Local_General_LLM,
                        Local_General_LLM_kwargs=Local_General_LLM_kwargs,
                        max_input_tokens=Local_LLM_Max_Input_Tokens,
                        device_map=device_map
                    )
                    return updated_reviews
                elif final_resulted_output_of_all_sites:  # Multiple Website given by the user
                    all_sites_updated_reviews = []
                    for site_reviews in final_resulted_output_of_all_sites:
                        site_id = site_reviews[0]
                        reviews = site_reviews[1:]
                        updated_reviews = LocalLLM_inference_list(
                            final_resulted_output,
                            model_name=Local_General_LLM,
                            Local_General_LLM_kwargs=Local_General_LLM_kwargs,
                            max_input_tokens=Local_LLM_Max_Input_Tokens,
                            device_map=device_map
                        )
                        all_sites_updated_reviews.append(
                            [site_id] + updated_reviews_site)
                    return all_sites_updated_reviews
                else:
                    print("Error: Didn't receive any result from the sites")
                    exit()
            elif get_Groq_Review:  # Groq LLM Inference
                if final_resulted_output:  # Single Website Given by the user
                    updated_reviews = Groq_inference_list(
                        final_resulted_output,
                        Groq_API,
                        temperature=Groq_LLM_Temperature,
                        top_p=Groq_LLM_top_p,
                        stream=Groq_LLM_stream,
                        max_input_tokens=Groq_LLM_Max_Input_Tokens,
                        max_tokens=Groq_LLM_Max_Tokens
                    )
                    return updated_reviews
                elif final_resulted_output_of_all_sites:  # Multiple Website given by the user
                    all_sites_updated_reviews = []
                    for site_reviews in final_resulted_output_of_all_sites:
                        site_id = site_reviews[0]
                        reviews = site_reviews[1:]
                        updated_reviews_site = Groq_inference_list(
                            reviews,
                            Groq_API,
                            temperature=Groq_LLM_Temperature,
                            top_p=Groq_LLM_top_p,
                            stream=Groq_LLM_stream,
                            max_input_tokens=Groq_LLM_Max_Input_Tokens,
                            max_tokens=Groq_LLM_Max_Tokens
                        )
                        all_sites_updated_reviews.append(
                            [site_id] + updated_reviews_site)
                    return all_sites_updated_reviews
                else:
                    print("Error : Didnt Recieve any result from the sites")
                    exit()
            elif get_Gemini_Review:  # Groq LLM Inference
                if final_resulted_output:  # Single Website Given by the user
                    updated_reviews = Gemini_inference_list(
                        reviews=final_resulted_output,
                        KEY=Google_API,
                        temperature=Gemini_LLM_Temperature,
                        max_input_tokens=Gemini_LLM_Max_Input_Tokens,
                        max_tokens=Gemini_LLM_Max_Tokens,
                        model=Gemini_LLM
                    )
                    return updated_reviews
                elif final_resulted_output_of_all_sites:  # Multiple Website given by the user
                    all_sites_updated_reviews = []
                    for site_reviews in final_resulted_output_of_all_sites:
                        site_id = site_reviews[0]
                        reviews = site_reviews[1:]
                        updated_reviews = Gemini_inference_list(
                            reviews=final_resulted_output,
                            KEY=Google_API,
                            temperature=Gemini_LLM_Temperature,
                            max_input_tokens=Gemini_LLM_Max_Input_Tokens,
                            max_tokens=Gemini_LLM_Max_Tokens,
                            model=Gemini_LLM
                        )
                        all_sites_updated_reviews.append(
                            [site_id] + updated_reviews_site)
                    return all_sites_updated_reviews
                else:
                    print("Error : Didnt Recieve any result from the sites")
                    exit()
            elif get_OpenAI_review:  # OpenAI LLM Inference
                if final_resulted_output:  # Single Website Given by the user
                    updated_reviews = OpenAI_inference_list(
                        final_resulted_output,
                        OpenAi_API,
                        temperature=OpenAI_LLM_Temperature,
                        max_tokens=OpenAI_LLM_Max_Tokens,
                        stream=OpenAI_LLM_stream,
                        max_input_tokens=OpenAI_LLM_Max_Input_Tokens
                    )
                    return updated_reviews
                elif final_resulted_output_of_all_sites:  # Multiple Website given by the user
                    all_sites_updated_reviews = []
                    for site_reviews in final_resulted_output_of_all_sites:
                        site_id = site_reviews[0]
                        reviews = site_reviews[1:]
                        updated_reviews_site = OpenAI_inference_list(
                            reviews,
                            OpenAi_API,  # Add your OpenAI API key here
                            temperature=OpenAI_LLM_Temperature,
                            max_tokens=OpenAI_LLM_Max_Tokens,
                            stream=OpenAI_LLM_stream,
                            max_input_tokens=OpenAI_LLM_Max_Input_Tokens
                        )
                        all_sites_updated_reviews.append(
                            [site_id] + updated_reviews_site)
                    return all_sites_updated_reviews
            else:
                print("No valid review source selected.")
                return None
        else:
            if final_resulted_output:
                return final_resulted_output
            elif final_resulted_output_of_all_sites:
                return final_resulted_output_of_all_sites
            else:
                print("Couldn't Fetch the Reviews and Emotion Genration Failed")

        return "Function Didnt Properly Execute"

    def get_sentiment_from_website_each_feedback_sentiment(
            self,
            target_website,
            Use_Local_Sentiment_LLM=None,
            Use_Local_General_LLM=None,
            Use_Local_Scraper=None,
            Use_Scraper_API=None,
            Use_Groq_API=None,
            Use_Gemini_API=False,
            Use_Open_API=None,
            Local_Sentiment_LLM=None,
            Local_General_LLM="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            Local_General_LLM_kwargs={
                'temperature': 0.1,
                'top_p': 1
            },
            Groq_API=None,
            OpenAi_API=None,
            Google_API=None,
            HuggingFace_API=None,
            Scraper_api_key=None,
            Local_api_key=None,
            Groq_LLM="llama3-8b-8192",
            OpenAI_LLM="GPT-3.5",
            Gemini_LLM="gemini-1.5-pro",
            device_map="auto",
            get_Groq_Review=False, 
            get_Gemini_Review=False,
            get_OpenAI_review=False,
            get_localLLM_review=False,
            Groq_LLM_Temperature=0.1,
            Groq_LLM_Max_Tokens=100,
            Groq_LLM_Max_Input_Tokens=300,
            Groq_LLM_top_p=1,
            Groq_LLM_stream=False,
            OpenAI_LLM_Temperature=0.1,
            OpenAI_LLM_Max_Tokens=100,
            OpenAI_LLM_stream=False,
            OpenAI_LLM_Max_Input_Tokens=300,
            Local_LLM_Max_Input_Tokens=300,
            Gemini_LLM_Temperature=0.1,
            Gemini_LLM_Max_Tokens=100,
            Gemini_LLM_Max_Input_Tokens=300


    ):
        """
        Fetches reviews from specified websites and performs sentiment analysis on each review.
        Supports different methods for fetching and analyzing reviews, including local LLMs and APIs.
        Returns sentiment analysis results for the fetched reviews, optionally including additional comments from other LLMs.

        Parameters:
        - target_website: A string or list of strings representing the websites to fetch reviews from.
        - Use_Local_Sentiment_LLM: Boolean indicating whether to use the local sentiment LLM.
        - Use_Local_General_LLM: Boolean indicating whether to use the local general LLM.
        - Use_Local_Scraper: Boolean indicating whether to use a local scraper.
        - Use_Scraper_API: Boolean indicating whether to use a scraper API.
        - Use_Groq_API: Boolean indicating whether to use the Groq API.
        - Use_Open_API: Boolean indicating whether to use the OpenAI API.
        - Local_Sentiment_LLM: The specific local sentiment model to use.
        - Local_General_LLM: The specific local general model to use.
        - Local_General_LLM_kwargs: Additional arguments for the local general LLM.
        - Groq_API: API key for the Groq API.
        - OpenAi_API: API key for OpenAI.
        - HuggingFace_API: API key for Hugging Face.
        - Scraper_api_key: API key for the scraper service.
        - Local_api_key: API key for the local API.
        - Groq_LLM: Model identifier for Groq LLM.
        - OpenAI_LLM: Model identifier for OpenAI LLM.
        - device_map: Specifies the device to run the model on (e.g., "cpu" or "cuda").
        - get_Groq_Review: Boolean indicating whether to get additional comments from Groq LLM.
        - get_OpenAI_review: Boolean indicating whether to get additional comments from OpenAI.
        - get_localLLM_review: Boolean indicating whether to get additional comments from the local LLM.
        - Groq_LLM_Temperature: Temperature setting for the Groq LLM.
        - Groq_LLM_Max_Tokens: Maximum tokens for the Groq LLM.
        - Groq_LLM_Max_Input_Tokens: Maximum input tokens for the Groq LLM.
        - Groq_LLM_top_p: Top-p value for the Groq LLM.
        - Groq_LLM_stream: Boolean indicating whether to use streaming for Groq LLM.
        - OpenAI_LLM_Temperature: Temperature setting for the OpenAI LLM.
        - OpenAI_LLM_Max_Tokens: Maximum tokens for the OpenAI LLM.
        - OpenAI_LLM_stream: Boolean indicating whether to use streaming for OpenAI LLM.
        - OpenAI_LLM_Max_Input_Tokens: Maximum input tokens for the OpenAI LLM.
        - Local_LLM_Max_Input_Tokens: Maximum input tokens for the local LLM.

        Returns:
        - Sentiment analysis results for the fetched reviews, optionally with additional comments from Groq, OpenAI, or local LLMs.
        """

        Use_Local_Scraper = Use_Local_Scraper if Use_Local_Scraper is not None else self.Use_Local_Scraper
        Use_Scraper_API = Use_Scraper_API if Use_Scraper_API is not None else self.Use_Scraper_API
        Scraper_api_key = Scraper_api_key if Scraper_api_key is not None else self.Scraper_api_key
        Local_api_key = Local_api_key if Local_api_key is not None else self.Local_api_key
        Local_Sentiment_LLM = Local_Sentiment_LLM if Local_Sentiment_LLM is not None else self.Local_Sentiment_LLM
        device_map = device_map if device_map is not None else self.device_map

        Groq_API = Groq_API if Groq_API is not None else self.Groq_API
        Use_Groq_API = Use_Groq_API if Use_Groq_API is not None else self.Use_Groq_API
        Groq_LLM = Groq_LLM if Groq_LLM is not None else self.Groq_LLM

        Use_Open_API = Use_Open_API if Use_Open_API is not None else self.Use_Open_API
        OpenAi_API = OpenAi_API if OpenAi_API is not None else self.OpenAi_API
        OpenAI_LLM = OpenAI_LLM if OpenAI_LLM is not None else self.OpenAI_LLM

        Google_API = Google_API if Google_API is not None else self.Google_API
        Use_Gemini_API = Use_Gemini_API if Use_Gemini_API is not None else self.Use_Gemini_API
        Gemini_LLM = Gemini_LLM if Gemini_LLM is not None else self.Gemini_LLM

        Use_Local_General_LLM = Use_Local_General_LLM if Use_Local_General_LLM is not None else self.Use_Local_General_LLM
        Local_General_LLM = Local_General_LLM if Local_General_LLM is not None else self.Local_General_LLM
        Local_General_LLM_kwargs = Local_General_LLM_kwargs if Local_General_LLM_kwargs is not None else self.Local_General_LLM_kwargs
        HuggingFace_API = HuggingFace_API if HuggingFace_API is not None else self.HuggingFace_API

        fetched_review = []
        fetched_review_array = []
        final_resulted_output = None
        final_resulted_output_of_all_sites = []

        # Getting the Reviews From Site
        if isinstance(target_website, str):
            fetched_review = Fetch_Reviews(
                target_website,
                Use_Local_Scraper,
                Use_Scraper_API,
                Scraper_api_key,
                Local_api_key
            )
        elif isinstance(target_website, list) and all(isinstance(item, str) for item in target_website):
            for site in target_website:
                fetched_review_temp = Fetch_Reviews(
                    site,
                    Use_Local_Scraper,
                    Use_Scraper_API,
                    Scraper_api_key,
                    Local_api_key
                )
                fetched_review_array.append(fetched_review_temp)
        else:
            return "Error: The accepted format is either a list of strings or a single string."

        # Genrating Sentiment For Those Reviews
        if fetched_review:
            final_resulted_output = Struct_Generation_Pipeline(
                text_message=fetched_review,
                Use_Local_Sentiment_LLM=True,
                model_id=Local_Sentiment_LLM,
                device_map=device_map
            )
        elif fetched_review_array:
            webcount = 1
            for each_website_reviews in fetched_review_array:
                final_resulted_output_of_each_site = Struct_Generation_Pipeline(
                    text_message=each_website_reviews,
                    Use_Local_Sentiment_LLM=True,
                    model_id=Local_Sentiment_LLM,
                    device_map=device_map
                )
                final_resulted_output_of_each_site.insert(0, webcount)
                webcount = webcount + 1
                final_resulted_output_of_all_sites.append(
                    final_resulted_output_of_each_site)
        else:
            return "Error: Didnt Find Any Inputs"

        # Adding Additional Comments To those Reviews
        if get_Groq_Review == True or get_OpenAI_review == True or get_localLLM_review == True or get_Gemini_Review == True:
            if get_localLLM_review:  # Local LLM Inference
                if final_resulted_output:  # Single Website Given by the user
                    updated_reviews = LocalLLM_inference_list(
                        final_resulted_output,
                        model_name=Local_General_LLM,
                        Local_General_LLM_kwargs=Local_General_LLM_kwargs,
                        max_input_tokens=Local_LLM_Max_Input_Tokens,
                        device_map=device_map
                    )
                    return updated_reviews
                elif final_resulted_output_of_all_sites:  # Multiple Website given by the user
                    all_sites_updated_reviews = []
                    for site_reviews in final_resulted_output_of_all_sites:
                        site_id = site_reviews[0]
                        reviews = site_reviews[1:]
                        updated_reviews = LocalLLM_inference_list(
                            final_resulted_output,
                            model_name=Local_General_LLM,
                            Local_General_LLM_kwargs=Local_General_LLM_kwargs,
                            max_input_tokens=Local_LLM_Max_Input_Tokens,
                            device_map=device_map
                        )
                        all_sites_updated_reviews.append(
                            [site_id] + updated_reviews_site)
                    return all_sites_updated_reviews
                else:
                    print("Error: Didn't receive any result from the sites")
                    exit()
            elif get_Groq_Review:  # Groq LLM Inference
                if final_resulted_output:  # Single Website Given by the user
                    updated_reviews = Groq_inference_list(
                        final_resulted_output,
                        Groq_API,
                        temperature=Groq_LLM_Temperature,
                        top_p=Groq_LLM_top_p,
                        stream=Groq_LLM_stream,
                        max_input_tokens=Groq_LLM_Max_Input_Tokens,
                        max_tokens=Groq_LLM_Max_Tokens
                    )
                    return updated_reviews
                elif final_resulted_output_of_all_sites:  # Multiple Website given by the user
                    all_sites_updated_reviews = []
                    for site_reviews in final_resulted_output_of_all_sites:
                        site_id = site_reviews[0]
                        reviews = site_reviews[1:]
                        updated_reviews_site = Groq_inference_list(
                            reviews,
                            Groq_API,
                            temperature=Groq_LLM_Temperature,
                            top_p=Groq_LLM_top_p,
                            stream=Groq_LLM_stream,
                            max_input_tokens=Groq_LLM_Max_Input_Tokens,
                            max_tokens=Groq_LLM_Max_Tokens
                        )
                        all_sites_updated_reviews.append(
                            [site_id] + updated_reviews_site)
                    return all_sites_updated_reviews
                else:
                    print("Error : Didnt Recieve any result from the sites")
                    exit()
            elif get_Gemini_Review:  # Groq LLM Inference
                if final_resulted_output:  # Single Website Given by the user
                    updated_reviews = Gemini_inference_list(
                        reviews=final_resulted_output,
                        KEY=Google_API,
                        temperature=Gemini_LLM_Temperature,
                        max_input_tokens=Gemini_LLM_Max_Input_Tokens,
                        max_tokens=Gemini_LLM_Max_Tokens,
                        model=Gemini_LLM
                    )
                    return updated_reviews
                elif final_resulted_output_of_all_sites:  # Multiple Website given by the user
                    all_sites_updated_reviews = []
                    for site_reviews in final_resulted_output_of_all_sites:
                        site_id = site_reviews[0]
                        reviews = site_reviews[1:]
                        updated_reviews = Gemini_inference_list(
                            reviews=final_resulted_output,
                            KEY=Google_API,
                            temperature=Gemini_LLM_Temperature,
                            max_input_tokens=Gemini_LLM_Max_Input_Tokens,
                            max_tokens=Gemini_LLM_Max_Tokens,
                            model=Gemini_LLM
                        )
                        all_sites_updated_reviews.append(
                            [site_id] + updated_reviews_site)
                    return all_sites_updated_reviews
                else:
                    print("Error : Didnt Recieve any result from the sites")
                    exit()
            elif get_OpenAI_review:  # OpenAI LLM Inference
                if final_resulted_output:  # Single Website Given by the user
                    updated_reviews = OpenAI_inference_list(
                        final_resulted_output,
                        OpenAi_API,
                        temperature=OpenAI_LLM_Temperature,
                        max_tokens=OpenAI_LLM_Max_Tokens,
                        stream=OpenAI_LLM_stream,
                        max_input_tokens=OpenAI_LLM_Max_Input_Tokens
                    )
                    return updated_reviews
                elif final_resulted_output_of_all_sites:  # Multiple Website given by the user
                    all_sites_updated_reviews = []
                    for site_reviews in final_resulted_output_of_all_sites:
                        site_id = site_reviews[0]
                        reviews = site_reviews[1:]
                        updated_reviews_site = OpenAI_inference_list(
                            reviews,
                            OpenAi_API,  # Add your OpenAI API key here
                            temperature=OpenAI_LLM_Temperature,
                            max_tokens=OpenAI_LLM_Max_Tokens,
                            stream=OpenAI_LLM_stream,
                            max_input_tokens=OpenAI_LLM_Max_Input_Tokens
                        )
                        all_sites_updated_reviews.append(
                            [site_id] + updated_reviews_site)
                    return all_sites_updated_reviews
            else:
                print("No valid review source selected.")
                return None
        else:
            if final_resulted_output:
                return final_resulted_output
            elif final_resulted_output_of_all_sites:
                return final_resulted_output_of_all_sites
            else:
                print("Couldn't Fetch the Reviews and Sentiment Genration Failed")

        return "Function Didnt Properly Execute"

    def get_sentiment_from_website_overall_summary(
            self,
            target_website,
            Use_Local_Sentiment_LLM=None,
            Use_Local_General_LLM=None,
            Use_Local_Scraper=None,
            Use_Scraper_API=None,
            Use_Groq_API=None,
            Use_Open_API=None,
            Use_Gemini_API=None,
            Local_Sentiment_LLM=None,
            Local_General_LLM="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            Local_General_LLM_kwargs={
                'temperature': 0.1,
                'top_p': 1
            },
            Groq_API=None,
            OpenAi_API=None,
            HuggingFace_API=None,
            Scraper_api_key=None,
            Google_API=None,
            Local_api_key=None,
            Groq_LLM="llama3-8b-8192",
            OpenAI_LLM="GPT-3.5",
            device_map="auto",
            Gemini_LLM="gemini-1.5-pro",
            Groq_LLM_Temperature=0.1,
            Groq_LLM_Max_Tokens=100,
            Groq_LLM_Max_Input_Tokens=300,
            Groq_LLM_top_p=1,
            Groq_LLM_stream=False,
            OpenAI_LLM_Temperature=0.1,
            OpenAI_LLM_Max_Tokens=100,
            OpenAI_LLM_stream=False,
            OpenAI_LLM_Max_Input_Tokens=300,
            Local_LLM_Max_Input_Tokens=300,
            Gemini_LLM_Temperature=0.1,
            Gemini_LLM_Max_Tokens=100,
            Gemini_LLM_Max_Input_Tokens=300,
            Ollama_Model="llama3.1",
            Ollama_Model_EndPoint = "http://localhost:11434/api/generate" 

    ):
        """
        Fetches reviews from a specified website and generates an overall sentiment summary.
        Uses various methods for sentiment analysis and summarization, depending on the available options and configurations.

        Parameters:
        - target_website: The website from which to fetch reviews.
        - Use_Local_Sentiment_LLM: Boolean indicating whether to use the local sentiment LLM for initial sentiment analysis.
        - Use_Local_General_LLM: Boolean indicating whether to use a local general LLM for summarization.
        - Use_Local_Scraper: Boolean indicating whether to use a local scraper to fetch reviews.
        - Use_Scraper_API: Boolean indicating whether to use a scraper API for fetching reviews.
        - Use_Groq_API: Boolean indicating whether to use the Groq API for summarization.
        - Use_Open_API: Boolean indicating whether to use the OpenAI API for summarization.
        - Local_Sentiment_LLM: Model identifier for the local sentiment LLM.
        - Local_General_LLM: Model identifier for the local general LLM used for summarization.
        - Local_General_LLM_kwargs: Additional arguments for the local general LLM.
        - Groq_API: API key for the Groq API.
        - OpenAi_API: API key for the OpenAI API.
        - HuggingFace_API: API key for Hugging Face.
        - Scraper_api_key: API key for the scraper service.
        - Local_api_key: API key for the local API.
        - Groq_LLM: Model identifier for the Groq LLM.
        - OpenAI_LLM: Model identifier for the OpenAI LLM.
        - device_map: Device to run the model on (e.g., "cpu" or "cuda").
        - Groq_LLM_Temperature: Temperature setting for the Groq LLM.
        - Groq_LLM_Max_Tokens: Maximum tokens for the Groq LLM.
        - Groq_LLM_Max_Input_Tokens: Maximum input tokens for the Groq LLM.
        - Groq_LLM_top_p: Top-p value for the Groq LLM.
        - Groq_LLM_stream: Boolean indicating whether to use streaming for Groq LLM.
        - OpenAI_LLM_Temperature: Temperature setting for the OpenAI LLM.
        - OpenAI_LLM_Max_Tokens: Maximum tokens for the OpenAI LLM.
        - OpenAI_LLM_stream: Boolean indicating whether to use streaming for OpenAI LLM.
        - OpenAI_LLM_Max_Input_Tokens: Maximum input tokens for the OpenAI LLM.
        - Local_LLM_Max_Input_Tokens: Maximum input tokens for the local LLM.

        Returns:
        - Summarized sentiment results based on the selected summarization method (Groq API, OpenAI API, or local general LLM).
        - Error message if no reviews were fetched or no LLM inference method is selected.
        """

        Use_Local_Scraper = Use_Local_Scraper if Use_Local_Scraper is not None else self.Use_Local_Scraper
        Use_Scraper_API = Use_Scraper_API if Use_Scraper_API is not None else self.Use_Scraper_API
        Scraper_api_key = Scraper_api_key if Scraper_api_key is not None else self.Scraper_api_key
        Local_api_key = Local_api_key if Local_api_key is not None else self.Local_api_key
        Local_Sentiment_LLM = Local_Sentiment_LLM if Local_Sentiment_LLM is not None else self.Local_Sentiment_LLM
        device_map = device_map if device_map is not None else self.device_map

        Groq_API = Groq_API if Groq_API is not None else self.Groq_API
        Use_Groq_API = Use_Groq_API if Use_Groq_API is not None else self.Use_Groq_API
        Groq_LLM = Groq_LLM if Groq_LLM is not None else self.Groq_LLM

        Google_API = Google_API if Google_API is not None else self.Google_API
        Use_Gemini_API = Use_Gemini_API if Use_Gemini_API is not None else self.Use_Gemini_API
        Gemini_LLM = Gemini_LLM if Gemini_LLM is not None else self.Gemini_LLM

        Use_Open_API = Use_Open_API if Use_Open_API is not None else self.Use_Open_API
        OpenAi_API = OpenAi_API if OpenAi_API is not None else self.OpenAi_API
        OpenAI_LLM = OpenAI_LLM if OpenAI_LLM is not None else self.OpenAI_LLM

        Use_Local_General_LLM = Use_Local_General_LLM if Use_Local_General_LLM is not None else self.Use_Local_General_LLM
        Local_General_LLM = Local_General_LLM if Local_General_LLM is not None else self.Local_General_LLM
        Local_General_LLM_kwargs = Local_General_LLM_kwargs if Local_General_LLM_kwargs is not None else self.Local_General_LLM_kwargs
        HuggingFace_API = HuggingFace_API if HuggingFace_API is not None else self.HuggingFace_API
        final_resulted_output = None

        
        if target_website:
            try:
                fetched_review = Fetch_Reviews(
                    target_website,
                    Use_Local_Scraper,
                    Use_Scraper_API,
                    Scraper_api_key,
                    Local_api_key
                )

                if fetched_review:
                    final_resulted_output = Struct_Generation_Pipeline(
                        text_message=fetched_review,
                        Use_Local_Sentiment_LLM=True,
                        model_id=Local_Sentiment_LLM,
                        device_map=device_map
                    )
                else:
                    raise ValueError("Couldn't Fetch the Reviews From the Site")

            except ValueError as e:
                print(f"Error: {e}")
                exit()

            except ConnectionError:
                print(
                    "Error: Unable to connect to the website. Please check your internet connection.")
                exit()

            except TimeoutError:
                print("Error: The request timed out. Try again later.")
                exit()

            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                exit()

            if Use_Groq_API == True or Use_Open_API == True or Use_Local_General_LLM == True or Use_Gemini_API == True:
                if Use_Groq_API:
                    if final_resulted_output:
                        Summarized_result_Groq = summarize_reviews(
                            reviews=final_resulted_output,
                            KEY=Groq_API,
                            max_tokens=Groq_LLM_Max_Tokens,
                            temperature=Groq_LLM_Temperature,
                            top_p=Groq_LLM_top_p,
                            stream=Groq_LLM_stream,
                            model_id=Groq_LLM
                        )
                        return Summarized_result_Groq
                    else:
                        print("Couldnt Return Formatted Output")
                        exit()
                elif Use_Gemini_API:
                    if final_resulted_output:
                        Summarized_result_Gemini = summarize_reviews_gemini(
                            reviews=final_resulted_output,
                            KEY=Google_API,
                            max_tokens=Gemini_LLM_Max_Tokens,
                            temperature=Gemini_LLM_Temperature,
                            model=Gemini_LLM
                        )
                        return Summarized_result_Gemini
                    else:
                        print("Couldnt Return Formatted Output")
                        exit()
                elif Use_Open_API:
                    if final_resulted_output:
                        Summarized_result_OpenAi = summarize_reviews_openai(
                            reviews=final_resulted_output,
                            KEY=OpenAi_API,
                            model_id=OpenAI_LLM,
                            max_tokens=OpenAI_LLM_Max_Tokens,
                            temperature=OpenAI_LLM_Temperature,
                            stream=OpenAI_LLM_stream
                        )
                        return Summarized_result_OpenAi
                    else:
                        print("Couldnt Return Formatted Output")
                        exit()
                elif Use_Local_General_LLM:
                    if final_resulted_output:
                        Summarized_result_LocalLLM = Ollama_Local_Summarize(
                            reviews=final_resulted_output,
                            Model_Name=Ollama_Model,
                            Ollama_Model_EndPoint=Ollama_Model_EndPoint
                        )
                        return Summarized_result_LocalLLM
                    else:
                        print("Couldnt Return Formatted Output")
                else:
                    print("Error : use any one of the LLM inference")
        else:
            raise ValueError("Error : No URL supplied")
        
    def get_Emotion_from_website_overall_summary(
            self,
            target_website,
            Use_Local_Emotion_LLM=None,
            Use_Local_General_LLM=None,
            Use_Local_Scraper=None,
            Use_Scraper_API=None,
            Use_Groq_API=None,
            Use_Open_API=None,
            Use_Gemini_API=None,
            Local_Emotion_LLM="SamLowe/roberta-base-go_emotions",
            Local_General_LLM="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            Local_General_LLM_kwargs={
                'temperature': 0.1,
                'top_p': 1
            },
            Groq_API=None,
            OpenAi_API=None,
            HuggingFace_API=None,
            Scraper_api_key=None,
            Google_API=None,
            Local_api_key=None,
            Groq_LLM="llama3-8b-8192",
            OpenAI_LLM="GPT-3.5",
            device_map="auto",
            Gemini_LLM="gemini-1.5-pro",
            Groq_LLM_Temperature=0.1,
            Groq_LLM_Max_Tokens=100,
            Groq_LLM_Max_Input_Tokens=300,
            Groq_LLM_top_p=1,
            Groq_LLM_stream=False,
            OpenAI_LLM_Temperature=0.1,
            OpenAI_LLM_Max_Tokens=100,
            OpenAI_LLM_stream=False,
            OpenAI_LLM_Max_Input_Tokens=300,
            Local_LLM_Max_Input_Tokens=300,
            Gemini_LLM_Temperature=0.1,
            Gemini_LLM_Max_Tokens=100,
            Gemini_LLM_Max_Input_Tokens=300,
            Ollama_Model="llama3.1",
            Ollama_Model_EndPoint = "http://localhost:11434/api/generate" 

    ):
        """
        Fetches reviews from a specified website and generates an overall sentiment summary.
        Uses various methods for sentiment analysis and summarization, depending on the available options and configurations.

        Parameters:
        - target_website: The website from which to fetch reviews.
        - Use_Local_Sentiment_LLM: Boolean indicating whether to use the local sentiment LLM for initial sentiment analysis.
        - Use_Local_General_LLM: Boolean indicating whether to use a local general LLM for summarization.
        - Use_Local_Scraper: Boolean indicating whether to use a local scraper to fetch reviews.
        - Use_Scraper_API: Boolean indicating whether to use a scraper API for fetching reviews.
        - Use_Groq_API: Boolean indicating whether to use the Groq API for summarization.
        - Use_Open_API: Boolean indicating whether to use the OpenAI API for summarization.
        - Local_Sentiment_LLM: Model identifier for the local sentiment LLM.
        - Local_General_LLM: Model identifier for the local general LLM used for summarization.
        - Local_General_LLM_kwargs: Additional arguments for the local general LLM.
        - Groq_API: API key for the Groq API.
        - OpenAi_API: API key for the OpenAI API.
        - HuggingFace_API: API key for Hugging Face.
        - Scraper_api_key: API key for the scraper service.
        - Local_api_key: API key for the local API.
        - Groq_LLM: Model identifier for the Groq LLM.
        - OpenAI_LLM: Model identifier for the OpenAI LLM.
        - device_map: Device to run the model on (e.g., "cpu" or "cuda").
        - Groq_LLM_Temperature: Temperature setting for the Groq LLM.
        - Groq_LLM_Max_Tokens: Maximum tokens for the Groq LLM.
        - Groq_LLM_Max_Input_Tokens: Maximum input tokens for the Groq LLM.
        - Groq_LLM_top_p: Top-p value for the Groq LLM.
        - Groq_LLM_stream: Boolean indicating whether to use streaming for Groq LLM.
        - OpenAI_LLM_Temperature: Temperature setting for the OpenAI LLM.
        - OpenAI_LLM_Max_Tokens: Maximum tokens for the OpenAI LLM.
        - OpenAI_LLM_stream: Boolean indicating whether to use streaming for OpenAI LLM.
        - OpenAI_LLM_Max_Input_Tokens: Maximum input tokens for the OpenAI LLM.
        - Local_LLM_Max_Input_Tokens: Maximum input tokens for the local LLM.

        Returns:
        - Summarized sentiment results based on the selected summarization method (Groq API, OpenAI API, or local general LLM).
        - Error message if no reviews were fetched or no LLM inference method is selected.
        """

        Use_Local_Scraper = Use_Local_Scraper if Use_Local_Scraper is not None else self.Use_Local_Scraper
        Use_Scraper_API = Use_Scraper_API if Use_Scraper_API is not None else self.Use_Scraper_API
        Scraper_api_key = Scraper_api_key if Scraper_api_key is not None else self.Scraper_api_key
        Local_api_key = Local_api_key if Local_api_key is not None else self.Local_api_key
        Local_Emotion_LLM = Local_Emotion_LLM if Local_Emotion_LLM is not None else self.Local_Emotion_LLM
        device_map = device_map if device_map is not None else self.device_map

        Groq_API = Groq_API if Groq_API is not None else self.Groq_API
        Use_Groq_API = Use_Groq_API if Use_Groq_API is not None else self.Use_Groq_API
        Groq_LLM = Groq_LLM if Groq_LLM is not None else self.Groq_LLM

        Google_API = Google_API if Google_API is not None else self.Google_API
        Use_Gemini_API = Use_Gemini_API if Use_Gemini_API is not None else self.Use_Gemini_API
        Gemini_LLM = Gemini_LLM if Gemini_LLM is not None else self.Gemini_LLM

        Use_Open_API = Use_Open_API if Use_Open_API is not None else self.Use_Open_API
        OpenAi_API = OpenAi_API if OpenAi_API is not None else self.OpenAi_API
        OpenAI_LLM = OpenAI_LLM if OpenAI_LLM is not None else self.OpenAI_LLM

        Use_Local_General_LLM = Use_Local_General_LLM if Use_Local_General_LLM is not None else self.Use_Local_General_LLM
        Local_General_LLM = Local_General_LLM if Local_General_LLM is not None else self.Local_General_LLM
        Local_General_LLM_kwargs = Local_General_LLM_kwargs if Local_General_LLM_kwargs is not None else self.Local_General_LLM_kwargs
        HuggingFace_API = HuggingFace_API if HuggingFace_API is not None else self.HuggingFace_API
        final_resulted_output = None

        
        if target_website:
            try:
                fetched_review = Fetch_Reviews(
                    target_website,
                    Use_Local_Scraper,
                    Use_Scraper_API,
                    Scraper_api_key,
                    Local_api_key
                )

                if fetched_review:
                    final_resulted_output = Struct_Emotion(
                        text_message=fetched_review,
                        Use_Local_Emotion_LLM=True,
                        model_id=Local_Emotion_LLM,
                        device_map=device_map
                    )
                else:
                    raise ValueError("Couldn't Fetch the Reviews From the Site")

            except ValueError as e:
                print(f"Error: {e}")
                exit()

            except ConnectionError:
                print(
                    "Error: Unable to connect to the website. Please check your internet connection.")
                exit()

            except TimeoutError:
                print("Error: The request timed out. Try again later.")
                exit()

            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                exit()

            if Use_Groq_API == True or Use_Open_API == True or Use_Local_General_LLM == True or Use_Gemini_API == True:
                if Use_Groq_API:
                    if final_resulted_output:
                        Summarized_result_Groq = summarize_Emotions_reviews(
                            emotions=final_resulted_output,
                            KEY=Groq_API,
                            max_tokens=Groq_LLM_Max_Tokens,
                            temperature=Groq_LLM_Temperature,
                            top_p=Groq_LLM_top_p,
                            stream=Groq_LLM_stream,
                            model_id=Groq_LLM
                        )
                        return Summarized_result_Groq
                    else:
                        print("Couldnt Return Formatted Output")
                        exit()
                elif Use_Gemini_API:
                    if final_resulted_output:
                        Summarized_result_Gemini = summarize_Emotion_reviews_gemini(
                            emotions=final_resulted_output,
                            KEY=Google_API,
                            max_tokens=Gemini_LLM_Max_Tokens,
                            temperature=Gemini_LLM_Temperature,
                            model=Gemini_LLM
                        )
                        return Summarized_result_Gemini
                    else:
                        print("Couldnt Return Formatted Output")
                        exit()
                elif Use_Open_API:
                    if final_resulted_output:
                        Summarized_result_OpenAi = summarize_reviews_openai(
                            reviews=final_resulted_output,
                            KEY=OpenAi_API,
                            model_id=OpenAI_LLM,
                            max_tokens=OpenAI_LLM_Max_Tokens,
                            temperature=OpenAI_LLM_Temperature,
                            stream=OpenAI_LLM_stream
                        )
                        return Summarized_result_OpenAi
                    else:
                        print("Couldnt Return Formatted Output")
                        exit()
                elif Use_Local_General_LLM:
                    if final_resulted_output:
                        Summarized_result_LocalLLM = Ollama_Local_Emotion_Summarize(
                            emotions=final_resulted_output,
                            Model_Name=Ollama_Model,
                            Ollama_Model_EndPoint=Ollama_Model_EndPoint
                        )
                        return Summarized_result_LocalLLM
                    else:
                        print("Couldnt Return Formatted Output")
                else:
                    print("Error : use any one of the LLM inference")
        else:
            raise ValueError("Error : No URL supplied")
    
    def get_analytical_customer_sentiments(
            self,
            target_website,
            Use_Local_Sentiment_LLM=True,
            Use_Local_Emotion_LLM = True,
            Use_Local_Scraper=None,
            Use_Scraper_API=None,
            Scraper_api_key=None,
            Local_api_key=None,
            Use_Bar_chart_visualize=False,
            Use_pie_chart_visualize=False,
            Use_violin_plot_visualize=False,
            Use_box_plot_visualize=False,
            Use_histogram_visualize=False,
            Use_Card_Emotion_Visulize=False,
            device_map="auto",
            Local_Emotion_LLM="SamLowe/roberta-base-go_emotions",
            Local_Sentiment_LLM="cardiffnlp/twitter-roberta-base-sentiment-latest",
    ):
        """
        Fetches reviews from a specified website and performs sentiment analysis, followed by optional visualization of the results.
        Uses a local sentiment LLM for sentiment analysis and can generate various types of visualizations based on user preferences.

        Parameters:
        - target_website: The website from which to fetch reviews.
        - Use_Local_Scraper: Boolean indicating whether to use a local scraper to fetch reviews.
        - Use_Scraper_API: Boolean indicating whether to use a scraper API for fetching reviews.
        - Scraper_api_key: API key for the scraper service.
        - Local_api_key: API key for the local API.
        - Use_Local_Sentiment_LLM: Boolean indicating whether to use the local sentiment LLM for analysis.
        - Use_Bar_chart_visualize: Boolean indicating whether to generate a bar chart visualization of sentiment distribution.
        - Use_pie_chart_visualize: Boolean indicating whether to generate a pie chart visualization of sentiment distribution.
        - Use_violin_plot_visualize: Boolean indicating whether to generate a violin plot visualization of sentiment distribution.
        - Use_box_plot_visualize: Boolean indicating whether to generate a box plot visualization of sentiment distribution.
        - Use_histogram_visualize: Boolean indicating whether to generate a histogram visualization of sentiment scores.
        - device_map: Device to run the model on (e.g., "cpu" or "cuda").
        - Local_Sentiment_LLM: Model identifier for the local sentiment LLM.

        Returns:
        - Sentiment visualizations based on the selected options (bar chart, pie chart, violin plot, box plot, or histogram).
        - Error message if no reviews were fetched.
        """

        Use_Local_Scraper = Use_Local_Scraper if Use_Local_Scraper is not None else self.Use_Local_Scraper
        Use_Scraper_API = Use_Scraper_API if Use_Scraper_API is not None else self.Use_Scraper_API
        Scraper_api_key = Scraper_api_key if Scraper_api_key is not None else self.Scraper_api_key
        Local_api_key = Local_api_key if Local_api_key is not None else self.Local_api_key
        Local_Sentiment_LLM = Local_Sentiment_LLM if Local_Sentiment_LLM is not None else self.Local_Sentiment_LLM
        device_map = device_map if device_map is not None else self.device_map
        Use_Local_Sentiment_LLM = Use_Local_Sentiment_LLM if Use_Local_Sentiment_LLM is not None else self.Use_Local_Sentiment_LLM
        Use_Local_Emotion_LLM = Use_Local_Emotion_LLM if Use_Local_Emotion_LLM is not None else self.Use_Local_Emotion_LLM

        fetched_review = Fetch_Reviews(
            target_website,
            Use_Local_Scraper,
            Use_Scraper_API,
            Scraper_api_key,
            Local_api_key
        )
        if fetched_review:
            if Use_Local_Sentiment_LLM:
                final_resulted_output = Struct_Generation_Pipeline_Visual(
                    text_message=fetched_review,
                    Use_Local_Sentiment_LLM=True,
                    model_id=Local_Sentiment_LLM,
                    device_map=device_map
                )
            if Use_Local_Emotion_LLM:
                final_resulted_output_Emotions = Struct_Emotion(
                    text_message=fetched_review,
                    Use_Local_Emotion_LLM=True,
                    model_id=Local_Emotion_LLM,
                    device_map=device_map
                )
        else:
            print("Error : No Values Have been Fetched")

        if Use_Bar_chart_visualize:
            plot_sentiment_distribution(final_resulted_output)

        if Use_pie_chart_visualize:
            plot_sentiment_pie_chart(final_resulted_output)

        if Use_violin_plot_visualize:
            plot_sentiment_violin_plot(final_resulted_output)

        if Use_box_plot_visualize:
            plot_sentiment_box_plot(final_resulted_output)

        if Use_histogram_visualize:
            plot_sentiment_histograms(final_resulted_output)

        if Use_Card_Emotion_Visulize:
            resulted_top5_emotions = calculate_top_emotions_percentages(reviews_emotions=final_resulted_output_Emotions)
            return resulted_top5_emotions

    def get_Sentiment_Audio_file(
            self,
            Audio_File_path=None,
            Use_Local_Sentiment_LLM=True,
            Local_Sentiment_LLM="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device_map="auto"
    ):
        """
        Analyzes the sentiment of text extracted from an audio file.

        Parameters:
        - Audio_File_path (str): The path to the audio file to be analyzed.
        - Use_Local_Sentiment_LLM (bool): Indicates whether to use a local sentiment analysis model.
        - Local_Sentiment_LLM (str): The identifier for the local sentiment model.
        - device_map (str): The device configuration for model inference.

        Returns:
        - sentiment: The sentiment analysis result of the text extracted from the audio file.
        """

        Use_Local_Sentiment_LLM = Use_Local_Sentiment_LLM if Use_Local_Sentiment_LLM is not None else self.Use_Local_Sentiment_LLM
        Local_Sentiment_LLM = Local_Sentiment_LLM if Local_Sentiment_LLM is not None else self.Local_Sentiment_LLM
        device_map = device_map if device_map is not None else self.device_map
        if Audio_File_path:
            retrieved_text = audio_to_text(Audio_File_path)
            sentiment = self.get_Quick_sentiment(
                text_message=retrieved_text,
                Use_Local_Sentiment_LLM=Use_Local_Sentiment_LLM,
                Local_Sentiment_LLM=Local_Sentiment_LLM,
                device_map=device_map
            )
            return [{'retrieved_text': retrieved_text}, sentiment]
        else:
            print("Mention the Audio Path")

    def compare_product_on_reviews(
        self,
        target_website1,
        target_website2,
        Use_Local_General_LLM=True,
        Use_Local_Sentiment_LLM=True,
        Use_Local_Scraper=None,
        Use_Scraper_API=None,
        Use_Groq_API=None,
        Use_Open_API=None,
        Use_Gemini_API=None,
        Local_Sentiment_LLM=None,
        Get_Graphical_View = False,
        Local_General_LLM="llama3.1",
        Local_General_LLM_kwargs={
            'temperature': 0.1,
            'top_p': 1
        },
        Groq_API=None,
        OpenAi_API=None,
        Google_API=None,
        HuggingFace_API=None,
        Scraper_api_key=None,
        Local_api_key=None,
        Groq_LLM="llama3-8b-8192",
        OpenAI_LLM="GPT-3.5",
        Gemini_LLM="gemini-1.5-pro",
        device_map="auto",
        Groq_LLM_Temperature=0.1,
        Groq_LLM_Max_Tokens=100,
        Groq_LLM_Max_Input_Tokens=300,
        Groq_LLM_top_p=1,
        Groq_LLM_stream=False,
        OpenAI_LLM_Temperature=0.1,
        OpenAI_LLM_Max_Tokens=100,
        OpenAI_LLM_stream=False,
        OpenAI_LLM_Max_Input_Tokens=300,
        Local_LLM_Max_Input_Tokens=300,
        Gemini_LLM_Temperature=0.1,
        Gemini_LLM_Max_Tokens=100,
        Gemini_LLM_Max_Input_Tokens=300,
        Ollama_Model_EndPoint = "http://localhost:11434/api/generate" 
    ):
        """
        Compares product reviews from two different websites using sentiment analysis and other configurations.

        Parameters:
        - target_website1 (str): The URL of the first website to fetch reviews from.
        - target_website2 (str): The URL of the second website to fetch reviews from.
        - Use_Local_Sentiment_LLM (bool): Indicates whether to use a local sentiment analysis model.
        - Use_Local_General_LLM (bool): Indicates whether to use a local general-purpose LLM.
        - Use_Local_Scraper (bool): Indicates whether to use a local web scraper.
        - Use_Scraper_API (bool): Indicates whether to use an external scraper API.
        - Use_Groq_API (bool): Indicates whether to use the Groq API for LLM.
        - Use_Open_API (bool): Indicates whether to use the OpenAI API.
        - Local_Sentiment_LLM (str): Identifier for the local sentiment analysis model.
        - Local_General_LLM (str): Identifier for the local general-purpose LLM.
        - Local_General_LLM_kwargs (dict): Parameters for the local general-purpose LLM.
        - Groq_API (str): API key for Groq LLM.
        - OpenAi_API (str): API key for OpenAI.
        - HuggingFace_API (str): API key for HuggingFace.
        - Scraper_api_key (str): API key for the scraper service.
        - Local_api_key (str): API key for the local services.
        - Groq_LLM (str): Identifier for the Groq LLM.
        - OpenAI_LLM (str): Identifier for the OpenAI LLM.
        - device_map (str): Device configuration for model inference.
        - Groq_LLM_Temperature (float): Temperature setting for Groq LLM.
        - Groq_LLM_Max_Tokens (int): Maximum number of tokens for Groq LLM.
        - Groq_LLM_Max_Input_Tokens (int): Maximum input tokens for Groq LLM.
        - Groq_LLM_top_p (float): Top-p setting for Groq LLM.
        - Groq_LLM_stream (bool): Streaming option for Groq LLM.
        - OpenAI_LLM_Temperature (float): Temperature setting for OpenAI LLM.
        - OpenAI_LLM_Max_Tokens (int): Maximum number of tokens for OpenAI LLM.
        - OpenAI_LLM_stream (bool): Streaming option for OpenAI LLM.
        - OpenAI_LLM_Max_Input_Tokens (int): Maximum input tokens for OpenAI LLM.
        - Local_LLM_Max_Input_Tokens (int): Maximum input tokens for local LLM.

        Returns:
        - compared_reviews: The result of comparing reviews from the two websites.
        """

        Use_Local_Scraper = Use_Local_Scraper if Use_Local_Scraper is not None else self.Use_Local_Scraper
        Use_Scraper_API = Use_Scraper_API if Use_Scraper_API is not None else self.Use_Scraper_API
        Scraper_api_key = Scraper_api_key if Scraper_api_key is not None else self.Scraper_api_key
        Local_api_key = Local_api_key if Local_api_key is not None else self.Local_api_key
        Local_Sentiment_LLM = Local_Sentiment_LLM if Local_Sentiment_LLM is not None else self.Local_Sentiment_LLM
        device_map = device_map if device_map is not None else self.device_map

        Groq_API = Groq_API if Groq_API is not None else self.Groq_API
        Use_Groq_API = Use_Groq_API if Use_Groq_API is not None else self.Use_Groq_API
        Groq_LLM = Groq_LLM if Groq_LLM is not None else self.Groq_LLM

        Google_API = Google_API if Google_API is not None else self.Google_API
        Use_Gemini_API = Use_Gemini_API if Use_Gemini_API is not None else self.Use_Gemini_API
        Gemini_LLM = Gemini_LLM if Gemini_LLM is not None else self.Gemini_LLM

        Use_Open_API = Use_Open_API if Use_Open_API is not None else self.Use_Open_API
        OpenAi_API = OpenAi_API if OpenAi_API is not None else self.OpenAi_API
        OpenAI_LLM = OpenAI_LLM if OpenAI_LLM is not None else self.OpenAI_LLM

        Use_Local_General_LLM = Use_Local_General_LLM if Use_Local_General_LLM is not None else self.Use_Local_General_LLM
        Local_General_LLM = Local_General_LLM if Local_General_LLM is not None else self.Local_General_LLM
        Local_General_LLM_kwargs = Local_General_LLM_kwargs if Local_General_LLM_kwargs is not None else self.Local_General_LLM_kwargs
        HuggingFace_API = HuggingFace_API if HuggingFace_API is not None else self.HuggingFace_API

        Ollama_Model_EndPoint = Ollama_Model_EndPoint if Ollama_Model_EndPoint is not None else self.Ollama_Model_EndPoint

        if isinstance(target_website1, str):
            fetched_review1 = Fetch_Reviews(
                target_website1,
                Use_Local_Scraper,
                Use_Scraper_API,
                Scraper_api_key,
                Local_api_key
            )

        if isinstance(target_website2, str):
            fetched_review2 = Fetch_Reviews(
                target_website2,
                Use_Local_Scraper,
                Use_Scraper_API,
                Scraper_api_key,
                Local_api_key
            )

        if fetched_review1 and fetched_review2:
            final_resulted_output1 = Struct_Generation_Pipeline(
                text_message=fetched_review1,
                Use_Local_Sentiment_LLM=True,
                model_id=Local_Sentiment_LLM,
                device_map=device_map
            )
            final_resulted_output2 = Struct_Generation_Pipeline(
                text_message=fetched_review2,
                Use_Local_Sentiment_LLM=True,
                model_id=Local_Sentiment_LLM,
                device_map=device_map
            )
        else:
            if not fetched_review1 :
                print("Couldn't Find Reviews in Site 1")
            if not fetched_review2:
                print("Could'nt Find Reviews in Site 2")
            print("Error : Coundnt find Reviews in either one of the sites")
            exit()

        if Get_Graphical_View:
            plot_sentiment_comparison(product1_reviews=final_resulted_output1,product2_reviews=final_resulted_output2)
        if final_resulted_output1 and final_resulted_output2:
            if Use_Groq_API==True:
                compared_reviews = compare_reviews_local(
                    reviews1=final_resulted_output1,
                    reviews2=final_resulted_output2,
                    KEY=Groq_API,
                    temperature=Groq_LLM_Temperature,
                    top_p=Groq_LLM_top_p,
                    stream=Groq_LLM_stream,
                    max_input_tokens=Groq_LLM_Max_Input_Tokens,
                    max_tokens=Groq_LLM_Max_Tokens,
                    model_id=Groq_LLM
                )
                return compared_reviews
            elif Use_Gemini_API==True:
                compared_reviews = compare_reviews_gemini(
                    reviews1=final_resulted_output1,
                    reviews2=final_resulted_output2,
                    KEY=Google_API,
                    max_tokens=Gemini_LLM_Max_Tokens,
                    temperature=Gemini_LLM_Temperature,
                    model=Gemini_LLM
                )
                return compared_reviews
            elif Use_Open_API==True:
                print("Still Under Development")
                exit()
            elif Use_Local_General_LLM==True:
                compared_reviews = Ollama_Local_Sentiment_Comparsion(
                    review1=final_resulted_output1,
                    review2=final_resulted_output2,
                    Ollama_Model_EndPoint = Ollama_Model_EndPoint,
                    Model_Name=Local_General_LLM
                )
                return compared_reviews
            else:
                print("Select a LLM for infernce")
                exit()
        else:
            print("Error : Sentiment Generation Error")
            exit()

    def get_Sentiment_Image_file(
            self,
            Image_File_path=None,
            Custom_Prompt="Explain what's happening in the image and give a detailed response. Also mention what kind of emotion does the image about",
            Use_Local_Sentiment_LLM=True,
            Local_Sentiment_LLM="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device_map="auto",
            LLava_Model="llava",
            Ollama_Model_EndPoint = "http://localhost:11434/api/generate"           
    ):
        """
        Analyzes the sentiment of text extracted from an image file.

        Parameters:
        - Image_File_path (str): The path to the image file to be analyzed.
        - Custom_Prompt (str): Custom prompt for the image-to-text model.
        - Use_Local_Sentiment_LLM (bool): Indicates whether to use a local sentiment analysis model.
        - Local_Sentiment_LLM (str): Identifier for the local sentiment model.
        - device_map (str): The device configuration for model inference.

        Returns:
        - result: A dictionary containing the extracted text and its sentiment, or an error message if text extraction fails.
        """

        Use_Local_Sentiment_LLM = Use_Local_Sentiment_LLM if Use_Local_Sentiment_LLM is not None else self.Use_Local_Sentiment_LLM
        Local_Sentiment_LLM = Local_Sentiment_LLM if Local_Sentiment_LLM is not None else self.Local_Sentiment_LLM
        device_map = device_map if device_map is not None else self.device_map

        if Image_File_path:
                Extracted_Text = Generate_Summary_From_Image(
                    image_path=Image_File_path,
                    Prompt=Custom_Prompt,
                    Ollama_Model_EndPoint=Ollama_Model_EndPoint,
                    Model_Name=LLava_Model
                )
                if Extracted_Text:
                    sentiment = self.get_Quick_sentiment(
                        text_message=Extracted_Text,
                        Use_Local_Sentiment_LLM=Use_Local_Sentiment_LLM,
                        Local_Sentiment_LLM=Local_Sentiment_LLM,
                        device_map=device_map
                    )
                    return [{'Extracted_Text': Extracted_Text}, sentiment]
                else:
                    print("Error : Couldn't Extract any text")
                    exit()
        else:
            print("Error : Image path Not Provided")
            exit()

    def Multi_language_Sentiment(
            self,
            text_message,
            Use_Local_Sentiment_LLM=True,
            Local_Sentiment_LLM=None,
            device_map=None
    ):
        """
        Analyzes sentiment for a given text message by first translating it to a common language.

        Parameters:
        - text_message (str): The text message to analyze sentiment for.
        - Use_Local_Sentiment_LLM (bool): Indicates whether to use a local sentiment analysis model.
        - Local_Sentiment_LLM (str): The identifier for the local sentiment model.
        - device_map (str): The device configuration for model inference.

        Returns:
        - result: A list containing the original text, translated text, and sentiment analysis result. If translation fails, an error message is printed. If no message is provided, an error message is printed.
        """
        Use_Local_Sentiment_LLM = Use_Local_Sentiment_LLM if Use_Local_Sentiment_LLM is not None else self.Use_Local_Sentiment_LLM
        Local_Sentiment_LLM = Local_Sentiment_LLM if Local_Sentiment_LLM is not None else self.Local_Sentiment_LLM
        device_map = device_map if device_map is not None else self.device_map

        if text_message:
            translated_text = Translate_text(message=text_message)
            if translated_text:
                sentiment = self.get_Quick_sentiment(
                    text_message=translated_text,
                    Use_Local_Sentiment_LLM=Use_Local_Sentiment_LLM,
                    Local_Sentiment_LLM=Local_Sentiment_LLM,
                    device_map=device_map
                )
                return [{'Original Text': text_message}, {'Translated Text': translated_text}, sentiment]
            else:
                print("Error : Couldn't Translate any text")
        else:
            print("Error : No message Recieved")

    def Config_Local_Scraper(
            self,
            action,
            tag=None,
            attrs=None,
    ):
        """
        Configures the local web scraper by adding or retrieving review patterns.

        Parameters:
        - action (str): Specifies the action to perform ('add' or 'get').
        - tag (str, optional): The HTML tag to add for review scraping.
        - attrs (dict, optional): Attributes associated with the tag for review scraping.

        Returns:
        - result: The retrieved review patterns if the action is 'get'. If an invalid action is provided, an error message is printed.
        """
        if action == 'add':
            if tag and attrs:
                add_review_pattern(tag=tag, attrs=attrs)
        elif action == 'get':
            result = get_review_patterns()
            return result
        else:
            print("Error : Invalid Action (Either Set to 'add' or 'get')")

    def Save_reviews_to_CSV(
            self,
            target_site,
            output_dir,
            file_name,
            Use_Local_Sentiment_LLM=None,
            Use_Local_Scraper=None
    ):
        """
        Fetches reviews from a website and saves them to a CSV file.

        Parameters:
        - target_site (str): The URL of the website to fetch reviews from.
        - output_dir (str): The directory where the CSV file will be saved.
        - file_name (str): The name of the CSV file to save the reviews.
        - Use_Local_Sentiment_LLM (bool, optional): Indicates whether to use a local sentiment analysis model.
        - Use_Local_Scraper (bool, optional): Indicates whether to use a local web scraper.

        Returns:
        - None: Reviews are saved to a CSV file. If reviews cannot be fetched or saved, no return value is provided.
        """

        Use_Local_Scraper = Use_Local_Scraper if Use_Local_Scraper is not None else self.Use_Local_Scraper
        Use_Local_Sentiment_LLM = Use_Local_Sentiment_LLM if Use_Local_Sentiment_LLM is not None else self.Use_Local_Sentiment_LLM
        Fetched_reviews = self.get_sentiment_from_website_each_feedback_sentiment(
            target_website=target_site,
            Use_Local_Sentiment_LLM=True,
            Use_Local_Scraper=True,
            get_Groq_Review=False,
            get_OpenAI_review=False,
            get_localLLM_review=False
        )
        save_reviews_to_csv(reviews=Fetched_reviews,
                            output_dir=output_dir, file_name=file_name)

    def get_analysis_report_from_imdb(
        self,
        Product_Name,
        Reviews_Count,
        IMDB_API,
        Use_Local_Sentiment_LLM=True,
        Use_Local_Emotion_LLM=None,
        Use_Groq_API=None,
        Use_Gemini_API=None,
        Use_Local_API=None,
        Use_OpenAI_API=None,
        Groq_API="",
        OpenAi_API="",
        HuggingFace_API="",
        Google_API="",
        Get_Suggestions=False,
        Local_Sentiment_LLM="cardiffnlp/twitter-roberta-base-sentiment-latest",
        Local_Emotion_LLM="SamLowe/roberta-base-go_emotions",
        Groq_LLM_Temperature=0.1,
        Groq_LLM_Max_Tokens=100,
        Groq_LLM_Max_Input_Tokens=300,
        Groq_LLM_top_p=1,
        Groq_LLM_stream=False,
        Groq_LLM="llama3-8b-8192",
        Gemini_LLM_Temperature=0.1,
        Gemini_LLM_Max_Tokens=100,
        Gemini_LLM_Max_Input_Tokens=300,
        OpenAI_LLM="GPT-3.5",
        OpenAI_LLM_Temperature=0.1,
        OpenAI_LLM_Max_Tokens=100,
        OpenAI_LLM_stream=False,
        OpenAI_LLM_Max_Input_Tokens=300,
        Gemini_LLM="gemini-1.5-pro",
        Local_General_LLM="llama3.1",
        device_map="auto",
        Ollama_Model_EndPoint = "http://localhost:11434/api/generate" 
    ):
        Groq_LLM = Groq_LLM if Groq_LLM is not None else self.Groq_LLM
        OpenAI_LLM = OpenAI_LLM if OpenAI_LLM is not None else self.OpenAI_LLM
        Gemini_LLM = Gemini_LLM if Gemini_LLM is not None else self.Gemini_LLM
        Local_General_LLM = Local_General_LLM if Local_General_LLM is not None else self.Local_General_LLM
        Local_Sentiment_LLM = Local_Sentiment_LLM if Local_Sentiment_LLM is not None else self.Local_Sentiment_LLM
        Local_Emotion_LLM = Local_Emotion_LLM if Local_Emotion_LLM is not None else self.Local_Emotion_LLM
        Google_API = Google_API if Google_API is not None else self.Google_API
        Groq_API = Groq_API if Groq_API is not None else self.Groq_API
        OpenAi_API = OpenAi_API if OpenAi_API is not None else self.OpenAi_API
        HuggingFace_API = HuggingFace_API if HuggingFace_API is not None else self.HuggingFace_API
        device_map = device_map if device_map is not None else self.device_map

        scraper = IMDBReviewScraper(IMDB_API)
        imdb_id = scraper.get_imdb_id(Product_Name)
        if imdb_id:
            reviews = scraper.get_imdb_reviews(imdb_id, Reviews_Count)
        else:
            print("Movie not found.")
            exit()

        if Use_Local_Sentiment_LLM:
            final_resulted_output_sentiment = Struct_Generation_Pipeline(
                text_message=reviews,
                Use_Local_Sentiment_LLM=True,
                model_id=Local_Sentiment_LLM,
                device_map=device_map
            )
            if Get_Suggestions:
                    Suggestions_result_LocalLLM = Ollama_Local_Suggestions(
                        reviews=final_resulted_output_sentiment,
                        Model_Name=Local_General_LLM,
                        Ollama_Model_EndPoint=Ollama_Model_EndPoint
                    )
            if Use_Local_API:
                Summarized_result_LocalLLM = Ollama_Local_Summarize(
                    reviews=final_resulted_output_sentiment,
                    Model_Name=Local_General_LLM,
                    Ollama_Model_EndPoint=Ollama_Model_EndPoint
                )
                return Summarized_result_LocalLLM + "\n" + Suggestions_result_LocalLLM
            elif Use_Groq_API:
                Summarized_result_Groq = summarize_reviews(
                    reviews=final_resulted_output_sentiment,
                    KEY=Groq_API,
                    max_tokens=Groq_LLM_Max_Tokens,
                    temperature=Groq_LLM_Temperature,
                    top_p=Groq_LLM_top_p,
                    stream=Groq_LLM_stream,
                    model_id=Groq_LLM
                )
                return Summarized_result_Groq + "\n" + Suggestions_result_LocalLLM
            elif Use_Gemini_API:
                Summarized_result_Gemini = summarize_reviews_gemini(
                    reviews=final_resulted_output_sentiment,
                    KEY=Google_API,
                    max_tokens=Gemini_LLM_Max_Tokens,
                    temperature=Gemini_LLM_Temperature,
                    model=Gemini_LLM
                )
                return Summarized_result_Gemini + "\n" + Suggestions_result_LocalLLM
            elif Use_OpenAI_API:
                Summarized_result_OpenAi = summarize_reviews_openai(
                    reviews=final_resulted_output_sentiment,
                    KEY=OpenAi_API,
                    model_id=OpenAI_LLM,
                    max_tokens=OpenAI_LLM_Max_Tokens,
                    temperature=OpenAI_LLM_Temperature,
                    stream=OpenAI_LLM_stream
                )
                return Summarized_result_OpenAi + "\n" + Suggestions_result_LocalLLM
            else:
                return "Activate Any one Of the LLM"
        if Use_Local_Emotion_LLM:
            final_resulted_output_emotion = Struct_Emotion(
                text_message=reviews,
                Use_Local_Emotion_LLM=True,
                model_id=Local_Emotion_LLM,
                device_map=device_map
            )
            if Use_Local_API:
                Summarized_result_LocalLLM = Ollama_Local_Emotion_Summarize(
                    emotions=final_resulted_output_emotion,
                    Model_Name=Local_General_LLM,
                    Ollama_Model_EndPoint=Ollama_Model_EndPoint
                )
                return Summarized_result_LocalLLM
            elif Use_Groq_API:
                Summarized_result_Groq = summarize_Emotions_reviews(
                    emotions=final_resulted_output_emotion,
                    KEY=Groq_API,
                    max_tokens=Groq_LLM_Max_Tokens,
                    temperature=Groq_LLM_Temperature,
                    top_p=Groq_LLM_top_p,
                    stream=Groq_LLM_stream,
                    model_id=Groq_LLM
                )
                return Summarized_result_Groq
            elif Use_Gemini_API:
                Summarized_result_Gemini = summarize_Emotion_reviews_gemini(
                    emotions=final_resulted_output_emotion,
                    KEY=Google_API,
                    max_tokens=Gemini_LLM_Max_Tokens,
                    temperature=Gemini_LLM_Temperature,
                    model=Gemini_LLM
                )
                return Summarized_result_Gemini
            elif Use_OpenAI_API:
                Summarized_result_OpenAi = summarize_reviews_openai(
                    reviews=final_resulted_output_emotion,
                    KEY=OpenAi_API,
                    model_id=OpenAI_LLM,
                    max_tokens=OpenAI_LLM_Max_Tokens,
                    temperature=OpenAI_LLM_Temperature,
                    stream=OpenAI_LLM_stream
                )
                return Summarized_result_OpenAi
            else:
                return "Activate Any one Of the LLM"
        if Use_Local_Sentiment_LLM == False and Use_Local_Sentiment_LLM == None and Use_Local_Emotion_LLM == False and Use_Local_Emotion_LLM == None:
            return "Either Enable Sentiment LLM or Emotion LLM to Perform Analysis" 
    
    def get_analysis_report_from_LetterBoxD(
        self,
        Product_Name,
        Reviews_Count,
        Use_Local_Sentiment_LLM=True,
        Use_Local_Emotion_LLM=None,
        Use_Groq_API=None,
        Use_Gemini_API=None,
        Use_Local_API=None,
        Use_OpenAI_API=None,
        Groq_API="",
        OpenAi_API="",
        HuggingFace_API="",
        Google_API="",
        Get_Suggestions=False,
        Local_Sentiment_LLM="cardiffnlp/twitter-roberta-base-sentiment-latest",
        Local_Emotion_LLM="SamLowe/roberta-base-go_emotions",
        Groq_LLM_Temperature=0.1,
        Groq_LLM_Max_Tokens=100,
        Groq_LLM_Max_Input_Tokens=300,
        Groq_LLM_top_p=1,
        Groq_LLM_stream=False,
        Groq_LLM="llama3-8b-8192",
        Gemini_LLM_Temperature=0.1,
        Gemini_LLM_Max_Tokens=100,
        Gemini_LLM_Max_Input_Tokens=300,
        OpenAI_LLM="GPT-3.5",
        OpenAI_LLM_Temperature=0.1,
        OpenAI_LLM_Max_Tokens=100,
        OpenAI_LLM_stream=False,
        OpenAI_LLM_Max_Input_Tokens=300,
        Gemini_LLM="gemini-1.5-pro",
        Local_General_LLM="llama3.1",
        device_map="auto",
        Ollama_Model_EndPoint = "http://localhost:11434/api/generate" 
    ):
        Groq_LLM = Groq_LLM if Groq_LLM is not None else self.Groq_LLM
        OpenAI_LLM = OpenAI_LLM if OpenAI_LLM is not None else self.OpenAI_LLM
        Gemini_LLM = Gemini_LLM if Gemini_LLM is not None else self.Gemini_LLM
        Local_General_LLM = Local_General_LLM if Local_General_LLM is not None else self.Local_General_LLM
        Local_Sentiment_LLM = Local_Sentiment_LLM if Local_Sentiment_LLM is not None else self.Local_Sentiment_LLM
        Local_Emotion_LLM = Local_Emotion_LLM if Local_Emotion_LLM is not None else self.Local_Emotion_LLM
        Google_API = Google_API if Google_API is not None else self.Google_API
        Groq_API = Groq_API if Groq_API is not None else self.Groq_API
        OpenAi_API = OpenAi_API if OpenAi_API is not None else self.OpenAi_API
        HuggingFace_API = HuggingFace_API if HuggingFace_API is not None else self.HuggingFace_API
        device_map = device_map if device_map is not None else self.device_map

        scraper = LetterboxdReviewScraper()
        reviews = scraper.get_letterboxd_reviews(Product_Name, Reviews_Count)
        if reviews:
            if Use_Local_Sentiment_LLM:
                final_resulted_output_sentiment = Struct_Generation_Pipeline(
                    text_message=reviews,
                    Use_Local_Sentiment_LLM=True,
                    model_id=Local_Sentiment_LLM,
                    device_map=device_map
                )
                if Get_Suggestions:
                    Suggestions_result_LocalLLM = Ollama_Local_Suggestions(
                        reviews=final_resulted_output_sentiment,
                        Model_Name=Local_General_LLM,
                        Ollama_Model_EndPoint=Ollama_Model_EndPoint
                    )
                if Use_Local_API:
                    Summarized_result_LocalLLM = Ollama_Local_Summarize(
                        reviews=final_resulted_output_sentiment,
                        Model_Name=Local_General_LLM,
                        Ollama_Model_EndPoint=Ollama_Model_EndPoint
                    )
                    return Summarized_result_LocalLLM + "\n" + Suggestions_result_LocalLLM
                elif Use_Groq_API:
                    Summarized_result_Groq = summarize_reviews(
                        reviews=final_resulted_output_sentiment,
                        KEY=Groq_API,
                        max_tokens=Groq_LLM_Max_Tokens,
                        temperature=Groq_LLM_Temperature,
                        top_p=Groq_LLM_top_p,
                        stream=Groq_LLM_stream,
                        model_id=Groq_LLM
                    )
                    return Summarized_result_Groq + "\n" + Suggestions_result_LocalLLM
                elif Use_Gemini_API:
                    Summarized_result_Gemini = summarize_reviews_gemini(
                        reviews=final_resulted_output_sentiment,
                        KEY=Google_API,
                        max_tokens=Gemini_LLM_Max_Tokens,
                        temperature=Gemini_LLM_Temperature,
                        model=Gemini_LLM
                    )
                    return Summarized_result_Gemini + "\n" + Suggestions_result_LocalLLM
                elif Use_OpenAI_API:
                    Summarized_result_OpenAi = summarize_reviews_openai(
                        reviews=final_resulted_output_sentiment,
                        KEY=OpenAi_API,
                        model_id=OpenAI_LLM,
                        max_tokens=OpenAI_LLM_Max_Tokens,
                        temperature=OpenAI_LLM_Temperature,
                        stream=OpenAI_LLM_stream
                    )
                    return Summarized_result_OpenAi + "\n" + Suggestions_result_LocalLLM
                else:
                    return "Activate Any one Of the LLM"
            if Use_Local_Emotion_LLM:
                final_resulted_output_emotion = Struct_Emotion(
                    text_message=reviews,
                    Use_Local_Emotion_LLM=True,
                    model_id=Local_Emotion_LLM,
                    device_map=device_map
                )
                if Use_Local_API:
                    Summarized_result_LocalLLM = Ollama_Local_Emotion_Summarize(
                        emotions=final_resulted_output_emotion,
                        Model_Name=Local_General_LLM,
                        Ollama_Model_EndPoint=Ollama_Model_EndPoint
                    )
                    return Summarized_result_LocalLLM
                elif Use_Groq_API:
                    Summarized_result_Groq = summarize_Emotions_reviews(
                        emotions=final_resulted_output_emotion,
                        KEY=Groq_API,
                        max_tokens=Groq_LLM_Max_Tokens,
                        temperature=Groq_LLM_Temperature,
                        top_p=Groq_LLM_top_p,
                        stream=Groq_LLM_stream,
                        model_id=Groq_LLM
                    )
                    return Summarized_result_Groq
                elif Use_Gemini_API:
                    Summarized_result_Gemini = summarize_Emotion_reviews_gemini(
                        emotions=final_resulted_output_emotion,
                        KEY=Google_API,
                        max_tokens=Gemini_LLM_Max_Tokens,
                        temperature=Gemini_LLM_Temperature,
                        model=Gemini_LLM
                    )
                    return Summarized_result_Gemini
                elif Use_OpenAI_API:
                    Summarized_result_OpenAi = summarize_reviews_openai(
                        reviews=final_resulted_output_emotion,
                        KEY=OpenAi_API,
                        model_id=OpenAI_LLM,
                        max_tokens=OpenAI_LLM_Max_Tokens,
                        temperature=OpenAI_LLM_Temperature,
                        stream=OpenAI_LLM_stream
                    )
                    return Summarized_result_OpenAi
                else:
                    return "Activate Any one Of the LLM"
            if Use_Local_Sentiment_LLM == False and Use_Local_Sentiment_LLM == None and Use_Local_Emotion_LLM == False and Use_Local_Emotion_LLM == None:
                return "Either Enable Sentiment LLM or Emotion LLM to Perform Analysis" 
        else:
            print("Movie not found.")
            exit()

    def get_analysis_report_from_metacritic(
        self,
        Product_Name,
        Reviews_Count,
        Use_Local_Sentiment_LLM=True,
        Use_Local_Emotion_LLM=None,
        Use_Groq_API=None,
        Use_Gemini_API=None,
        Use_Local_API=None,
        Use_OpenAI_API=None,
        Groq_API="",
        OpenAi_API="",
        HuggingFace_API="",
        Google_API="",
        Get_Suggestions=False,
        Local_Sentiment_LLM="cardiffnlp/twitter-roberta-base-sentiment-latest",
        Local_Emotion_LLM="SamLowe/roberta-base-go_emotions",
        platforms = ["playstation-5", "pc", "playstation-4"],
        Groq_LLM_Temperature=0.1,
        Groq_LLM_Max_Tokens=100,
        Groq_LLM_Max_Input_Tokens=300,
        Groq_LLM_top_p=1,
        Groq_LLM_stream=False,
        Groq_LLM="llama3-8b-8192",
        Gemini_LLM_Temperature=0.1,
        Gemini_LLM_Max_Tokens=100,
        Gemini_LLM_Max_Input_Tokens=300,
        OpenAI_LLM="GPT-3.5",
        OpenAI_LLM_Temperature=0.1,
        OpenAI_LLM_Max_Tokens=100,
        OpenAI_LLM_stream=False,
        OpenAI_LLM_Max_Input_Tokens=300,
        Gemini_LLM="gemini-1.5-pro",
        Local_General_LLM="llama3.1",
        device_map="auto",
        Ollama_Model_EndPoint = "http://localhost:11434/api/generate" 
    ):
        Groq_LLM = Groq_LLM if Groq_LLM is not None else self.Groq_LLM
        OpenAI_LLM = OpenAI_LLM if OpenAI_LLM is not None else self.OpenAI_LLM
        Gemini_LLM = Gemini_LLM if Gemini_LLM is not None else self.Gemini_LLM
        Local_General_LLM = Local_General_LLM if Local_General_LLM is not None else self.Local_General_LLM
        Local_Sentiment_LLM = Local_Sentiment_LLM if Local_Sentiment_LLM is not None else self.Local_Sentiment_LLM
        Local_Emotion_LLM = Local_Emotion_LLM if Local_Emotion_LLM is not None else self.Local_Emotion_LLM
        Google_API = Google_API if Google_API is not None else self.Google_API
        Groq_API = Groq_API if Groq_API is not None else self.Groq_API
        OpenAi_API = OpenAi_API if OpenAi_API is not None else self.OpenAi_API
        HuggingFace_API = HuggingFace_API if HuggingFace_API is not None else self.HuggingFace_API
        device_map = device_map if device_map is not None else self.device_map

        scraper = MetacriticReviewScraper()
        reviews = scraper.fetch_reviews(Product_Name, platforms, Reviews_Count)
        if reviews:
            if Use_Local_Sentiment_LLM:
                final_resulted_output_sentiment = Struct_Generation_Pipeline(
                    text_message=reviews,
                    Use_Local_Sentiment_LLM=True,
                    model_id=Local_Sentiment_LLM,
                    device_map=device_map
                )
                if Get_Suggestions:
                    Suggestions_result_LocalLLM = Ollama_Local_Suggestions(
                        reviews=final_resulted_output_sentiment,
                        Model_Name=Local_General_LLM,
                        Ollama_Model_EndPoint=Ollama_Model_EndPoint
                    )
                if Use_Local_API:
                    Summarized_result_LocalLLM = Ollama_Local_Summarize(
                        reviews=final_resulted_output_sentiment,
                        Model_Name=Local_General_LLM,
                        Ollama_Model_EndPoint=Ollama_Model_EndPoint
                    )
                    return Summarized_result_LocalLLM + "\n" + Suggestions_result_LocalLLM
                elif Use_Groq_API:
                    Summarized_result_Groq = summarize_reviews(
                        reviews=final_resulted_output_sentiment,
                        KEY=Groq_API,
                        max_tokens=Groq_LLM_Max_Tokens,
                        temperature=Groq_LLM_Temperature,
                        top_p=Groq_LLM_top_p,
                        stream=Groq_LLM_stream,
                        model_id=Groq_LLM
                    )
                    return Summarized_result_Groq + "\n" + Suggestions_result_LocalLLM
                elif Use_Gemini_API:
                    Summarized_result_Gemini = summarize_reviews_gemini(
                        reviews=final_resulted_output_sentiment,
                        KEY=Google_API,
                        max_tokens=Gemini_LLM_Max_Tokens,
                        temperature=Gemini_LLM_Temperature,
                        model=Gemini_LLM
                    )
                    return Summarized_result_Gemini + "\n" + Suggestions_result_LocalLLM
                elif Use_OpenAI_API:
                    Summarized_result_OpenAi = summarize_reviews_openai(
                        reviews=final_resulted_output_sentiment,
                        KEY=OpenAi_API,
                        model_id=OpenAI_LLM,
                        max_tokens=OpenAI_LLM_Max_Tokens,
                        temperature=OpenAI_LLM_Temperature,
                        stream=OpenAI_LLM_stream
                    )
                    return Summarized_result_OpenAi + "\n" + Suggestions_result_LocalLLM
                else:
                    return "Activate Any one Of the LLM"
            if Use_Local_Emotion_LLM:
                final_resulted_output_emotion = Struct_Emotion(
                    text_message=reviews,
                    Use_Local_Emotion_LLM=True,
                    model_id=Local_Emotion_LLM,
                    device_map=device_map
                )
                if Use_Local_API:
                    Summarized_result_LocalLLM = Ollama_Local_Emotion_Summarize(
                        emotions=final_resulted_output_emotion,
                        Model_Name=Local_General_LLM,
                        Ollama_Model_EndPoint=Ollama_Model_EndPoint
                    )
                    return Summarized_result_LocalLLM
                elif Use_Groq_API:
                    Summarized_result_Groq = summarize_Emotions_reviews(
                        emotions=final_resulted_output_emotion,
                        KEY=Groq_API,
                        max_tokens=Groq_LLM_Max_Tokens,
                        temperature=Groq_LLM_Temperature,
                        top_p=Groq_LLM_top_p,
                        stream=Groq_LLM_stream,
                        model_id=Groq_LLM
                    )
                    return Summarized_result_Groq
                elif Use_Gemini_API:
                    Summarized_result_Gemini = summarize_Emotion_reviews_gemini(
                        emotions=final_resulted_output_emotion,
                        KEY=Google_API,
                        max_tokens=Gemini_LLM_Max_Tokens,
                        temperature=Gemini_LLM_Temperature,
                        model=Gemini_LLM
                    )
                    return Summarized_result_Gemini
                elif Use_OpenAI_API:
                    Summarized_result_OpenAi = summarize_reviews_openai(
                        reviews=final_resulted_output_emotion,
                        KEY=OpenAi_API,
                        model_id=OpenAI_LLM,
                        max_tokens=OpenAI_LLM_Max_Tokens,
                        temperature=OpenAI_LLM_Temperature,
                        stream=OpenAI_LLM_stream
                    )
                    return Summarized_result_OpenAi
                else:
                    return "Activate Any one Of the LLM"
            if Use_Local_Sentiment_LLM == False and Use_Local_Sentiment_LLM == None and Use_Local_Emotion_LLM == False and Use_Local_Emotion_LLM == None:
                return "Either Enable Sentiment LLM or Emotion LLM to Perform Analysis" 
        else:
            print("Movie not found.")
            exit()

    def get_analysis_report_from_reddit(
        self,
        Product_Name,
        Reviews_Count,
        Use_Local_Sentiment_LLM=True,
        Use_Local_Emotion_LLM=None,
        Use_Groq_API=None,
        Use_Gemini_API=None,
        Use_Local_API=None,
        Use_OpenAI_API=None,
        reddit_client_id='qzUTCldtSi4hU19EXseCFg',
        reddit_client_secret='shDZVPvWAH7lb5Oe2NP9tKIqCSTSkQ',
        reddit_user_agent='textextracter/1.0 by/u/DeepReference5190',
        Groq_API="",
        OpenAi_API="",
        HuggingFace_API="",
        Google_API="",
        Get_Suggestions=False,
        Local_Sentiment_LLM="cardiffnlp/twitter-roberta-base-sentiment-latest",
        Local_Emotion_LLM="SamLowe/roberta-base-go_emotions",
        Groq_LLM_Temperature=0.1,
        Groq_LLM_Max_Tokens=100,
        Groq_LLM_Max_Input_Tokens=300,
        Groq_LLM_top_p=1,
        Groq_LLM_stream=False,
        Groq_LLM="llama3-8b-8192",
        Gemini_LLM_Temperature=0.1,
        Gemini_LLM_Max_Tokens=100,
        Gemini_LLM_Max_Input_Tokens=300,
        OpenAI_LLM="GPT-3.5",
        OpenAI_LLM_Temperature=0.1,
        OpenAI_LLM_Max_Tokens=100,
        OpenAI_LLM_stream=False,
        OpenAI_LLM_Max_Input_Tokens=300,
        Gemini_LLM="gemini-1.5-pro",
        Local_General_LLM="llama3.1",
        device_map="auto",
        Ollama_Model_EndPoint = "http://localhost:11434/api/generate" 
    ):
        Groq_LLM = Groq_LLM if Groq_LLM is not None else self.Groq_LLM
        OpenAI_LLM = OpenAI_LLM if OpenAI_LLM is not None else self.OpenAI_LLM
        Gemini_LLM = Gemini_LLM if Gemini_LLM is not None else self.Gemini_LLM
        Local_General_LLM = Local_General_LLM if Local_General_LLM is not None else self.Local_General_LLM
        Local_Sentiment_LLM = Local_Sentiment_LLM if Local_Sentiment_LLM is not None else self.Local_Sentiment_LLM
        Local_Emotion_LLM = Local_Emotion_LLM if Local_Emotion_LLM is not None else self.Local_Emotion_LLM
        Google_API = Google_API if Google_API is not None else self.Google_API
        Groq_API = Groq_API if Groq_API is not None else self.Groq_API
        OpenAi_API = OpenAi_API if OpenAi_API is not None else self.OpenAi_API
        HuggingFace_API = HuggingFace_API if HuggingFace_API is not None else self.HuggingFace_API
        device_map = device_map if device_map is not None else self.device_map

        scraper = RedditScraper(reddit_client_id, reddit_client_secret, reddit_user_agent)
        reviews = scraper.search_reddit_for_product(Product_Name, Reviews_Count)
        if reviews:
            if Use_Local_Sentiment_LLM:
                final_resulted_output_sentiment = Struct_Generation_Pipeline(
                    text_message=reviews,
                    Use_Local_Sentiment_LLM=True,
                    model_id=Local_Sentiment_LLM,
                    device_map=device_map
                )
                if Get_Suggestions:
                    Suggestions_result_LocalLLM = Ollama_Local_Suggestions(
                        reviews=final_resulted_output_sentiment,
                        Model_Name=Local_General_LLM,
                        Ollama_Model_EndPoint=Ollama_Model_EndPoint
                    )
                if Use_Local_API:
                    Summarized_result_LocalLLM = Ollama_Local_Summarize(
                        reviews=final_resulted_output_sentiment,
                        Model_Name=Local_General_LLM,
                        Ollama_Model_EndPoint=Ollama_Model_EndPoint
                    )
                    return Summarized_result_LocalLLM + "\n" + Suggestions_result_LocalLLM
                elif Use_Groq_API:
                    Summarized_result_Groq = summarize_reviews(
                        reviews=final_resulted_output_sentiment,
                        KEY=Groq_API,
                        max_tokens=Groq_LLM_Max_Tokens,
                        temperature=Groq_LLM_Temperature,
                        top_p=Groq_LLM_top_p,
                        stream=Groq_LLM_stream,
                        model_id=Groq_LLM
                    )
                    return Summarized_result_Groq + "\n" + Suggestions_result_LocalLLM
                elif Use_Gemini_API:
                    Summarized_result_Gemini = summarize_reviews_gemini(
                        reviews=final_resulted_output_sentiment,
                        KEY=Google_API,
                        max_tokens=Gemini_LLM_Max_Tokens,
                        temperature=Gemini_LLM_Temperature,
                        model=Gemini_LLM
                    )
                    return Summarized_result_Gemini + "\n" + Suggestions_result_LocalLLM
                elif Use_OpenAI_API:
                    Summarized_result_OpenAi = summarize_reviews_openai(
                        reviews=final_resulted_output_sentiment,
                        KEY=OpenAi_API,
                        model_id=OpenAI_LLM,
                        max_tokens=OpenAI_LLM_Max_Tokens,
                        temperature=OpenAI_LLM_Temperature,
                        stream=OpenAI_LLM_stream
                    )
                    return Summarized_result_OpenAi + "\n" + Suggestions_result_LocalLLM
                else:
                    return "Activate Any one Of the LLM"
            if Use_Local_Emotion_LLM:
                final_resulted_output_emotion = Struct_Emotion(
                    text_message=reviews,
                    Use_Local_Emotion_LLM=True,
                    model_id=Local_Emotion_LLM,
                    device_map=device_map
                )
                if Use_Local_API:
                    Summarized_result_LocalLLM = Ollama_Local_Emotion_Summarize(
                        emotions=final_resulted_output_emotion,
                        Model_Name=Local_General_LLM,
                        Ollama_Model_EndPoint=Ollama_Model_EndPoint
                    )
                    return Summarized_result_LocalLLM
                elif Use_Groq_API:
                    Summarized_result_Groq = summarize_Emotions_reviews(
                        emotions=final_resulted_output_emotion,
                        KEY=Groq_API,
                        max_tokens=Groq_LLM_Max_Tokens,
                        temperature=Groq_LLM_Temperature,
                        top_p=Groq_LLM_top_p,
                        stream=Groq_LLM_stream,
                        model_id=Groq_LLM
                    )
                    return Summarized_result_Groq
                elif Use_Gemini_API:
                    Summarized_result_Gemini = summarize_Emotion_reviews_gemini(
                        emotions=final_resulted_output_emotion,
                        KEY=Google_API,
                        max_tokens=Gemini_LLM_Max_Tokens,
                        temperature=Gemini_LLM_Temperature,
                        model=Gemini_LLM
                    )
                    return Summarized_result_Gemini
                elif Use_OpenAI_API:
                    Summarized_result_OpenAi = summarize_reviews_openai(
                        reviews=final_resulted_output_emotion,
                        KEY=OpenAi_API,
                        model_id=OpenAI_LLM,
                        max_tokens=OpenAI_LLM_Max_Tokens,
                        temperature=OpenAI_LLM_Temperature,
                        stream=OpenAI_LLM_stream
                    )
                    return Summarized_result_OpenAi
                else:
                    return "Activate Any one Of the LLM"
            if Use_Local_Sentiment_LLM == False and Use_Local_Sentiment_LLM == None and Use_Local_Emotion_LLM == False and Use_Local_Emotion_LLM == None:
                return "Either Enable Sentiment LLM or Emotion LLM to Perform Analysis" 
        else:
            print("Movie not found.")
            exit()

    def get_analysis_report_from_rottentomatoes(
        self,
        Product_Name,
        Reviews_Count,
        Use_Local_Sentiment_LLM=True,
        Use_Local_Emotion_LLM=None,
        Use_Groq_API=None,
        Use_Gemini_API=None,
        Use_Local_API=None,
        Use_OpenAI_API=None,
        Groq_API="",
        OpenAi_API="",
        HuggingFace_API="",
        Google_API="",
        Get_Suggestions=False,
        Local_Sentiment_LLM="cardiffnlp/twitter-roberta-base-sentiment-latest",
        Local_Emotion_LLM="SamLowe/roberta-base-go_emotions",
        Groq_LLM_Temperature=0.1,
        Groq_LLM_Max_Tokens=100,
        Groq_LLM_Max_Input_Tokens=300,
        Groq_LLM_top_p=1,
        Groq_LLM_stream=False,
        Groq_LLM="llama3-8b-8192",
        Gemini_LLM_Temperature=0.1,
        Gemini_LLM_Max_Tokens=100,
        Gemini_LLM_Max_Input_Tokens=300,
        OpenAI_LLM="GPT-3.5",
        OpenAI_LLM_Temperature=0.1,
        OpenAI_LLM_Max_Tokens=100,
        OpenAI_LLM_stream=False,
        OpenAI_LLM_Max_Input_Tokens=300,
        Gemini_LLM="gemini-1.5-pro",
        Local_General_LLM="llama3.1",
        device_map="auto",
        Ollama_Model_EndPoint = "http://localhost:11434/api/generate" 
    ):
        Groq_LLM = Groq_LLM if Groq_LLM is not None else self.Groq_LLM
        OpenAI_LLM = OpenAI_LLM if OpenAI_LLM is not None else self.OpenAI_LLM
        Gemini_LLM = Gemini_LLM if Gemini_LLM is not None else self.Gemini_LLM
        Local_General_LLM = Local_General_LLM if Local_General_LLM is not None else self.Local_General_LLM
        Local_Sentiment_LLM = Local_Sentiment_LLM if Local_Sentiment_LLM is not None else self.Local_Sentiment_LLM
        Local_Emotion_LLM = Local_Emotion_LLM if Local_Emotion_LLM is not None else self.Local_Emotion_LLM
        Google_API = Google_API if Google_API is not None else self.Google_API
        Groq_API = Groq_API if Groq_API is not None else self.Groq_API
        OpenAi_API = OpenAi_API if OpenAi_API is not None else self.OpenAi_API
        HuggingFace_API = HuggingFace_API if HuggingFace_API is not None else self.HuggingFace_API
        device_map = device_map if device_map is not None else self.device_map

        scraper = RottenTomatoesReviewScraper()
        reviews = scraper.get_rotten_tomatoes_reviews(Product_Name, Reviews_Count)
        if reviews:
            if Use_Local_Sentiment_LLM:
                final_resulted_output_sentiment = Struct_Generation_Pipeline(
                    text_message=reviews,
                    Use_Local_Sentiment_LLM=True,
                    model_id=Local_Sentiment_LLM,
                    device_map=device_map
                )
                if Get_Suggestions:
                    Suggestions_result_LocalLLM = Ollama_Local_Suggestions(
                        reviews=final_resulted_output_sentiment,
                        Model_Name=Local_General_LLM,
                        Ollama_Model_EndPoint=Ollama_Model_EndPoint
                    )
                if Use_Local_API:
                    Summarized_result_LocalLLM = Ollama_Local_Summarize(
                        reviews=final_resulted_output_sentiment,
                        Model_Name=Local_General_LLM,
                        Ollama_Model_EndPoint=Ollama_Model_EndPoint
                    )
                    return Summarized_result_LocalLLM + "\n" + Suggestions_result_LocalLLM
                elif Use_Groq_API:
                    Summarized_result_Groq = summarize_reviews(
                        reviews=final_resulted_output_sentiment,
                        KEY=Groq_API,
                        max_tokens=Groq_LLM_Max_Tokens,
                        temperature=Groq_LLM_Temperature,
                        top_p=Groq_LLM_top_p,
                        stream=Groq_LLM_stream,
                        model_id=Groq_LLM
                    )
                    return Summarized_result_Groq + "\n" + Suggestions_result_LocalLLM
                elif Use_Gemini_API:
                    Summarized_result_Gemini = summarize_reviews_gemini(
                        reviews=final_resulted_output_sentiment,
                        KEY=Google_API,
                        max_tokens=Gemini_LLM_Max_Tokens,
                        temperature=Gemini_LLM_Temperature,
                        model=Gemini_LLM
                    )
                    return Summarized_result_Gemini + "\n" + Suggestions_result_LocalLLM
                elif Use_OpenAI_API:
                    Summarized_result_OpenAi = summarize_reviews_openai(
                        reviews=final_resulted_output_sentiment,
                        KEY=OpenAi_API,
                        model_id=OpenAI_LLM,
                        max_tokens=OpenAI_LLM_Max_Tokens,
                        temperature=OpenAI_LLM_Temperature,
                        stream=OpenAI_LLM_stream
                    )
                    return Summarized_result_OpenAi + "\n" + Suggestions_result_LocalLLM
                else:
                    return "Activate Any one Of the LLM"
            if Use_Local_Emotion_LLM:
                final_resulted_output_emotion = Struct_Emotion(
                    text_message=reviews,
                    Use_Local_Emotion_LLM=True,
                    model_id=Local_Emotion_LLM,
                    device_map=device_map
                )
                if Use_Local_API:
                    Summarized_result_LocalLLM = Ollama_Local_Emotion_Summarize(
                        emotions=final_resulted_output_emotion,
                        Model_Name=Local_General_LLM,
                        Ollama_Model_EndPoint=Ollama_Model_EndPoint
                    )
                    return Summarized_result_LocalLLM
                elif Use_Groq_API:
                    Summarized_result_Groq = summarize_Emotions_reviews(
                        emotions=final_resulted_output_emotion,
                        KEY=Groq_API,
                        max_tokens=Groq_LLM_Max_Tokens,
                        temperature=Groq_LLM_Temperature,
                        top_p=Groq_LLM_top_p,
                        stream=Groq_LLM_stream,
                        model_id=Groq_LLM
                    )
                    return Summarized_result_Groq
                elif Use_Gemini_API:
                    Summarized_result_Gemini = summarize_Emotion_reviews_gemini(
                        emotions=final_resulted_output_emotion,
                        KEY=Google_API,
                        max_tokens=Gemini_LLM_Max_Tokens,
                        temperature=Gemini_LLM_Temperature,
                        model=Gemini_LLM
                    )
                    return Summarized_result_Gemini
                elif Use_OpenAI_API:
                    Summarized_result_OpenAi = summarize_reviews_openai(
                        reviews=final_resulted_output_emotion,
                        KEY=OpenAi_API,
                        model_id=OpenAI_LLM,
                        max_tokens=OpenAI_LLM_Max_Tokens,
                        temperature=OpenAI_LLM_Temperature,
                        stream=OpenAI_LLM_stream
                    )
                    return Summarized_result_OpenAi
                else:
                    return "Activate Any one Of the LLM"
            if Use_Local_Sentiment_LLM == False and Use_Local_Sentiment_LLM == None and Use_Local_Emotion_LLM == False and Use_Local_Emotion_LLM == None:
                return "Either Enable Sentiment LLM or Emotion LLM to Perform Analysis" 
        else:
            print("Movie not found.")
            exit()

    def get_analysis_report_from_steam(
        self,
        Product_Name,
        Reviews_Count,
        Use_Local_Sentiment_LLM=True,
        Use_Local_Emotion_LLM=None,
        Use_Groq_API=None,
        Use_Gemini_API=None,
        Use_Local_API=None,
        Use_OpenAI_API=None,
        Get_Suggestions=False,
        Groq_API="",
        OpenAi_API="",
        HuggingFace_API="",
        Google_API="",
        Local_Sentiment_LLM="cardiffnlp/twitter-roberta-base-sentiment-latest",
        Local_Emotion_LLM="SamLowe/roberta-base-go_emotions",
        platforms = ["playstation-5", "pc", "playstation-4"],
        Groq_LLM_Temperature=0.1,
        Groq_LLM_Max_Tokens=100,
        Groq_LLM_Max_Input_Tokens=300,
        Groq_LLM_top_p=1,
        Groq_LLM_stream=False,
        Groq_LLM="llama3-8b-8192",
        Gemini_LLM_Temperature=0.1,
        Gemini_LLM_Max_Tokens=100,
        Gemini_LLM_Max_Input_Tokens=300,
        OpenAI_LLM="GPT-3.5",
        OpenAI_LLM_Temperature=0.1,
        OpenAI_LLM_Max_Tokens=100,
        OpenAI_LLM_stream=False,
        OpenAI_LLM_Max_Input_Tokens=300,
        Gemini_LLM="gemini-1.5-pro",
        Local_General_LLM="llama3.1",
        device_map="auto",
        Ollama_Model_EndPoint = "http://localhost:11434/api/generate" 
    ):
        Groq_LLM = Groq_LLM if Groq_LLM is not None else self.Groq_LLM
        OpenAI_LLM = OpenAI_LLM if OpenAI_LLM is not None else self.OpenAI_LLM
        Gemini_LLM = Gemini_LLM if Gemini_LLM is not None else self.Gemini_LLM
        Local_General_LLM = Local_General_LLM if Local_General_LLM is not None else self.Local_General_LLM
        Local_Sentiment_LLM = Local_Sentiment_LLM if Local_Sentiment_LLM is not None else self.Local_Sentiment_LLM
        Local_Emotion_LLM = Local_Emotion_LLM if Local_Emotion_LLM is not None else self.Local_Emotion_LLM
        Google_API = Google_API if Google_API is not None else self.Google_API
        Groq_API = Groq_API if Groq_API is not None else self.Groq_API
        OpenAi_API = OpenAi_API if OpenAi_API is not None else self.OpenAi_API
        HuggingFace_API = HuggingFace_API if HuggingFace_API is not None else self.HuggingFace_API
        device_map = device_map if device_map is not None else self.device_map

        scraper = SteamReviewScraper()
        reviews = scraper.fetch_reviews_for_game(Product_Name, Reviews_Count)
        if reviews:
            if Use_Local_Sentiment_LLM:
                final_resulted_output_sentiment = Struct_Generation_Pipeline(
                    text_message=reviews,
                    Use_Local_Sentiment_LLM=True,
                    model_id=Local_Sentiment_LLM,
                    device_map=device_map
                )
                if Get_Suggestions:
                    Suggestions_result_LocalLLM = Ollama_Local_Suggestions(
                        reviews=final_resulted_output_sentiment,
                        Model_Name=Local_General_LLM,
                        Ollama_Model_EndPoint=Ollama_Model_EndPoint
                    )
                if Use_Local_API:
                    Summarized_result_LocalLLM = Ollama_Local_Summarize(
                        reviews=final_resulted_output_sentiment,
                        Model_Name=Local_General_LLM,
                        Ollama_Model_EndPoint=Ollama_Model_EndPoint
                    )
                    return Summarized_result_LocalLLM + "\n" + Suggestions_result_LocalLLM
                elif Use_Groq_API:
                    Summarized_result_Groq = summarize_reviews(
                        reviews=final_resulted_output_sentiment,
                        KEY=Groq_API,
                        max_tokens=Groq_LLM_Max_Tokens,
                        temperature=Groq_LLM_Temperature,
                        top_p=Groq_LLM_top_p,
                        stream=Groq_LLM_stream,
                        model_id=Groq_LLM
                    )
                    return Summarized_result_Groq + "\n" + Suggestions_result_LocalLLM
                elif Use_Gemini_API:
                    Summarized_result_Gemini = summarize_reviews_gemini(
                        reviews=final_resulted_output_sentiment,
                        KEY=Google_API,
                        max_tokens=Gemini_LLM_Max_Tokens,
                        temperature=Gemini_LLM_Temperature,
                        model=Gemini_LLM
                    )
                    return Summarized_result_Gemini + "\n" + Suggestions_result_LocalLLM
                elif Use_OpenAI_API:
                    Summarized_result_OpenAi = summarize_reviews_openai(
                        reviews=final_resulted_output_sentiment,
                        KEY=OpenAi_API,
                        model_id=OpenAI_LLM,
                        max_tokens=OpenAI_LLM_Max_Tokens,
                        temperature=OpenAI_LLM_Temperature,
                        stream=OpenAI_LLM_stream
                    )
                    return Summarized_result_OpenAi + "\n" + Suggestions_result_LocalLLM
                else:
                    return "Activate Any one Of the LLM"
            if Use_Local_Emotion_LLM:
                final_resulted_output_emotion = Struct_Emotion(
                    text_message=reviews,
                    Use_Local_Emotion_LLM=True,
                    model_id=Local_Emotion_LLM,
                    device_map=device_map
                )
                if Use_Local_API:
                    Summarized_result_LocalLLM = Ollama_Local_Emotion_Summarize(
                        emotions=final_resulted_output_emotion,
                        Model_Name=Local_General_LLM,
                        Ollama_Model_EndPoint=Ollama_Model_EndPoint
                    )
                    return Summarized_result_LocalLLM
                elif Use_Groq_API:
                    Summarized_result_Groq = summarize_Emotions_reviews(
                        emotions=final_resulted_output_emotion,
                        KEY=Groq_API,
                        max_tokens=Groq_LLM_Max_Tokens,
                        temperature=Groq_LLM_Temperature,
                        top_p=Groq_LLM_top_p,
                        stream=Groq_LLM_stream,
                        model_id=Groq_LLM
                    )
                    return Summarized_result_Groq
                elif Use_Gemini_API:
                    Summarized_result_Gemini = summarize_Emotion_reviews_gemini(
                        emotions=final_resulted_output_emotion,
                        KEY=Google_API,
                        max_tokens=Gemini_LLM_Max_Tokens,
                        temperature=Gemini_LLM_Temperature,
                        model=Gemini_LLM
                    )
                    return Summarized_result_Gemini
                elif Use_OpenAI_API:
                    Summarized_result_OpenAi = summarize_reviews_openai(
                        reviews=final_resulted_output_emotion,
                        KEY=OpenAi_API,
                        model_id=OpenAI_LLM,
                        max_tokens=OpenAI_LLM_Max_Tokens,
                        temperature=OpenAI_LLM_Temperature,
                        stream=OpenAI_LLM_stream
                    )
                    return Summarized_result_OpenAi
                else:
                    return "Activate Any one Of the LLM"
            if Use_Local_Sentiment_LLM == False and Use_Local_Sentiment_LLM == None and Use_Local_Emotion_LLM == False and Use_Local_Emotion_LLM == None:
                return "Either Enable Sentiment LLM or Emotion LLM to Perform Analysis" 
        else:
            print("Movie not found.")
            exit()

    def get_analysis_report_from_youtube(
        self,
        Product_Name,
        Use_Local_Sentiment_LLM=True,
        Use_Local_Emotion_LLM=None,
        Use_Groq_API=None,
        Use_Gemini_API=None,
        Use_Local_API=None,
        Use_OpenAI_API=None,
        Get_Suggestions=False,
        Groq_API="",
        OpenAi_API="",
        HuggingFace_API="",
        Google_API="",
        Youtube_API="",
        Local_Sentiment_LLM="cardiffnlp/twitter-roberta-base-sentiment-latest",
        Local_Emotion_LLM="SamLowe/roberta-base-go_emotions",
        Groq_LLM_Temperature=0.1,
        Groq_LLM_Max_Tokens=100,
        Groq_LLM_Max_Input_Tokens=300,
        Groq_LLM_top_p=1,
        Groq_LLM_stream=False,
        Groq_LLM="llama3-8b-8192",
        Gemini_LLM_Temperature=0.1,
        Gemini_LLM_Max_Tokens=100,
        Gemini_LLM_Max_Input_Tokens=300,
        OpenAI_LLM="GPT-3.5",
        OpenAI_LLM_Temperature=0.1,
        OpenAI_LLM_Max_Tokens=100,
        OpenAI_LLM_stream=False,
        OpenAI_LLM_Max_Input_Tokens=300,
        Gemini_LLM="gemini-1.5-pro",
        Local_General_LLM="llama3.1",
        device_map="auto",
        Ollama_Model_EndPoint = "http://localhost:11434/api/generate" 
    ):
        Groq_LLM = Groq_LLM if Groq_LLM is not None else self.Groq_LLM
        OpenAI_LLM = OpenAI_LLM if OpenAI_LLM is not None else self.OpenAI_LLM
        Gemini_LLM = Gemini_LLM if Gemini_LLM is not None else self.Gemini_LLM
        Local_General_LLM = Local_General_LLM if Local_General_LLM is not None else self.Local_General_LLM
        Local_Sentiment_LLM = Local_Sentiment_LLM if Local_Sentiment_LLM is not None else self.Local_Sentiment_LLM
        Local_Emotion_LLM = Local_Emotion_LLM if Local_Emotion_LLM is not None else self.Local_Emotion_LLM
        Google_API = Google_API if Google_API is not None else self.Google_API
        Groq_API = Groq_API if Groq_API is not None else self.Groq_API
        OpenAi_API = OpenAi_API if OpenAi_API is not None else self.OpenAi_API
        HuggingFace_API = HuggingFace_API if HuggingFace_API is not None else self.HuggingFace_API
        device_map = device_map if device_map is not None else self.device_map

        fetcher = YouTubeDataFetcher(Youtube_API)
        reviews = fetcher.fetch_youtube_data_for_product(Product_Name, max_videos=10)
        if reviews:
            if Use_Local_Sentiment_LLM:
                final_resulted_output_sentiment = Struct_Generation_Pipeline(
                    text_message=reviews,
                    Use_Local_Sentiment_LLM=True,
                    model_id=Local_Sentiment_LLM,
                    device_map=device_map
                )
                if Get_Suggestions:
                    Suggestions_result_LocalLLM = Ollama_Local_Suggestions(
                        reviews=final_resulted_output_sentiment,
                        Model_Name=Local_General_LLM,
                        Ollama_Model_EndPoint=Ollama_Model_EndPoint
                    )
                if Use_Local_API:
                    Summarized_result_LocalLLM = Ollama_Local_Summarize(
                        reviews=final_resulted_output_sentiment,
                        Model_Name=Local_General_LLM,
                        Ollama_Model_EndPoint=Ollama_Model_EndPoint
                    )
                    return Summarized_result_LocalLLM + "\n" + Suggestions_result_LocalLLM
                elif Use_Groq_API:
                    Summarized_result_Groq = summarize_reviews(
                        reviews=final_resulted_output_sentiment,
                        KEY=Groq_API,
                        max_tokens=Groq_LLM_Max_Tokens,
                        temperature=Groq_LLM_Temperature,
                        top_p=Groq_LLM_top_p,
                        stream=Groq_LLM_stream,
                        model_id=Groq_LLM
                    )
                    return Summarized_result_Groq + "\n" + Suggestions_result_LocalLLM
                elif Use_Gemini_API:
                    Summarized_result_Gemini = summarize_reviews_gemini(
                        reviews=final_resulted_output_sentiment,
                        KEY=Google_API,
                        max_tokens=Gemini_LLM_Max_Tokens,
                        temperature=Gemini_LLM_Temperature,
                        model=Gemini_LLM
                    )
                    return Summarized_result_Gemini + "\n" + Suggestions_result_LocalLLM
                elif Use_OpenAI_API:
                    Summarized_result_OpenAi = summarize_reviews_openai(
                        reviews=final_resulted_output_sentiment,
                        KEY=OpenAi_API,
                        model_id=OpenAI_LLM,
                        max_tokens=OpenAI_LLM_Max_Tokens,
                        temperature=OpenAI_LLM_Temperature,
                        stream=OpenAI_LLM_stream
                    )
                    return Summarized_result_OpenAi + "\n" + Suggestions_result_LocalLLM
                else:
                    return "Activate Any one Of the LLM"
            if Use_Local_Emotion_LLM:
                final_resulted_output_emotion = Struct_Emotion(
                    text_message=reviews,
                    Use_Local_Emotion_LLM=True,
                    model_id=Local_Emotion_LLM,
                    device_map=device_map
                )
                if Use_Local_API:
                    Summarized_result_LocalLLM = Ollama_Local_Emotion_Summarize(
                        emotions=final_resulted_output_emotion,
                        Model_Name=Local_General_LLM,
                        Ollama_Model_EndPoint=Ollama_Model_EndPoint
                    )
                    return Summarized_result_LocalLLM
                elif Use_Groq_API:
                    Summarized_result_Groq = summarize_Emotions_reviews(
                        emotions=final_resulted_output_emotion,
                        KEY=Groq_API,
                        max_tokens=Groq_LLM_Max_Tokens,
                        temperature=Groq_LLM_Temperature,
                        top_p=Groq_LLM_top_p,
                        stream=Groq_LLM_stream,
                        model_id=Groq_LLM
                    )
                    return Summarized_result_Groq
                elif Use_Gemini_API:
                    Summarized_result_Gemini = summarize_Emotion_reviews_gemini(
                        emotions=final_resulted_output_emotion,
                        KEY=Google_API,
                        max_tokens=Gemini_LLM_Max_Tokens,
                        temperature=Gemini_LLM_Temperature,
                        model=Gemini_LLM
                    )
                    return Summarized_result_Gemini
                elif Use_OpenAI_API:
                    Summarized_result_OpenAi = summarize_reviews_openai(
                        reviews=final_resulted_output_emotion,
                        KEY=OpenAi_API,
                        model_id=OpenAI_LLM,
                        max_tokens=OpenAI_LLM_Max_Tokens,
                        temperature=OpenAI_LLM_Temperature,
                        stream=OpenAI_LLM_stream
                    )
                    return Summarized_result_OpenAi
                else:
                    return "Activate Any one Of the LLM"
            if Use_Local_Sentiment_LLM == False and Use_Local_Sentiment_LLM == None and Use_Local_Emotion_LLM == False and Use_Local_Emotion_LLM == None:
                return "Either Enable Sentiment LLM or Emotion LLM to Perform Analysis" 
        else:
            print("Movie not found.")
            exit()

    def get_suggestions_from_website(
            self,
            target_website,
            Use_Local_Sentiment_LLM=None,
            Use_Local_General_LLM=None,
            Use_Local_Scraper=None,
            Use_Scraper_API=None,
            Use_Groq_API=None,
            Use_Open_API=None,
            Use_Gemini_API=None,
            Local_Sentiment_LLM=None,
            Local_General_LLM="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            Local_General_LLM_kwargs={
                'temperature': 0.1,
                'top_p': 1
            },
            Groq_API=None,
            OpenAi_API=None,
            HuggingFace_API=None,
            Scraper_api_key=None,
            Google_API=None,
            Local_api_key=None,
            Groq_LLM="llama3-8b-8192",
            OpenAI_LLM="GPT-3.5",
            device_map="auto",
            Gemini_LLM="gemini-1.5-pro",
            Groq_LLM_Temperature=0.1,
            Groq_LLM_Max_Tokens=100,
            Groq_LLM_Max_Input_Tokens=300,
            Groq_LLM_top_p=1,
            Groq_LLM_stream=False,
            OpenAI_LLM_Temperature=0.1,
            OpenAI_LLM_Max_Tokens=100,
            OpenAI_LLM_stream=False,
            OpenAI_LLM_Max_Input_Tokens=300,
            Local_LLM_Max_Input_Tokens=300,
            Gemini_LLM_Temperature=0.1,
            Gemini_LLM_Max_Tokens=100,
            Gemini_LLM_Max_Input_Tokens=300,
            Ollama_Model="llama3.1",
            Ollama_Model_EndPoint = "http://localhost:11434/api/generate" 

    ):
        """
        Fetches reviews from a specified website and generates an overall sentiment summary.
        Uses various methods for sentiment analysis and summarization, depending on the available options and configurations.

        Parameters:
        - target_website: The website from which to fetch reviews.
        - Use_Local_Sentiment_LLM: Boolean indicating whether to use the local sentiment LLM for initial sentiment analysis.
        - Use_Local_General_LLM: Boolean indicating whether to use a local general LLM for summarization.
        - Use_Local_Scraper: Boolean indicating whether to use a local scraper to fetch reviews.
        - Use_Scraper_API: Boolean indicating whether to use a scraper API for fetching reviews.
        - Use_Groq_API: Boolean indicating whether to use the Groq API for summarization.
        - Use_Open_API: Boolean indicating whether to use the OpenAI API for summarization.
        - Local_Sentiment_LLM: Model identifier for the local sentiment LLM.
        - Local_General_LLM: Model identifier for the local general LLM used for summarization.
        - Local_General_LLM_kwargs: Additional arguments for the local general LLM.
        - Groq_API: API key for the Groq API.
        - OpenAi_API: API key for the OpenAI API.
        - HuggingFace_API: API key for Hugging Face.
        - Scraper_api_key: API key for the scraper service.
        - Local_api_key: API key for the local API.
        - Groq_LLM: Model identifier for the Groq LLM.
        - OpenAI_LLM: Model identifier for the OpenAI LLM.
        - device_map: Device to run the model on (e.g., "cpu" or "cuda").
        - Groq_LLM_Temperature: Temperature setting for the Groq LLM.
        - Groq_LLM_Max_Tokens: Maximum tokens for the Groq LLM.
        - Groq_LLM_Max_Input_Tokens: Maximum input tokens for the Groq LLM.
        - Groq_LLM_top_p: Top-p value for the Groq LLM.
        - Groq_LLM_stream: Boolean indicating whether to use streaming for Groq LLM.
        - OpenAI_LLM_Temperature: Temperature setting for the OpenAI LLM.
        - OpenAI_LLM_Max_Tokens: Maximum tokens for the OpenAI LLM.
        - OpenAI_LLM_stream: Boolean indicating whether to use streaming for OpenAI LLM.
        - OpenAI_LLM_Max_Input_Tokens: Maximum input tokens for the OpenAI LLM.
        - Local_LLM_Max_Input_Tokens: Maximum input tokens for the local LLM.

        Returns:
        - Summarized sentiment results based on the selected summarization method (Groq API, OpenAI API, or local general LLM).
        - Error message if no reviews were fetched or no LLM inference method is selected.
        """

        Use_Local_Scraper = Use_Local_Scraper if Use_Local_Scraper is not None else self.Use_Local_Scraper
        Use_Scraper_API = Use_Scraper_API if Use_Scraper_API is not None else self.Use_Scraper_API
        Scraper_api_key = Scraper_api_key if Scraper_api_key is not None else self.Scraper_api_key
        Local_api_key = Local_api_key if Local_api_key is not None else self.Local_api_key
        Local_Sentiment_LLM = Local_Sentiment_LLM if Local_Sentiment_LLM is not None else self.Local_Sentiment_LLM
        device_map = device_map if device_map is not None else self.device_map

        Groq_API = Groq_API if Groq_API is not None else self.Groq_API
        Use_Groq_API = Use_Groq_API if Use_Groq_API is not None else self.Use_Groq_API
        Groq_LLM = Groq_LLM if Groq_LLM is not None else self.Groq_LLM

        Google_API = Google_API if Google_API is not None else self.Google_API
        Use_Gemini_API = Use_Gemini_API if Use_Gemini_API is not None else self.Use_Gemini_API
        Gemini_LLM = Gemini_LLM if Gemini_LLM is not None else self.Gemini_LLM

        Use_Open_API = Use_Open_API if Use_Open_API is not None else self.Use_Open_API
        OpenAi_API = OpenAi_API if OpenAi_API is not None else self.OpenAi_API
        OpenAI_LLM = OpenAI_LLM if OpenAI_LLM is not None else self.OpenAI_LLM

        Use_Local_General_LLM = Use_Local_General_LLM if Use_Local_General_LLM is not None else self.Use_Local_General_LLM
        Local_General_LLM = Local_General_LLM if Local_General_LLM is not None else self.Local_General_LLM
        Local_General_LLM_kwargs = Local_General_LLM_kwargs if Local_General_LLM_kwargs is not None else self.Local_General_LLM_kwargs
        HuggingFace_API = HuggingFace_API if HuggingFace_API is not None else self.HuggingFace_API
        final_resulted_output = None

        
        if target_website:
            try:
                fetched_review = Fetch_Reviews(
                    target_website,
                    Use_Local_Scraper,
                    Use_Scraper_API,
                    Scraper_api_key,
                    Local_api_key
                )

                if fetched_review:
                    final_resulted_output = Struct_Generation_Pipeline(
                        text_message=fetched_review,
                        Use_Local_Sentiment_LLM=True,
                        model_id=Local_Sentiment_LLM,
                        device_map=device_map
                    )
                else:
                    raise ValueError("Couldn't Fetch the Reviews From the Site")

            except ValueError as e:
                print(f"Error: {e}")
                exit()

            except ConnectionError:
                print(
                    "Error: Unable to connect to the website. Please check your internet connection.")
                exit()

            except TimeoutError:
                print("Error: The request timed out. Try again later.")
                exit()

            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                exit()

            if Use_Groq_API == True or Use_Open_API == True or Use_Local_General_LLM == True or Use_Gemini_API == True:
                if Use_Groq_API:
                    if final_resulted_output:
                        Suggestions_result_Groq = suggest_reviews(
                            reviews=final_resulted_output,
                            KEY=Groq_API,
                            max_tokens=Groq_LLM_Max_Tokens,
                            temperature=Groq_LLM_Temperature,
                            top_p=Groq_LLM_top_p,
                            stream=Groq_LLM_stream,
                            model_id=Groq_LLM
                        )
                        return Suggestions_result_Groq
                    else:
                        print("Couldnt Return Formatted Output")
                        exit()
                elif Use_Gemini_API:
                    if final_resulted_output:
                        Suggestions_result_Gemini = suggest_reviews_gemini(
                            reviews=final_resulted_output,
                            KEY=Google_API,
                            max_tokens=Gemini_LLM_Max_Tokens,
                            temperature=Gemini_LLM_Temperature,
                            model=Gemini_LLM
                        )
                        return Suggestions_result_Gemini
                    else:
                        print("Couldnt Return Formatted Output")
                        exit()
                elif Use_Open_API:
                    if final_resulted_output:
                        Suggestions_result_OpenAi = summarize_reviews_openai(
                            reviews=final_resulted_output,
                            KEY=OpenAi_API,
                            model_id=OpenAI_LLM,
                            max_tokens=OpenAI_LLM_Max_Tokens,
                            temperature=OpenAI_LLM_Temperature,
                            stream=OpenAI_LLM_stream
                        )
                        return Suggestions_result_OpenAi
                    else:
                        print("Couldnt Return Formatted Output")
                        exit()
                elif Use_Local_General_LLM:
                    if final_resulted_output:
                        Suggestions_result_LocalLLM = Ollama_Local_Suggestions(
                            reviews=final_resulted_output,
                            Model_Name=Ollama_Model,
                            Ollama_Model_EndPoint=Ollama_Model_EndPoint
                        )
                        return Suggestions_result_LocalLLM
                    else:
                        print("Couldnt Return Formatted Output")
                else:
                    print("Error : use any one of the LLM inference")
        else:
            raise ValueError("Error : No URL supplied")
        
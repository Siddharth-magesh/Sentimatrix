from Sentimatrix.utils.Quick_Sentiment import Generation_Pipeline
from Sentimatrix.utils.web_scraper import Fetch_Reviews

class SentConfig:
    """
    Main Class that Contains the Following Sentiment Analysis Functions

    Constructor Function : Takes the initial Values

    Function_Name : get_Quick_sentiment

    Function_Name : get_sentiment_from_website_each_feedback_sentiment

    Function_Name : get_sentiment_from_website_overall_summary

    Function_Name : get_analytical_customer_sentiments

    Function_Name : get_Sentiment_Audio_file

    Function_Name : compare_product_on_reviews

    """
    def __init__(
            self,
            Use_Local_Sentiment_LLM = True,
            Use_Local_General_LLM = False,
            Use_Groq_API = False,
            Use_Open_API = False,
            Use_Local_Scraper = False,
            Use_Scraper_API = False,
            Local_Sentiment_LLM = "cardiffnlp/twitter-roberta-base-sentiment-latest",
            Local_General_LLM = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            Local_General_LLM_kwargs = {
                    'temperature':0.1,
                    'top_p':1
                } ,
            Groq_API = "",
            OpenAi_API = "",
            HuggingFace_API = "",
            Local_api_key = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            Scraper_api_key = None,
            Groq_LLM = "llama3-8b-8192",
            OpenAI_LLM = "GPT-3.5",
            device_map = "auto"
    ):
        self.Use_Local_Sentiment_LLM = Use_Local_Sentiment_LLM
        self.Use_Local_General_LLM = Use_Local_General_LLM
        self.Use_Groq_API = Use_Groq_API
        self.Use_Open_API = Use_Open_API
        self.Use_Local_Scraper = Use_Local_Scraper
        self.Use_Scraper_API = Use_Scraper_API
        self.Local_Sentiment_LLM = Local_Sentiment_LLM
        self.Local_General_LLM = Local_General_LLM
        self.Local_General_LLM_kwargs = Local_General_LLM_kwargs
        self.Groq_API = Groq_API
        self.OpenAi_API = OpenAi_API
        self.Local_api_key = Local_api_key
        self.Scraper_api_key = Scraper_api_key
        self.HuggingFace_API = HuggingFace_API
        self.Groq_LLM = Groq_LLM
        self.OpenAI_LLM = OpenAI_LLM
        self.device_map = device_map

        

    def get_Quick_sentiment(
            self,
            text_message,
            Use_Local_Sentiment_LLM = None,
            Local_Sentiment_LLM = None,
            device_map = None
    ):
        """
        Function Description : Gives a Sentiment analysis on Text Messages . Runs On local machine with HF LLM.

        input params : 
        text_message -> Str or List of Str containing Text messages
        Use_Local_Sentiment_LLM : Bool , must be Set to True
        Local_Sentiment_LLM : Sentiment LLM
        device_map : "cuda" or "cpu" or "cuda"

        return : 
        {'label': str, 'score': float}
        """
        Use_Local_Sentiment_LLM = Use_Local_Sentiment_LLM if Use_Local_Sentiment_LLM is not None else self.Use_Local_Sentiment_LLM
        Local_Sentiment_LLM = Local_Sentiment_LLM if Local_Sentiment_LLM is not None else self.Local_Sentiment_LLM
        device_map = device_map if device_map is not None else self.device_map
        final_result = Generation_Pipeline(
            text_message=text_message,
            Use_Local_Sentiment_LLM = Use_Local_Sentiment_LLM,
            model_id = Local_Sentiment_LLM,
            device_map = device_map
        )
        return final_result

    def get_sentiment_from_website_each_feedback_sentiment(
            self,
            target_website,
            Use_Local_Sentiment_LLM = True,
            Use_Local_General_LLM = False,
            Use_Local_Scraper = None,
            Use_Scraper_API = None,
            Use_Groq_API = False,
            Use_Open_API = False,
            Local_Sentiment_LLM = "cardiffnlp/twitter-roberta-base-sentiment-latest",
            Local_General_LLM = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            Local_General_LLM_kwargs = {
                    'temperature':0.1,
                    'top_p':1
                } ,
            Groq_API = "",
            OpenAi_API = "",
            HuggingFace_API = "",
            Scraper_api_key = None,
            Local_api_key = None,
            Groq_LLM = "llama3-8b-8192",
            OpenAI_LLM = "GPT-3.5",
            device_map = "auto",
            get_Groq_Review = False, #Handle these
            get_OpenAI_review = False 

    ):
        Use_Local_Scraper = Use_Local_Scraper if Use_Local_Scraper is not None else self.Use_Local_Scraper
        Use_Scraper_API = Use_Scraper_API if Use_Scraper_API is not None else self.Use_Scraper_API
        Scraper_api_key = Scraper_api_key if Scraper_api_key is not None else self.Scraper_api_key
        Local_api_key = Local_api_key if Local_api_key is not None else self.Local_api_key
        if isinstance(target_website,str):
            fetched_review = Fetch_Reviews(
                target_website,
                Use_Local_Scraper,
                Use_Scraper_API,
                Scraper_api_key,
                Local_api_key
            )
        elif isinstance(target_website , list) and all(isinstance(item,str) for item in target_website):
            fetched_review_array = []
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
        return

    def get_sentiment_from_website_overall_summary(
            self,
            target_website,
            Use_Local_Sentiment_LLM = True,
            Use_Local_General_LLM = False,
            Use_Groq_API = False,
            Use_Open_API = False,
            Local_Sentiment_LLM = "cardiffnlp/twitter-roberta-base-sentiment-latest",
            Local_General_LLM = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            Local_General_LLM_kwargs = {
                    'temperature':0.1,
                    'top_p':1
                } ,
            Groq_API = "",
            OpenAi_API = "",
            HuggingFace_API = "",
            Groq_LLM = "llama3-8b-8192",
            OpenAI_LLM = "GPT-3.5",
            device_map = "auto"
    ):
        pass

    def get_analytical_customer_sentiments(
            self,
            target_website, 
            Multi_Website_Visualization = False,
            Graph_Config = {
                'Use_Bar_chart_visualize' : True, 
                'Use_pie_chart_visualize' : True, 
                'Use_sidebyside_bar_chart_visualize' :False   
            },
            Use_Local_Sentiment_LLM = True,
            Use_Local_General_LLM = False,
            Use_Groq_API = False,
            Use_Open_API = False,
            Local_Sentiment_LLM = "cardiffnlp/twitter-roberta-base-sentiment-latest",
            Local_General_LLM = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            Local_General_LLM_kwargs = {
                    'temperature':0.1,
                    'top_p':1
                } ,
            Groq_API = "",
            OpenAi_API = "",
            HuggingFace_API = "",
            Groq_LLM = "llama3-8b-8192",
            device_map = "auto",
            OpenAI_LLM = "GPT-3.5"
            
    ):
        pass

    def get_Sentiment_Audio_file(
            self,
            Audio_File_path,
            Use_Local_Sentiment_LLM = True,
            Use_Local_General_LLM = False,
            Use_Groq_API = False,
            Use_Open_API = False,
            Local_Sentiment_LLM = "cardiffnlp/twitter-roberta-base-sentiment-latest",
            Local_General_LLM = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            Local_General_LLM_kwargs = {
                    'temperature':0.1,
                    'top_p':1
                } ,
            Groq_API = "",
            OpenAi_API = "",
            HuggingFace_API = "",
            Groq_LLM = "llama3-8b-8192",
            device_map = "auto",
            OpenAI_LLM = "GPT-3.5"
    ):
        pass

    def compare_product_on_reviews(
        self,
        
        Use_Local_Sentiment_LLM = True,
        Use_Local_General_LLM = False,
        Use_Groq_API = False,
        Use_Open_API = False,
        Local_Sentiment_LLM = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        Local_General_LLM = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        Local_General_LLM_kwargs = {
                'temperature':0.1,
                'top_p':1
            } ,
        Groq_API = "",
        OpenAi_API = "",
        HuggingFace_API = "",
        Groq_LLM = "llama3-8b-8192",
        device_map = "auto",
        OpenAI_LLM = "GPT-3.5"  
    ):
        pass
from Sentimatrix.utils.Quick_Sentiment import Generation_Pipeline

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
        self.Use_Local_Sentiment_LLM = Use_Local_Sentiment_LLM
        self.Use_Local_General_LLM = Use_Local_General_LLM
        self.Use_Groq_API = Use_Groq_API
        self.Use_Open_API = Use_Open_API
        self.Local_Sentiment_LLM = Local_Sentiment_LLM
        self.Local_General_LLM = Local_General_LLM
        self.Local_General_LLM_kwargs = Local_General_LLM_kwargs
        self.Groq_API = Groq_API
        self.OpenAi_API = OpenAi_API
        self.HuggingFace_API = HuggingFace_API
        self.Groq_LLM = Groq_LLM
        self.OpenAI_LLM = OpenAI_LLM
        self.device_map = device_map

    def get_Quick_sentiment(
            self,
            text_message,
            Use_Local_Sentiment_LLM = True,
            Local_Sentiment_LLM = "cardiffnlp/twitter-roberta-base-sentiment-latest",
            device_map = "auto"
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
        self.text_message = text_message
        self.Use_Local_Sentiment_LLM = Use_Local_Sentiment_LLM
        self.Local_Sentiment_LLM = Local_Sentiment_LLM
        self.device_map = device_map
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
            device_map = "auto",
            Groq_label = ['Positive','Netural','Negative'],
            OpenAi_label = ['Positive','Netural','Negative']
    ):
        pass

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
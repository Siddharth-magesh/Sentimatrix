from Sentimatrix.utils.Quick_Sentiment import Generation_Pipeline
from Sentimatrix.utils.web_scraper import Fetch_Reviews
from Sentimatrix.utils.Structured_Sentiment import Struct_Generation_Pipeline
from Sentimatrix.utils.llm_inference.groq_inference import Groq_inference_list 
from Sentimatrix.utils.llm_inference.openai_inference import OpenAI_inference_list
from Sentimatrix.utils.llm_inference.localLLM_inference import LocalLLM_inference_list
from Sentimatrix.utils.llm_inference.groq_inference import summarize_reviews
from Sentimatrix.utils.llm_inference.openai_inference import summarize_reviews_openai
from Sentimatrix.utils.llm_inference.localLLM_inference import summarize_reviews_local
from Sentimatrix.utils.Structured_Sentiment import Struct_Generation_Pipeline_Visual
from Sentimatrix.utils.visualization import plot_sentiment_box_plot , plot_sentiment_distribution , plot_sentiment_histograms , plot_sentiment_pie_chart , plot_sentiment_violin_plot
from Sentimatrix.utils.wav_to_text import audio_to_text
from Sentimatrix.utils.llm_inference.groq_inference import compare_reviews_local
from Sentimatrix.utils.llm_inference.Florence_2_text import Convert_Image_to_Text
from Sentimatrix.utils.text_translation import Translate_text
from Sentimatrix.utils.web_scraper import add_review_pattern , get_review_patterns
from Sentimatrix.utils.save_to_csv import save_reviews_to_csv

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
        {'label': str, 'score': float} or [{'label': str, 'score': float},{'label': str, 'score': float},...]
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
            Use_Local_Sentiment_LLM = None,
            Use_Local_General_LLM = None,
            Use_Local_Scraper = None,
            Use_Scraper_API = None,
            Use_Groq_API = None,
            Use_Open_API = None,
            Local_Sentiment_LLM = None,
            Local_General_LLM = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            Local_General_LLM_kwargs = {
                    'temperature':0.1,
                    'top_p':1
                } ,
            Groq_API = None,
            OpenAi_API = None,
            HuggingFace_API = None,
            Scraper_api_key = None,
            Local_api_key = None,
            Groq_LLM = "llama3-8b-8192",
            OpenAI_LLM = "GPT-3.5",
            device_map = "auto",
            get_Groq_Review = False, #Handle these
            get_OpenAI_review = False,
            get_localLLM_review = False,
            Groq_LLM_Temperature = 0.1,
            Groq_LLM_Max_Tokens = 100,
            Groq_LLM_Max_Input_Tokens = 300,
            Groq_LLM_top_p = 1,
            Groq_LLM_stream = False,
            OpenAI_LLM_Temperature = 0.1,
            OpenAI_LLM_Max_Tokens = 100,
            OpenAI_LLM_stream = False,
            OpenAI_LLM_Max_Input_Tokens = 300,
            Local_LLM_Max_Input_Tokens = 300

    ):
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

        Use_Local_General_LLM = Use_Local_General_LLM if Use_Local_General_LLM is not None else self.Use_Local_General_LLM
        Local_General_LLM = Local_General_LLM if Local_General_LLM is not None else self.Local_General_LLM
        Local_General_LLM_kwargs = Local_General_LLM_kwargs if Local_General_LLM_kwargs is not None else self.Local_General_LLM_kwargs
        HuggingFace_API = HuggingFace_API if HuggingFace_API is not None else self.HuggingFace_API

        fetched_review = []
        fetched_review_array = []
        final_resulted_output = None
        final_resulted_output_of_all_sites = [] 

        #Getting the Reviews From Site
        if isinstance(target_website,str):
            fetched_review = Fetch_Reviews(
                target_website,
                Use_Local_Scraper,
                Use_Scraper_API,
                Scraper_api_key,
                Local_api_key
            )
        elif isinstance(target_website , list) and all(isinstance(item,str) for item in target_website):
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
        
        #Genrating Sentiment For Those Reviews
        if fetched_review:
            final_resulted_output = Struct_Generation_Pipeline(
                text_message=fetched_review,
                Use_Local_Sentiment_LLM = True,
                model_id = Local_Sentiment_LLM,
                device_map=device_map
            )
            #return final_resulted_output
        elif fetched_review_array:
            webcount = 1
            for each_website_reviews in fetched_review_array:
                final_resulted_output_of_each_site = Struct_Generation_Pipeline(
                    text_message=each_website_reviews,
                    Use_Local_Sentiment_LLM = True,
                    model_id = Local_Sentiment_LLM,
                    device_map=device_map
                )
                final_resulted_output_of_each_site.insert(0,webcount)
                webcount = webcount + 1
                final_resulted_output_of_all_sites.append(final_resulted_output_of_each_site)
            #return final_resulted_output_of_all_sites
        else:
            return "Error: Didnt Find Any Inputs"
        
        #Adding Additional Comments To those Reviews
        if get_Groq_Review ==True or get_OpenAI_review ==True or get_localLLM_review ==True:
            if get_localLLM_review: #Local LLM Inference
                if final_resulted_output: #Single Website Given by the user
                    updated_reviews = LocalLLM_inference_list(
                        final_resulted_output,
                        model_name=Local_General_LLM,
                        Local_General_LLM_kwargs=Local_General_LLM_kwargs,
                        max_input_tokens=Local_LLM_Max_Input_Tokens,
                        device_map=device_map
                    )
                    return updated_reviews
                elif final_resulted_output_of_all_sites: #Multiple Website given by the user
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
                        all_sites_updated_reviews.append([site_id] + updated_reviews_site)
                    return all_sites_updated_reviews
                else:
                    print("Error: Didn't receive any result from the sites")
            elif get_Groq_Review: #Groq LLM Inference
                if final_resulted_output: #Single Website Given by the user
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
                elif final_resulted_output_of_all_sites:  #Multiple Website given by the user
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
                        all_sites_updated_reviews.append([site_id] + updated_reviews_site)
                    return all_sites_updated_reviews
                else:
                    print("Error : Didnt Recieve any result from the sites")
            elif get_OpenAI_review: #OpenAI LLM Inference
                if final_resulted_output: #Single Website Given by the user
                    updated_reviews = OpenAI_inference_list(
                        final_resulted_output,
                        OpenAi_API,
                        temperature=OpenAI_LLM_Temperature,
                        max_tokens=OpenAI_LLM_Max_Tokens,
                        stream=OpenAI_LLM_stream,
                        max_input_tokens=OpenAI_LLM_Max_Input_Tokens
                    )
                    return updated_reviews
                elif final_resulted_output_of_all_sites:  #Multiple Website given by the user
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
                        all_sites_updated_reviews.append([site_id] + updated_reviews_site)
                    return all_sites_updated_reviews
            else:
                print("No valid review source selected.")
                return None

        return "Function Didnt Properly Execute"

    def get_sentiment_from_website_overall_summary(
            self,
            target_website,
            Use_Local_Sentiment_LLM = None,
            Use_Local_General_LLM = None,
            Use_Local_Scraper = None,
            Use_Scraper_API = None,
            Use_Groq_API = None,
            Use_Open_API = None,
            Local_Sentiment_LLM = None,
            Local_General_LLM = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            Local_General_LLM_kwargs = {
                    'temperature':0.1,
                    'top_p':1
                } ,
            Groq_API = None,
            OpenAi_API = None,
            HuggingFace_API = None,
            Scraper_api_key = None,
            Local_api_key = None,
            Groq_LLM = "llama3-8b-8192",
            OpenAI_LLM = "GPT-3.5",
            device_map = "auto",
            Groq_LLM_Temperature = 0.1,
            Groq_LLM_Max_Tokens = 100,
            Groq_LLM_Max_Input_Tokens = 300,
            Groq_LLM_top_p = 1,
            Groq_LLM_stream = False,
            OpenAI_LLM_Temperature = 0.1,
            OpenAI_LLM_Max_Tokens = 100,
            OpenAI_LLM_stream = False,
            OpenAI_LLM_Max_Input_Tokens = 300,
            Local_LLM_Max_Input_Tokens = 300
    ):
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

        Use_Local_General_LLM = Use_Local_General_LLM if Use_Local_General_LLM is not None else self.Use_Local_General_LLM
        Local_General_LLM = Local_General_LLM if Local_General_LLM is not None else self.Local_General_LLM
        Local_General_LLM_kwargs = Local_General_LLM_kwargs if Local_General_LLM_kwargs is not None else self.Local_General_LLM_kwargs
        HuggingFace_API = HuggingFace_API if HuggingFace_API is not None else self.HuggingFace_API

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
                Use_Local_Sentiment_LLM = True,
                model_id = Local_Sentiment_LLM,
                device_map=device_map
            )
        else:
            print("Error : No Values Have been Fetched")

        if Use_Groq_API ==True or Use_Open_API ==True or Use_Local_General_LLM ==True:
            if Use_Groq_API:
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
            elif Use_Open_API:
                Summarized_result_OpenAi = summarize_reviews_openai(
                    reviews=final_resulted_output,
                    KEY=OpenAi_API,
                    model_id=OpenAI_LLM,
                    max_tokens=OpenAI_LLM_Max_Tokens,
                    temperature=OpenAI_LLM_Temperature,
                    stream=OpenAI_LLM_stream
                )
                return Summarized_result_OpenAi
            elif Use_Local_General_LLM:
                Summarized_result_LocalLLM = summarize_reviews_local(
                    reviews=final_resulted_output,
                    model_path=Local_General_LLM
                )
                return Summarized_result_LocalLLM
            else:
                print("Error : use any one of the LLM inference")

    def get_analytical_customer_sentiments(
            self,
            target_website, 
            Use_Local_Scraper = None,
            Use_Scraper_API = None,
            Scraper_api_key = None,
            Local_api_key = None,
            Use_Local_Sentiment_LLM = True,
            Use_Bar_chart_visualize = False,
            Use_pie_chart_visualize = False,
            Use_violin_plot_visualize = False,
            Use_box_plot_visualize = False,
            Use_histogram_visualize = False,
            device_map = "auto",
            Local_Sentiment_LLM = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ):
        Use_Local_Scraper = Use_Local_Scraper if Use_Local_Scraper is not None else self.Use_Local_Scraper
        Use_Scraper_API = Use_Scraper_API if Use_Scraper_API is not None else self.Use_Scraper_API
        Scraper_api_key = Scraper_api_key if Scraper_api_key is not None else self.Scraper_api_key
        Local_api_key = Local_api_key if Local_api_key is not None else self.Local_api_key
        Local_Sentiment_LLM = Local_Sentiment_LLM if Local_Sentiment_LLM is not None else self.Local_Sentiment_LLM
        device_map = device_map if device_map is not None else self.device_map
        Use_Local_Sentiment_LLM = Use_Local_Sentiment_LLM if Use_Local_Sentiment_LLM is not None else self.Use_Local_Sentiment_LLM

        fetched_review = Fetch_Reviews(
            target_website,
            Use_Local_Scraper,
            Use_Scraper_API,
            Scraper_api_key,
            Local_api_key
        )
        if fetched_review:
            final_resulted_output = Struct_Generation_Pipeline_Visual(
                text_message=fetched_review,
                Use_Local_Sentiment_LLM = True,
                model_id = Local_Sentiment_LLM,
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
        

    def get_Sentiment_Audio_file(
            self,
            Audio_File_path = None,
            Use_Local_Sentiment_LLM = True,
            Local_Sentiment_LLM = "cardiffnlp/twitter-roberta-base-sentiment-latest",
            device_map = "auto"
    ):
        Use_Local_Sentiment_LLM = Use_Local_Sentiment_LLM if Use_Local_Sentiment_LLM is not None else self.Use_Local_Sentiment_LLM
        Local_Sentiment_LLM = Local_Sentiment_LLM if Local_Sentiment_LLM is not None else self.Local_Sentiment_LLM
        device_map = device_map if device_map is not None else self.device_map
        if Audio_File_path:
            retrieved_text = audio_to_text(Audio_File_path)
            sentiment = self.get_Quick_sentiment(
                text_message=retrieved_text,
                Use_Local_Sentiment_LLM=Use_Local_Sentiment_LLM,
                Local_Sentiment_LLM = Local_Sentiment_LLM,
                device_map=device_map
            )
            return sentiment
        else:
            print("Mention the Audio Path")

    def compare_product_on_reviews(
        self,
        target_website1,
        target_website2,
        Use_Local_Sentiment_LLM = None,
        Use_Local_General_LLM = None,
        Use_Local_Scraper = None,
        Use_Scraper_API = None,
        Use_Groq_API = None,
        Use_Open_API = None,
        Local_Sentiment_LLM = None,
        Local_General_LLM = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        Local_General_LLM_kwargs = {
                'temperature':0.1,
                'top_p':1
            } ,
        Groq_API = None,
        OpenAi_API = None,
        HuggingFace_API = None,
        Scraper_api_key = None,
        Local_api_key = None,
        Groq_LLM = "llama3-8b-8192",
        OpenAI_LLM = "GPT-3.5",
        device_map = "auto",
        Groq_LLM_Temperature = 0.1,
        Groq_LLM_Max_Tokens = 100,
        Groq_LLM_Max_Input_Tokens = 300,
        Groq_LLM_top_p = 1,
        Groq_LLM_stream = False,
        OpenAI_LLM_Temperature = 0.1,
        OpenAI_LLM_Max_Tokens = 100,
        OpenAI_LLM_stream = False,
        OpenAI_LLM_Max_Input_Tokens = 300,
        Local_LLM_Max_Input_Tokens = 300 
    ):
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

        Use_Local_General_LLM = Use_Local_General_LLM if Use_Local_General_LLM is not None else self.Use_Local_General_LLM
        Local_General_LLM = Local_General_LLM if Local_General_LLM is not None else self.Local_General_LLM
        Local_General_LLM_kwargs = Local_General_LLM_kwargs if Local_General_LLM_kwargs is not None else self.Local_General_LLM_kwargs
        HuggingFace_API = HuggingFace_API if HuggingFace_API is not None else self.HuggingFace_API

        if isinstance(target_website1,str):
            fetched_review1 = Fetch_Reviews(
                target_website1,
                Use_Local_Scraper,
                Use_Scraper_API,
                Scraper_api_key,
                Local_api_key
            )
        else:
            print("Error : Coundnt Fetch Values from Site 1")

        if isinstance(target_website2,str):
            fetched_review2 = Fetch_Reviews(
                target_website2,
                Use_Local_Scraper,
                Use_Scraper_API,
                Scraper_api_key,
                Local_api_key
            )
        else:
            print("Error : Coundnt Fetch Values from Site 2")

        if fetched_review1 and fetched_review2:
            final_resulted_output1 = Struct_Generation_Pipeline(
                text_message=fetched_review1,
                Use_Local_Sentiment_LLM = True,
                model_id = Local_Sentiment_LLM,
                device_map=device_map
            )
            final_resulted_output2 = Struct_Generation_Pipeline(
                text_message=fetched_review2,
                Use_Local_Sentiment_LLM = True,
                model_id = Local_Sentiment_LLM,
                device_map=device_map
            )
        else:
            print("Error : Coundnt find Reviews in either one of the sites")

        print(final_resulted_output1)
        print('\n\n------------')
        print(final_resulted_output1)

        if final_resulted_output1 and final_resulted_output2:
            compared_reviews = compare_reviews_local(
                        reviews1=final_resulted_output1,
                        reviews2=final_resulted_output2,
                        KEY=Groq_API,
                        temperature=Groq_LLM_Temperature,
                        top_p=Groq_LLM_top_p,
                        stream=Groq_LLM_stream,
                        max_input_tokens=Groq_LLM_Max_Input_Tokens,
                        max_tokens=Groq_LLM_Max_Tokens
            )
            return compared_reviews
        else:
            print("Error : Sentiment Generation Error")


    def get_Sentiment_Image_file(
            self,
            Image_File_path = None,
            Custom_Prompt = None,
            Main_Prompt = '<MORE_DETAILED_CAPTION>',
            Use_Local_Sentiment_LLM = True,
            Local_Sentiment_LLM = "cardiffnlp/twitter-roberta-base-sentiment-latest",
            device_map = "auto",
            Image_to_Text_Model = None
    ):
        Use_Local_Sentiment_LLM = Use_Local_Sentiment_LLM if Use_Local_Sentiment_LLM is not None else self.Use_Local_Sentiment_LLM
        Local_Sentiment_LLM = Local_Sentiment_LLM if Local_Sentiment_LLM is not None else self.Local_Sentiment_LLM
        device_map = device_map if device_map is not None else self.device_map

        if Image_File_path :
            if Image_to_Text_Model == 'microsoft/Florence-2-large':
                Extracted_Text = Convert_Image_to_Text(
                    image_path=Image_File_path,
                    text_input=Custom_Prompt,
                    task_prompt=Main_Prompt
                )
                if Extracted_Text:
                    sentiment = self.get_Quick_sentiment(
                        text_message=Extracted_Text,
                        Use_Local_Sentiment_LLM=Use_Local_Sentiment_LLM,
                        Local_Sentiment_LLM = Local_Sentiment_LLM,
                        device_map=device_map
                    )
                    return [{'Extracted_Text':Extracted_Text},sentiment]
                else:
                    print("Error : Couldn't Extract any text")
        else:
            print("Error : Image path Not Provided")

    def Multi_language_Sentiment(
            self,
            text_message,
            Use_Local_Sentiment_LLM = True,
            Local_Sentiment_LLM = None,
            device_map = None,
            
    ):
        Use_Local_Sentiment_LLM = Use_Local_Sentiment_LLM if Use_Local_Sentiment_LLM is not None else self.Use_Local_Sentiment_LLM
        Local_Sentiment_LLM = Local_Sentiment_LLM if Local_Sentiment_LLM is not None else self.Local_Sentiment_LLM
        device_map = device_map if device_map is not None else self.device_map

        if text_message:
            translated_text = Translate_text(message=text_message)
            if translated_text:
                sentiment = self.get_Quick_sentiment(
                        text_message=translated_text,
                        Use_Local_Sentiment_LLM=Use_Local_Sentiment_LLM,
                        Local_Sentiment_LLM = Local_Sentiment_LLM,
                        device_map=device_map
                    )
                return [{'Original Text':text_message},{'Translated Text':translated_text},sentiment]
            else:
                    print("Error : Couldn't Translate any text")
        else:
            print("Error : No message Recieved")

    def Config_Local_Scraper(
            self,
            action,
            tag = None,
            attrs = None,
    ):
        if action=='add':
            if tag and attrs:
                add_review_pattern(tag=tag,attrs=attrs)
        elif action=='get':
            result = get_review_patterns()
            return result
        else:
            print("Error : Invalid Action (Either Set to 'add' or 'get')")

    def Save_reviews_to_CSV(
            self,
            target_site,
            output_dir,
            file_name,
            Use_Local_Sentiment_LLM = None,
            Use_Local_Scraper = None
    ):
        Use_Local_Scraper = Use_Local_Scraper if Use_Local_Scraper is not None else self.Use_Local_Scraper
        Use_Local_Sentiment_LLM = Use_Local_Sentiment_LLM if Use_Local_Sentiment_LLM is not None else self.Use_Local_Sentiment_LLM
        Fetched_reviews = self.get_sentiment_from_website_each_feedback_sentiment(
            target_website=target_site,
             Use_Local_Sentiment_LLM = True,
             Use_Local_Scraper=True,
             get_Groq_Review = False,
             get_OpenAI_review = False,
             get_localLLM_review = False
        )
        save_reviews_to_csv(reviews=Fetched_reviews,output_dir=output_dir,file_name=file_name)
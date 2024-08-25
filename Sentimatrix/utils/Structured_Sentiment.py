import os
import warnings
import logging
import transformers
from transformers import pipeline, AutoTokenizer
from Sentimatrix.utils.device_compactability_check import check_cuda_availability

# Suppresses TensorFlow and ONEDNN optimization warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Ignores general and future warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

# Sets logging level for the transformers library to avoid excessive output
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

def Struct_Generation_Pipeline(
        text_message,
        Use_Local_Sentiment_LLM,
        model_id,
        device_map,
):
    """
    Performs sentiment analysis on the given text(s) and returns both the original text and sentiment results.

    Args:
        text_message (str or list of str): The text(s) to analyze.
        Use_Local_Sentiment_LLM (bool): Whether to use a locally stored sentiment analysis model.
        model_id (str): The ID of the sentiment analysis model to use.
        device_map (str): Device configuration ("auto", "cpu", "cuda"). "auto" will determine the best device automatically.

    Returns:
        list: For a single text input, a list containing the original text and its sentiment.
              For multiple texts, a list of such lists for each text.
    """
    
    TASK_TYPE = "text-classification"  # Specifies the type of pipeline task (sentiment analysis)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Set truncation length to the smaller of the model's max length or 512 tokens
    truncation_length = min(tokenizer.model_max_length, 512)

    # Automatically determine the device if set to "auto"
    if device_map.lower() == "auto":
        device_map = check_cuda_availability()

    # Validate the device_map input; fallback to automatic check if invalid
    if device_map.lower() not in ["auto", "cpu", "cuda"]:
        device_map = check_cuda_availability()
        print("The Device Map Should be either set to 'auto', 'cuda', or 'cpu'.")

    # Proceed if using the local model for sentiment analysis
    if Use_Local_Sentiment_LLM:
        # Initialize the sentiment analysis pipeline with the selected model and device
        pipe = pipeline(
            TASK_TYPE,
            model=model_id,
            device_map=device_map,
            tokenizer=tokenizer
        )

        # If input is a single string, process and return the result
        if isinstance(text_message, str):
            result = pipe(text_message, truncation=True, max_length=truncation_length)
            formatted_output = [{'text-message': text_message}, result[0]]
            return formatted_output

        # If input is a list of strings, process each text and return the results
        elif isinstance(text_message, list) and all(isinstance(item, str) for item in text_message):
            result_array = []
            for message in text_message:
                result = pipe(message, truncation=True, max_length=truncation_length)
                result_array.append([{'text-message': message}, result[0]])
            return result_array

        # Handle cases where input format is invalid
        else:
            return "Error: The accepted format is either a list of strings or a single string."
    
    else:
        return "Use_Local_Sentiment_LLM = False. This parameter should be set to True for local model usage."


def Struct_Generation_Pipeline_Visual(
        text_message,
        Use_Local_Sentiment_LLM,
        model_id,
        device_map,
):
    """
    Similar to Struct_Generation_Pipeline, but returns only the sentiment analysis results without the original text.

    Args:
        text_message (str or list of str): The text(s) to analyze.
        Use_Local_Sentiment_LLM (bool): Whether to use a locally stored sentiment analysis model.
        model_id (str): The ID of the sentiment analysis model to use.
        device_map (str): Device configuration ("auto", "cpu", "cuda"). "auto" will determine the best device automatically.

    Returns:
        list: For a single text input, a list containing only the sentiment.
              For multiple texts, a list of such results for each text.
    """
    
    TASK_TYPE = "text-classification"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Set truncation length to the smaller of the model's max length or 512 tokens
    truncation_length = min(tokenizer.model_max_length, 512)

    # Automatically determine the device if set to "auto"
    if device_map.lower() == "auto":
        device_map = check_cuda_availability()

    # Validate the device_map input; fallback to automatic check if invalid
    if device_map.lower() not in ["auto", "cpu", "cuda"]:
        device_map = check_cuda_availability()
        print("The Device Map Should be either set to 'auto', 'cuda', or 'cpu'.")

    # Proceed if using the local model for sentiment analysis
    if Use_Local_Sentiment_LLM:
        # Initialize the sentiment analysis pipeline with the selected model and device
        pipe = pipeline(
            TASK_TYPE,
            model=model_id,
            device_map=device_map,
            tokenizer=tokenizer
        )

        # If input is a single string, process and return the result
        if isinstance(text_message, str):
            result = pipe(text_message, truncation=True, max_length=truncation_length)
            formatted_output = [{'text-message': text_message}, result[0]]
            return formatted_output

        # If input is a list of strings, process each text and return the results without the original message
        elif isinstance(text_message, list) and all(isinstance(item, str) for item in text_message):
            result_array = []
            for message in text_message:
                result = pipe(message, truncation=True, max_length=truncation_length)
                result_array.append([result[0]])
            return result_array

        # Handle cases where input format is invalid
        else:
            return "Error: The accepted format is either a list of strings or a single string."
    
    else:
        return "Use_Local_Sentiment_LLM = False. This parameter should be set to True for local model usage."
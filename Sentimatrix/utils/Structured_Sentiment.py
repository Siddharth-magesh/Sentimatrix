import os
import warnings
import logging
import transformers
from transformers import pipeline, AutoTokenizer
from Sentimatrix.utils.device_compactability_check import check_cuda_availability

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

def Struct_Generation_Pipeline(
        text_message,
        Use_Local_Sentiment_LLM,
        model_id,
        device_map,
):
    TASK_TYPE = "text-classification"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Ensure that the truncation length is within a reasonable range
    truncation_length = min(tokenizer.model_max_length, 512)  # Default to 512 tokens max

    if device_map.lower() == "auto":
        device_map = check_cuda_availability()
    
    if device_map.lower() not in ["auto", "cpu", "cuda"]:
        device_map = check_cuda_availability()
        print("The Device Map Should be either set to 'auto', 'cuda', or 'cpu'.")

    if Use_Local_Sentiment_LLM:
        pipe = pipeline(
            TASK_TYPE,
            model=model_id,
            device_map=device_map,
            tokenizer=tokenizer
        )

        if isinstance(text_message, str):
            result = pipe(text_message, truncation=True, max_length=truncation_length)
            formatted_output = [{'text-message': text_message}, result[0]]
            return formatted_output

        elif isinstance(text_message, list) and all(isinstance(item, str) for item in text_message):
            result_array = []
            for message in text_message:
                result = pipe(message, truncation=True, max_length=truncation_length)
                result_array.append([{'text-message': message}, result[0]])
            return result_array

        else:
            return "Error: The accepted format is either a list of strings or a single string."
    
    else:
        return "Use_Local_Sentiment_LLM = False. This param should be set to True."
    
def Struct_Generation_Pipeline_Visual(
        text_message,
        Use_Local_Sentiment_LLM,
        model_id,
        device_map,
):
    TASK_TYPE = "text-classification"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Ensure that the truncation length is within a reasonable range
    truncation_length = min(tokenizer.model_max_length, 512)  # Default to 512 tokens max

    if device_map.lower() == "auto":
        device_map = check_cuda_availability()
    
    if device_map.lower() not in ["auto", "cpu", "cuda"]:
        device_map = check_cuda_availability()
        print("The Device Map Should be either set to 'auto', 'cuda', or 'cpu'.")

    if Use_Local_Sentiment_LLM:
        pipe = pipeline(
            TASK_TYPE,
            model=model_id,
            device_map=device_map,
            tokenizer=tokenizer
        )

        if isinstance(text_message, str):
            result = pipe(text_message, truncation=True, max_length=truncation_length)
            formatted_output = [{'text-message': text_message}, result[0]]
            return formatted_output

        elif isinstance(text_message, list) and all(isinstance(item, str) for item in text_message):
            result_array = []
            for message in text_message:
                result = pipe(message, truncation=True, max_length=truncation_length)
                result_array.append([result[0]])
            return result_array

        else:
            return "Error: The accepted format is either a list of strings or a single string."
    
    else:
        return "Use_Local_Sentiment_LLM = False. This param should be set to True."

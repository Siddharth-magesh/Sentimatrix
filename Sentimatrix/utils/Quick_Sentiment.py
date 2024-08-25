import os
import warnings
import logging
import transformers
from transformers import pipeline
from .device_compactability_check import check_cuda_availability

# Suppresses TensorFlow and ONEDNN optimization warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Ignores general and future warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

# Sets logging level for the transformers library to avoid excessive output
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

def Generation_Pipeline(
        text_message,
        Use_Local_Sentiment_LLM,
        model_id,
        device_map
):
    """
    Generates sentiment based on the given text input.

    Args:
        text_message (str or list of str): The text(s) for which sentiment analysis needs to be performed.
        Use_Local_Sentiment_LLM (bool): Indicates whether to use a locally stored sentiment analysis model.
        model_id (str): The ID of the model to be used for sentiment analysis.
        device_map (str): Specifies the device to be used ("auto", "cpu", or "cuda"). If set to "auto", 
                          it will automatically determine device compatibility.

    Returns:
        dict or list of dict: If a single text message is passed, returns a dictionary containing sentiment 
                              information with 'label' and 'score' fields.
                              For a list of messages, returns a list of dictionaries with the 
                              sentiment information for each message.
    """
    TASK_TYPE = "text-classification"

    # Check and set the appropriate device map
    if device_map.lower() == "auto":
        device_map = check_cuda_availability()

    # Validate device_map input and fallback if invalid
    if device_map.lower() not in ["auto", "cpu", "cuda"]:
        device_map = check_cuda_availability()
        print("The Device Map should be either set to 'auto', 'cuda', or 'cpu'.")

    # Process sentiment analysis using the local model if enabled
    if Use_Local_Sentiment_LLM:
        pipe = pipeline(
            TASK_TYPE,
            model=model_id,
            device_map=device_map
        )

        # Check if input is a string
        if isinstance(text_message, str):
            result = pipe(text_message)
            return result[0]

        # Check if input is a list of strings
        elif isinstance(text_message, list) and all(isinstance(item, str) for item in text_message):
            result_array = []
            for message in text_message:
                result = pipe(message)
                result_array.append(result[0])
            return result_array

        # Return an error message if input format is incorrect
        else:
            return "Error: The accepted format is either a list of strings or a single string."
    
    else:
        return "Use_Local_Sentiment_LLM = False. This parameter should be set to True for local model usage."

# Example output:
# {'label': 'neutral', 'score': 0.45358896255493164}
# If a single input text message is provided, the output will be as shown above.
# If a list of messages is provided, the output will be a list of dictionaries containing similar results.
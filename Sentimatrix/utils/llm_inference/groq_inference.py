from groq import Groq

def Groq_inference_list(
        reviews,  # List of lists containing reviews and corresponding sentiment data
        KEY: str,  # Groq API key for authentication
        max_tokens: int = 100,  # Maximum number of tokens to generate for the Groq review
        max_input_tokens: int = 300,  # Maximum number of tokens to consider from the review content (input truncation)
        temperature: float = 0.1,  # Sampling temperature for text generation (lower values = more deterministic)
        top_p: float = 1,  # Top-p sampling value for controlling diversity in generated text
        model_id: str = "llama3-8b-8192",  # Model ID for Groq LLM used for inference
        stream: bool = False  # Whether to stream the response from the API (default is False)
    ):
    """
    Function that takes a list of reviews and their corresponding sentiment data, generates a Groq model inference 
    for each review, and appends the generated Groq review to each review-sentiment pair.
    
    Parameters:
    - reviews: List[List[Dict]] 
        A list of lists where each sublist contains:
        - A dictionary with the key 'text-message' representing the review content.
        - A dictionary with sentiment data, including 'label' (e.g., 'positive', 'neutral') and 'score' (a confidence score).
    
    - KEY: str
        The Groq API key required to authenticate and make requests to the Groq API.
    
    - max_tokens: int (default = 100)
        The maximum number of tokens the model can generate in response.
    
    - max_input_tokens: int (default = 300)
        The maximum number of tokens to consider from the input review content. If the review exceeds this limit, it will be truncated.
    
    - temperature: float (default = 0.1)
        The temperature setting for the model, controlling the randomness of the output. Lower values result in more deterministic responses.
    
    - top_p: float (default = 1)
        Top-p sampling controls the diversity of the generated text. A value of 1 means no restriction on sampling.
    
    - model_id: str (default = "llama3-8b-8192")
        The specific model ID for the Groq LLM to use for inference.
    
    - stream: bool (default = False)
        Whether or not to enable streaming responses from the API.
    
    Returns:
    - reviews: List[List[Dict]]
        The input list with each sublist now having an additional dictionary containing the generated Groq review under the key 'groq-review'.
    """
    
    # Create a Groq client with the provided API key
    client = Groq(api_key=KEY)
    
    # Loop over each review and sentiment in the list of lists
    for review_data in reviews:
        # Extract the review content from the first dictionary in the sublist
        content = review_data[0].get('text-message', '')
        # Extract the sentiment data from the second dictionary in the sublist
        sentiment = review_data[1]
        
        # If the review content exceeds the max_input_tokens limit, truncate it
        if len(content) > max_input_tokens:
            content = content[:max_input_tokens] + "..."
        
        # Call the Groq API to generate a response based on the review and sentiment
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Given the review and sentiment, provide insights on the customer's feelings about the product."
                },
                {
                    "role": "user",
                    "content": f"REVIEW: {content}\nSENTIMENT: {sentiment}\nQUESTION: How does the customer feel about the product?"
                }
            ],
            model=model_id,  # Specify the model ID
            temperature=temperature,  # Set the temperature
            top_p=top_p,  # Set the top-p value
            max_tokens=max_tokens,  # Limit the number of output tokens
            stop=None,  # No stop condition provided
            stream=stream  # Enable or disable streaming
        )
        
        # Extract the generated Groq review from the API response
        groq_review = chat_completion.choices[0].message.content
        
        # Append the generated Groq review to the current review data
        review_data.append({"groq-review": groq_review})
    
    # Return the modified list with Groq reviews appended to each sublist
    return reviews

def summarize_reviews(
        reviews,  # List of lists containing reviews and corresponding sentiment data
        KEY: str,  # Groq API key for authentication
        max_tokens: int = 150,  # Maximum number of tokens to generate for the summary
        temperature: float = 0.3,  # Sampling temperature for text generation (lower values = more deterministic)
        top_p: float = 1,  # Top-p sampling value for controlling diversity in generated text
        model_id: str = "llama3-8b-8192",  # Model ID for Groq LLM used for inference
        stream: bool = False  # Whether to stream the response from the API (default is False)
    ):
    """
    Function that takes a list of reviews and their corresponding sentiment data, 
    and generates a summary of all reviews using the Groq model.
    
    Parameters:
    - reviews: List[List[Dict]]
        A list of lists where each sublist contains:
        - A dictionary with the key 'text-message' representing the review content.
        - A dictionary with sentiment data, including 'label' (e.g., 'positive', 'neutral') and 'score' (a confidence score).
    
    - KEY: str
        The Groq API key required to authenticate and make requests to the Groq API.
    
    - max_tokens: int (default = 150)
        The maximum number of tokens the model can generate for the summary.
    
    - temperature: float (default = 0.3)
        The temperature setting for the model, controlling the randomness of the output. Lower values result in more deterministic responses.
    
    - top_p: float (default = 1)
        Top-p sampling controls the diversity of the generated text. A value of 1 means no restriction on sampling.
    
    - model_id: str (default = "llama3-8b-8192")
        The specific model ID for the Groq LLM to use for inference.
    
    - stream: bool (default = False)
        Whether or not to enable streaming responses from the API.
    
    Returns:
    - summary: str
        A summary of the reviews generated by the Groq model.
    """
    
    # Create a Groq client with the provided API key
    client = Groq(api_key=KEY)
    
    # Combine all the reviews into a single string for summarization
    combined_reviews = "\n".join([f"REVIEW: {review[0]['text-message']} (Sentiment: {review[1]['label']}, Score: {review[1]['score']})" for review in reviews])
    
    # Call the Groq API to generate a summary based on the combined reviews
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Summarize the overall sentiment and key points from the reviews provided below."
            },
            {
                "role": "user",
                "content": f"Here are the reviews:\n{combined_reviews}\nPlease provide a concise summary of the overall sentiment and main feedback points."
            }
        ],
        model=model_id,  # Specify the model ID
        temperature=temperature,  # Set the temperature
        top_p=top_p,  # Set the top-p value
        max_tokens=max_tokens,  # Limit the number of output tokens
        stop=None,  # No stop condition provided
        stream=stream  # Enable or disable streaming
    )
    
    # Extract the generated summary from the API response
    summary = chat_completion.choices[0].message.content
    
    return summary

from groq import Groq

def compare_reviews_local(
        reviews1,  # Reviews for the first product
        reviews2,  # Reviews for the second product
        KEY: str,  # Groq API key for authentication
        max_tokens: int = 500,  # Maximum number of tokens to generate for the summary
        max_input_tokens: int = 150,  # Maximum number of tokens to consider from the review content (input truncation)
        temperature: float = 0.3,  # Sampling temperature for text generation (lower values = more deterministic)
        top_p: float = 1.0,  # Top-p sampling value for controlling diversity in generated text
        model_id: str = "llama3-8b-8192",  # Model ID for Groq LLM used for inference
        stream: bool = False  # Whether to stream the response from the API (default is False)
    ):
    """
    Function to compare reviews from two different products and generate a comparison summary using the Groq model.
    
    Parameters:
    - reviews1: List[List[Dict]]
        A list of lists where each sublist contains:
        - A dictionary with the key 'text-message' representing the review content for the first product.
        - A dictionary with sentiment data, including 'label' and 'score'.
    
    - reviews2: List[List[Dict]]
        A list of lists where each sublist contains:
        - A dictionary with the key 'text-message' representing the review content for the second product.
        - A dictionary with sentiment data, including 'label' and 'score'.
    
    - KEY: str
        The Groq API key required to authenticate and make requests to the Groq API.
    
    - max_tokens: int (default = 150)
        The maximum number of tokens the model can generate in response.
    
    - max_input_tokens: int (default = 500)
        The maximum number of tokens to consider from the input review content. If the review exceeds this limit, it will be truncated.
    
    - temperature: float (default = 0.3)
        The temperature setting for the model, controlling the randomness of the output. Lower values result in more deterministic responses.
    
    - top_p: float (default = 1.0)
        Top-p sampling controls the diversity of the generated text. A value of 1.0 means no restriction on sampling.
    
    - model_id: str (default = "llama3-8b-8192")
        The specific model ID for the Groq LLM to use for inference.
    
    - stream: bool (default = False)
        Whether or not to enable streaming responses from the API.
    
    Returns:
    - comparison_summary: str
        A comparison summary of the reviews for the two products generated by the Groq model.
    """
    
    # Create a Groq client with the provided API key
    client = Groq(api_key=KEY)
    
    # Combine the reviews for both products into a single string
    reviews1_text = "\n".join([f"Product 1 - REVIEW: {review[0]['text-message']} (Sentiment: {review[1]['label']}, Score: {review[1]['score']})" for review in reviews1])
    reviews2_text = "\n".join([f"Product 2 - REVIEW: {review[0]['text-message']} (Sentiment: {review[1]['label']}, Score: {review[1]['score']})" for review in reviews2])
    
    prompt = (
        f"Compare the reviews for the two products below and provide a summary of how they compare in terms of customer satisfaction and feedback:\n\n"
        f"Product 1 Reviews:\n{reviews1_text}\n\n"
        f"Product 2 Reviews:\n{reviews2_text}\n\n"
        f"Please provide a concise summary comparing the overall sentiment and key points from both products."
    )
    
    # If the prompt content exceeds the max_input_tokens limit, truncate it
    if len(prompt) > max_input_tokens:
        prompt = prompt[:max_input_tokens] + "..."
    
    # Call the Groq API to generate a response based on the comparison prompt
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Compare the reviews for the two products and provide a summary of how they compare."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=model_id,  # Specify the model ID
        temperature=temperature,  # Set the temperature
        top_p=top_p,  # Set the top-p value
        max_tokens=max_tokens,  # Limit the number of output tokens
        stop=None,  # No stop condition provided
        stream=stream  # Enable or disable streaming
    )
    
    # Extract the generated comparison summary from the API response
    comparison_summary = chat_completion.choices[0].message.content
    
    return comparison_summary
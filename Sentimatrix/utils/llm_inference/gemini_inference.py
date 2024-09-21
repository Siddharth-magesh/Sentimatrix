from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

def Gemini_inference_list(
    reviews,
    KEY: str,
    max_tokens: int = 100,
    max_input_tokens: int = 300,
    temperature: float = 0.1,
    model: str = "gemini-1.5-pro"
):
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=KEY,
    )

    for review_data in reviews:
        content = review_data[0].get('text-message', '')
        sentiment = review_data[1]

        if len(content) > max_input_tokens:
            content = content[:max_input_tokens] + "..."

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Given the review and sentiment, provide insights on the customer's feelings about the product."),
                ("human", "REVIEW: {content}\nSENTIMENT: {sentiment_label}\nQUESTION: How does the customer feel about the product?"),
            ]
        )
        chain = prompt | llm
        response = chain.invoke({
            "content": content,
            "sentiment_label": sentiment['label']
        })
        gemini_review = response.content
        review_data.append({"gemini-review": gemini_review})

    return reviews

def summarize_reviews_gemini(
    reviews,
    KEY: str,
    max_tokens: int = 500,
    temperature: float = 0.3,
    model: str = "gemini-1.5-pro",
):
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=KEY,
    )

    combined_reviews = "\n".join(
        [f"REVIEW: {review[0]['text-message']} (Sentiment: {review[1]['label']}, Score: {review[1]['score']})" for review in reviews]
    )

    prompt = f"Summarize the overall sentiment and key points from the reviews:\n{combined_reviews}"

    chain = ChatPromptTemplate.from_messages([("system", "Summarize the reviews:"), ("human", prompt)]) | llm
    summary = chain.invoke({"content": combined_reviews})

    return summary.content

def compare_reviews_gemini(
    reviews1,
    reviews2,
    KEY: str,
    max_tokens: int = 500,
    temperature: float = 0.3,
    model: str = "gemini-1.5-pro",
):
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=KEY,
    )

    reviews1_text = "\n".join(
        [f"Product 1 - REVIEW: {review[0]['text-message']} (Sentiment: {review[1]['label']}, Score: {review[1]['score']})" for review in reviews1]
    )
    reviews2_text = "\n".join(
        [f"Product 2 - REVIEW: {review[0]['text-message']} (Sentiment: {review[1]['label']}, Score: {review[1]['score']})" for review in reviews2]
    )

    prompt = (
        f"Compare the reviews for the two products below and provide a summary of how they compare in terms of customer satisfaction and feedback:\n\n"
        f"Product 1 Reviews:\n{reviews1_text}\n\n"
        f"Product 2 Reviews:\n{reviews2_text}\n\n"
        f"Please provide a concise summary comparing the overall sentiment and main feedback points from both products."
    )

    chain = ChatPromptTemplate.from_messages([("system", "Compare the reviews:"), ("human", prompt)]) | llm
    comparison_summary = chain.invoke({"content": prompt})

    return comparison_summary.content

def summarize_Emotion_reviews_gemini(
    emotions,
    KEY: str,
    max_tokens: int = 500,
    temperature: float = 0.3,
    model: str = "gemini-1.5-pro",
):
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=KEY,
    )

    combined_emotions = "\n".join(
        [f"TEXT: {emotion[0]['text-message']}\n" + "\n".join(
            [f"Emotion: {emo['label']} (Score: {emo['score']})" for emo in emotion[1]]
        ) for emotion in emotions]
    )

    prompt = f"Summarize the overall sentiment and key points from the reviews:\n{combined_emotions}"

    chain = ChatPromptTemplate.from_messages([("system", "Summarize the reviews:"), ("human", prompt)]) | llm
    summary = chain.invoke({"content": combined_emotions})

    return summary.content

def suggest_reviews_gemini(
    reviews,
    KEY: str,
    max_tokens: int = 500,
    temperature: float = 0.3,
    model: str = "gemini-1.5-pro",
):
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=KEY,
    )

    combined_reviews = "\n".join(
        [f"REVIEW: {review[0]['text-message']} (Sentiment: {review[1]['label']}, Score: {review[1]['score']})" for review in reviews]
    )

    prompt = f"Given the Reviews and Sentiment , generate some suggestions that can improve the product performence\n{combined_reviews}"

    chain = ChatPromptTemplate.from_messages([("system", "You are a suggestion providing Chatbot which analyses all the customer feedback and generates suggestions to improve the products:"), ("human", prompt)]) | llm
    summary = chain.invoke({"content": combined_reviews})

    return summary.content
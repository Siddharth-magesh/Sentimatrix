import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from itertools import chain
import numpy as np

# Flatten the list of lists of dictionaries into a single list of dictionaries
def flatten_reviews(reviews):
    """
    Flattens a nested list of lists of dictionaries into a single list of dictionaries.

    Args:
        reviews (list of lists of dicts): A nested list where each sublist contains dictionaries 
                                          representing sentiment reviews.

    Returns:
        list of dicts: A single list containing all dictionaries from the nested structure.
    """
    return list(chain.from_iterable(reviews))

# Function to plot a bar plot of sentiment labels
def plot_sentiment_distribution(reviews):
    """
    Plots a bar plot showing the distribution of sentiment labels.

    Args:
        reviews (list of lists of dicts): A nested list where each sublist contains dictionaries 
                                          representing sentiment reviews. Each dictionary must have 
                                          a 'label' key.

    Returns:
        None: Displays a bar plot of the sentiment label distribution.
    """
    reviews = flatten_reviews(reviews)  # Flatten the reviews list
    label_counts = Counter([review['label'] for review in reviews])
    
    # Bar Plot
    plt.bar(label_counts.keys(), label_counts.values(), color=['green', 'gray', 'red'])
    plt.title('Distribution of Sentiment Labels')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Reviews')
    plt.show()

# Function to plot a pie chart of sentiment labels
def plot_sentiment_pie_chart(reviews):
    """
    Plots a pie chart showing the proportion of sentiment labels.

    Args:
        reviews (list of lists of dicts): A nested list where each sublist contains dictionaries 
                                          representing sentiment reviews. Each dictionary must have 
                                          a 'label' key.

    Returns:
        None: Displays a pie chart of the sentiment label proportions.
    """
    reviews = flatten_reviews(reviews)  # Flatten the reviews list
    label_counts = Counter([review['label'] for review in reviews])
    
    # Pie Chart
    plt.pie(label_counts.values(), labels=label_counts.keys(), autopct='%1.1f%%', colors=['green', 'gray', 'red'])
    plt.title('Proportion of Sentiment Labels')
    plt.show()

# Function to plot histograms of sentiment scores
def plot_sentiment_histograms(reviews):
    """
    Plots histograms of sentiment scores, separated by sentiment label.

    Args:
        reviews (list of lists of dicts): A nested list where each sublist contains dictionaries 
                                          representing sentiment reviews. Each dictionary must have 
                                          'label' and 'score' keys.

    Returns:
        None: Displays histograms of sentiment scores for each label.
    """
    reviews = flatten_reviews(reviews)  # Flatten the reviews list
    df = pd.DataFrame(reviews)
    
    # Separate by label
    positive_reviews = df[df['label'] == 'positive']['score']
    neutral_reviews = df[df['label'] == 'neutral']['score']
    negative_reviews = df[df['label'] == 'negative']['score']
    
    # Plot histograms, only if the list is non-empty
    if not positive_reviews.empty:
        sns.histplot(positive_reviews, color='green', label='Positive', kde=True, binwidth=0.1)
    
    if not neutral_reviews.empty:
        if len(neutral_reviews) > 1:  # Only apply binwidth if there are enough values
            sns.histplot(neutral_reviews, color='gray', label='Neutral', kde=True, binwidth=0.1)
        else:
            sns.histplot(neutral_reviews, color='gray', label='Neutral', kde=True)
    
    if not negative_reviews.empty:
        if len(negative_reviews) > 1:  # Only apply binwidth if there are enough values
            sns.histplot(negative_reviews, color='red', label='Negative', kde=True, binwidth=0.1)
        else:
            sns.histplot(negative_reviews, color='red', label='Negative', kde=True)
    
    # Add labels and show the plot
    plt.title('Distribution of Sentiment Scores by Label')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

# Function to plot a box plot of sentiment scores
def plot_sentiment_box_plot(reviews):
    """
    Plots a box plot showing the distribution of sentiment scores across different labels.

    Args:
        reviews (list of lists of dicts): A nested list where each sublist contains dictionaries 
                                          representing sentiment reviews. Each dictionary must have 
                                          'label' and 'score' keys.

    Returns:
        None: Displays a box plot of sentiment scores for each label.
    """
    reviews = flatten_reviews(reviews)  # Flatten the reviews list
    df = pd.DataFrame(reviews)
    
    # Box Plot
    sns.boxplot(x='label', y='score', data=df, palette={'positive':'green', 'neutral':'gray', 'negative':'red'})
    plt.title('Sentiment Score Distribution Across Labels')
    plt.show()

# Function to plot a violin plot of sentiment scores
def plot_sentiment_violin_plot(reviews):
    """
    Plots a violin plot showing the distribution and density of sentiment scores across different labels.

    Args:
        reviews (list of lists of dicts): A nested list where each sublist contains dictionaries 
                                          representing sentiment reviews. Each dictionary must have 
                                          'label' and 'score' keys.

    Returns:
        None: Displays a violin plot of sentiment scores for each label.
    """
    reviews = flatten_reviews(reviews)  # Flatten the reviews list
    df = pd.DataFrame(reviews)
    
    # Violin Plot
    sns.violinplot(x='label', y='score', data=df, palette={'positive':'green', 'neutral':'gray', 'negative':'red'})
    plt.title('Violin Plot of Sentiment Scores by Label')
    plt.show()

def calculate_top_emotions_percentages(reviews_emotions):
    # Extract emotions from each review
    all_emotions = [emotion['label'] for review in reviews_emotions for emotion in review[1]]
    
    # Count the frequency of each emotion
    emotion_counts = Counter(all_emotions)
    
    # Get total number of emotions
    total_emotions_count = sum(emotion_counts.values())
    
    # Get top 5 emotions
    top_5_emotions = emotion_counts.most_common(5)
    
    # Calculate percentage for each top emotion
    top_5_emotions_percentages = [(emotion, (count / total_emotions_count) * 100) for emotion, count in top_5_emotions]
    
    # Output results
    return top_5_emotions_percentages


# Function to count sentiments
def count_sentiments(reviews):
    sentiment_count = {'positive': 0, 'neutral': 0, 'negative': 0}
    for review in reviews:
        sentiment = review[1]['label']
        sentiment_count[sentiment] += 1
    return sentiment_count

# Function to plot grouped bar chart
def plot_sentiment_comparison(product1_reviews, product2_reviews):
    # Count sentiments for both products
    product1_sentiments = count_sentiments(product1_reviews)
    product2_sentiments = count_sentiments(product2_reviews)

    # Bar chart data
    labels = ['Positive', 'Neutral', 'Negative']
    product1_counts = [product1_sentiments['positive'], product1_sentiments['neutral'], product1_sentiments['negative']]
    product2_counts = [product2_sentiments['positive'], product2_sentiments['neutral'], product2_sentiments['negative']]

    # Plotting the grouped bar chart
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, product1_counts, width, label='Product 1')
    bars2 = ax.bar(x + width/2, product2_counts, width, label='Product 2')

    # Add labels, title and custom x-axis tick labels
    ax.set_xlabel('Sentiments')
    ax.set_ylabel('Counts')
    ax.set_title('Sentiment Comparison between Product 1 and Product 2')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Show the plot
    plt.show()
import csv
import os

def save_reviews_to_csv(reviews, output_dir, file_name):
    """
    Saves the given reviews and sentiment data as a CSV file in the specified output directory.
    
    Parameters:
    - reviews: List[List[Dict]]
        A list of lists where each sublist contains:
        - A dictionary with the key 'text-message' representing the review content.
        - A dictionary with sentiment data, including 'label' (e.g., 'positive', 'neutral', 'negative') and 'score' (a confidence score).
    
    - output_dir: str
        The directory path where the CSV file should be saved.
    
    - file_name: str
        The name of the CSV file to be created.
    
    Returns:
    - None
    """
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set the output file path
    output_file = os.path.join(output_dir, file_name)
    
    # Define the header for the CSV file
    headers = ["Review", "Sentiment Label", "Sentiment Score"]
    
    # Open the file in write mode
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write the header row
        writer.writerow(headers)
        
        # Write each review, label, and score to the CSV
        for review_data in reviews:
            try:
                if isinstance(review_data, list) and len(review_data) == 2:
                    review_text = review_data[0].get('text-message', 'N/A')
                    sentiment_label = review_data[1].get('label', 'N/A')
                    sentiment_score = review_data[1].get('score', 'N/A')
                    writer.writerow([review_text, sentiment_label, sentiment_score])
                else:
                    print(f"Skipping invalid review format: {review_data}")
            except (IndexError, TypeError, AttributeError) as e:
                print(f"Error processing review: {review_data}, Error: {e}")
                continue

    print(f"Reviews saved to {output_file}")
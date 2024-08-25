import unittest
from unittest.mock import patch, MagicMock
from Sentimatrix.sentiment_generation import SentConfig

class TestSentConfig(unittest.TestCase):

    def setUp(self):
        self.sent = SentConfig(
            Use_Local_Sentiment_LLM=True,
            Use_Local_Scraper=True,
            device_map="auto"
        )

    @patch('Sentimatrix.sentiment_generation.SentConfig.get_Quick_sentiment')
    def test_get_Quick_sentiment(self, mock_get_Quick_sentiment):
        mock_get_Quick_sentiment.return_value = [1, -1, 0]
        sentiments = ["I am very happy", "I am very sad", "I am alright"]
        sentiment_result = self.sent.get_Quick_sentiment(text_message=sentiments, device_map="auto")
        self.assertEqual(sentiment_result, [1, -1, 0])

    @patch('Sentimatrix.sentiment_generation.SentConfig.get_sentiment_from_website_each_feedback_sentiment')
    def test_get_sentiment_from_website_each_feedback_sentiment(self, mock_get_sentiment_from_website_each_feedback_sentiment):
        mock_get_sentiment_from_website_each_feedback_sentiment.return_value = {'feedbacks': [{'text': 'Great product!', 'sentiment': 1}]}
        target = 'https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1'
        result = self.sent.get_sentiment_from_website_each_feedback_sentiment(target_website=target)
        expected_result = {'feedbacks': [{'text': 'Great product!', 'sentiment': 1}]}
        self.assertEqual(result, expected_result)

    @patch('Sentimatrix.sentiment_generation.SentConfig.get_sentiment_from_website_each_feedback_sentiment')
    def test_get_sentiment_from_website_each_feedback_sentiment_with_scraper_api(self, mock_get_sentiment_from_website_each_feedback_sentiment):
        mock_get_sentiment_from_website_each_feedback_sentiment.return_value = {'feedbacks': [{'text': 'Great product!', 'sentiment': 1}]}
        target = 'https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1'
        result = self.sent.get_sentiment_from_website_each_feedback_sentiment(
            target_website=target,
            Use_Scraper_API=True,
            Scraper_api_key=""
        )
        expected_result = {'feedbacks': [{'text': 'Great product!', 'sentiment': 1}]}
        self.assertEqual(result, expected_result)

    @patch('Sentimatrix.sentiment_generation.SentConfig.get_sentiment_from_website_overall_summary')
    def test_get_sentiment_from_website_overall_summary_using_groq(self, mock_get_sentiment_from_website_overall_summary):
        mock_get_sentiment_from_website_overall_summary.return_value = {'summary': 'Overall positive sentiment'}
        target = 'https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1'
        result = self.sent.get_sentiment_from_website_overall_summary(target_website=target)
        expected_result = {'summary': 'Overall positive sentiment'}
        self.assertEqual(result, expected_result)

    @patch('Sentimatrix.sentiment_generation.SentConfig.get_analytical_customer_sentiments')
    def test_get_analytical_customer_sentiments(self, mock_get_analytical_customer_sentiments):
        mock_get_analytical_customer_sentiments.return_value = {
            'bar_chart': 'bar_chart_data',
            'box_plot': 'box_plot_data',
            'histogram': 'histogram_data',
            'pie_chart': 'pie_chart_data',
            'violin_plot': 'violin_plot_data'
        }
        target = 'https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1'
        result = self.sent.get_analytical_customer_sentiments(
            target_website=target,
            Use_Bar_chart_visualize=True,
            Use_box_plot_visualize=True,
            Use_histogram_visualize=True,
            Use_pie_chart_visualize=True,
            Use_violin_plot_visualize=True
        )
        expected_result = {
            'bar_chart': 'bar_chart_data',
            'box_plot': 'box_plot_data',
            'histogram': 'histogram_data',
            'pie_chart': 'pie_chart_data',
            'violin_plot': 'violin_plot_data'
        }
        self.assertEqual(result, expected_result)

    @patch('Sentimatrix.sentiment_generation.SentConfig.get_Sentiment_Audio_file')
    def test_get_Sentiment_Audio_file(self, mock_get_Sentiment_Audio_file):
        mock_get_Sentiment_Audio_file.return_value = {'sentiment': 'positive'}
        audio_path = r'D:\Sentimatrix\tests\voice_datasets-wav\review_1.wav'
        result = self.sent.get_Sentiment_Audio_file(audio_path)
        expected_result = {'sentiment': 'positive'}
        self.assertEqual(result, expected_result)

    @patch('Sentimatrix.sentiment_generation.SentConfig.compare_product_on_reviews')
    def test_compare_product_on_reviews(self, mock_compare_product_on_reviews):
        mock_compare_product_on_reviews.return_value = {'comparison': 'product1 is better'}
        targetsite1 = 'https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1'
        targetsite2 = 'https://www.amazon.in/dp/B0CV9S7ZV6/ref=sspa_dk_detail_0?pd_rd_i=B0CV9S7ZV6'
        result = self.sent.compare_product_on_reviews(
            target_website1=targetsite1,
            target_website2=targetsite2
        )
        expected_result = {'comparison': 'product1 is better'}
        self.assertEqual(result, expected_result)

    @patch('Sentimatrix.sentiment_generation.SentConfig.get_Sentiment_Image_file')
    def test_get_Sentiment_Image_file(self, mock_get_Sentiment_Image_file):
        mock_get_Sentiment_Image_file.return_value = {'sentiment': 'neutral'}
        image_path = r'D:\Sentimatrix\tests\image_datasets\review_1.jpg'
        result = self.sent.get_Sentiment_Image_file(Image_File_path=image_path, Image_to_Text_Model='microsoft/Florence-2-large')
        expected_result = {'sentiment': 'neutral'}
        self.assertEqual(result, expected_result)

    @patch('Sentimatrix.sentiment_generation.SentConfig.Multi_language_Sentiment')
    def test_Multi_language_Sentiment(self, mock_Multi_language_Sentiment):
        mock_Multi_language_Sentiment.return_value = {'sentiment': 'negative'}
        message = 'நான் இந்த தயாரிப்பை வெறுக்கிறேன்'
        result = self.sent.Multi_language_Sentiment(message)
        expected_result = {'sentiment': 'negative'}
        self.assertEqual(result, expected_result)

    @patch('Sentimatrix.sentiment_generation.SentConfig.Config_Local_Scraper')
    def test_Config_Local_Scraper(self, mock_Config_Local_Scraper):
        mock_Config_Local_Scraper.return_value = {'status': 'scraper configured'}
        result = self.sent.Config_Local_Scraper(action='get')
        expected_result = {'status': 'scraper configured'}
        self.assertEqual(result, expected_result)

    @patch('Sentimatrix.sentiment_generation.SentConfig.Save_reviews_to_CSV')
    def test_Save_reviews_to_CSV(self, mock_Save_reviews_to_CSV):
        mock_Save_reviews_to_CSV.return_value = {'status': 'reviews saved'}
        target = 'https://www.amazon.in/ASUS-Battery-i7-13650HX-Windows-G614JU-N3200WS/dp/B0C4TVHMR9?th=1'
        result = self.sent.Save_reviews_to_CSV(
            target_site=target,
            output_dir=r'D:\Sentimatrix\tests\csv_outputs',
            file_name='review.csv'
        )
        expected_result = {'status': 'reviews saved'}
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()

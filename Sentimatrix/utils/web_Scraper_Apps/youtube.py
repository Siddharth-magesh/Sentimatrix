from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

class YouTubeDataFetcher:
    def __init__(self, youtube_api_key):
        # Initialize YouTube API
        self.youtube = build('youtube', 'v3', developerKey=youtube_api_key)

    def search_youtube_videos(self, product_name, max_results=50):
        """
        Search for YouTube videos related to a product and return a list of video details.
        """
        videos = []
        next_page_token = None

        while len(videos) < max_results:
            request = self.youtube.search().list(
                q=product_name,
                part='snippet',
                type='video',
                maxResults=min(max_results - len(videos), 50),
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response['items']:
                video_info = {
                    'video_id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'published_at': item['snippet']['publishedAt'],
                    'channel_title': item['snippet']['channelTitle'],
                    'thumbnail_url': item['snippet']['thumbnails']['high']['url'],
                    'video_url': f"https://www.youtube.com/watch?v={item['id']['videoId']}"
                }
                videos.append(video_info)

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

        return videos

    def get_full_captions(self, video_id):
        """
        Fetch and return the full captions for a given YouTube video ID.
        """
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            full_caption = " ".join([segment['text'] for segment in transcript])
            return full_caption
        except NoTranscriptFound:
            return "No transcript found"
        except TranscriptsDisabled:
            return "Transcripts disabled"
        except Exception as e:
            return f"Error: {str(e)}"

    def get_video_comments(self, video_id, max_results=50):
        """
        Fetch and return the top comments for a given YouTube video ID.
        """
        comments = []
        next_page_token = None

        while len(comments) < max_results:
            try:
                request = self.youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(50, max_results - len(comments)),
                    pageToken=next_page_token
                )
                response = request.execute()

                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    comments.append(comment)

                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break

            except Exception as e:
                print(f"Error fetching comments for video ID {video_id}: {e}")
                break

        return comments

    def fetch_youtube_data_for_product(self, product_name, max_videos=10):
        """
        Fetch YouTube data (videos, captions, comments) for a product and return a single list of comma-separated values.
        """
        videos = self.search_youtube_videos(product_name, max_results=max_videos)

        result = []
        for video in videos:
            # Get captions
            captions = self.get_full_captions(video['video_id'])

            # Get comments
            comments = self.get_video_comments(video['video_id'], max_results=50)

            # Add the first 10 comments to the result
            result.extend(comments[:10])  # Flatten the list by extending the result with comments
            
        return result

# Example usage
if __name__ == "__main__":
    # YouTube API key
    youtube_api_key = 'AIzaSyDdGyupUJqws-7toxs4bSBUfAT0BoMzrb0'

    product_name = 'Oneplus 12'

    # Initialize the YouTubeDataFetcher class
    fetcher = YouTubeDataFetcher(youtube_api_key)

    # Fetch combined data from YouTube
    data = fetcher.fetch_youtube_data_for_product(product_name, max_videos=10)

    # Output the single list of all comments
    print(data)  # This will print a flattened list of comments

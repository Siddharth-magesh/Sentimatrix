import speech_recognition as sr

def audio_to_text(wav_file):
    """
    Transcribes audio from a WAV file to text using Google's Web Speech API.

    Args:
        wav_file (str): Path to the WAV file containing the audio to be transcribed.

    Returns:
        str: The transcribed text if successful, or an error message if the transcription fails.
    """
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(wav_file) as source:
        # Listen to the file and store the audio data
        audio_data = recognizer.record(source)
        
        try:
            # Use Google Web Speech API to transcribe the audio to text
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            # Error if the audio is unintelligible
            return "Google Speech Recognition could not understand the audio"
        except sr.RequestError as e:
            # Error if there is an issue with the request to the Google API
            return f"Could not request results from Google Speech Recognition service; {e}"
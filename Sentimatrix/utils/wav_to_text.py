import speech_recognition as sr

def audio_to_text(wav_file):
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(wav_file) as source:
        # Listen to the file
        audio_data = recognizer.record(source)
        
        try:
            # Use Google Web Speech API to transcribe the audio to text
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand the audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"

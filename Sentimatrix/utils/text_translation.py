from deep_translator import GoogleTranslator

def Translate_text(message):
    """
    Translates the input text to the target language using the GoogleTranslator from the deep_translator library.

    Args:
        message (str): The text message to be translated.

    Returns:
        str: The translated text.
    """
    # Initialize the translator with automatic detection of the source language
    # and specify the target language (set to 'en' for English here, can be customized).
    translator = GoogleTranslator(source='auto', target='en')

    # Perform the translation
    result = translator.translate(message)
    
    return result
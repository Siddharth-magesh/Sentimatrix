from deep_translator import GoogleTranslator

def Translate_text(message):
    translator = GoogleTranslator(source='auto', target='en')  # Set target language as needed
    result = translator.translate(message)
    return result

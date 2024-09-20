from transformers import pipeline

def Struct_Emotion(
    text_message,
    Use_Local_Emotion_LLM,
    model_id="SamLowe/roberta-base-go_emotions",
    device_map="auto"
):
    if Use_Local_Emotion_LLM==True:
        classifier = pipeline(
            task="text-classification", 
            model=model_id, 
            top_k=None, 
            device_map=device_map
        )
        if isinstance(text_message, str):
            model_outputs = classifier(text_message,truncation=True, max_length=512)
            top_5_outputs = sorted(model_outputs[0], key=lambda x: x['score'], reverse=True)[:5]
            formatted_output = [{'text-message': text_message},top_5_outputs]
            return formatted_output
        elif isinstance(text_message, list) and all(isinstance(item, str) for item in text_message):
            result_array = []
            for message in text_message:
                model_outputs = classifier(message,truncation=True, max_length=512)
                top_5_outputs = sorted(model_outputs[0], key=lambda x: x['score'], reverse=True)[:5]
                formatted_output = [{'text-message': message},top_5_outputs]
                result_array.append(formatted_output)
            return result_array
        else:
            return "Error: The accepted format is either a list of strings or a single string."
    else:
        print("Set the Use_Local_Emotion_LLM to True")
        exit()
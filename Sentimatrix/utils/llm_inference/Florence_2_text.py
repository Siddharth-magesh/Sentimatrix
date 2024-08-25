from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image
import torch

def Convert_Image_to_Text(image_path, task_prompt='<MORE_DETAILED_CAPTION>', text_input=None):
    """
    Converts an image to text using a pre-trained model from Microsoft.

    Args:
        image_path (str): Path to the image file to be processed.
        task_prompt (str, optional): Prompt specifying the type of caption to generate. Defaults to '<MORE_DETAILED_CAPTION>'.
        text_input (str, optional): Additional text input to be appended to the prompt.

    Returns:
        str: The generated text caption for the image.
    """
    # Load the model and processor from the pre-trained Microsoft Florence model
    model_id = 'microsoft/Florence-2-large'
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # Open and preprocess the image
    image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format

    # Validate and set the task prompt
    if task_prompt not in ['<MORE_DETAILED_CAPTION>','<CAPTION>','<DETAILED_CAPTION>']:
        print("Invalid Custom Prompt, Default Value '<MORE_DETAILED_CAPTION>' Assigned")
        task_prompt = '<MORE_DETAILED_CAPTION>'

    # Prepare the input prompt
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    # Process inputs for the model
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)

    # Generate the text caption for the image
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )

    # Decode the generated text and process it
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )

    return parsed_answer
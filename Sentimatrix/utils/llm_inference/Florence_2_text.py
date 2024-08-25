from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image
import torch

def Convert_Image_to_Text(image_path, task_prompt='<MORE_DETAILED_CAPTION>', text_input=None):
    # Load and preprocess the image
    model_id = 'microsoft/Florence-2-large'
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    image = Image.open(image_path).convert("RGB")  # Convert image to RGB

    if task_prompt not in ['<MORE_DETAILED_CAPTION>','<CAPTION>','<DETAILED_CAPTION>']:
        print("Invalid Custom Prompt, Default Value '<MORE_DETAILED_CAPTION>' Assigned")
        task_prompt = '<MORE_DETAILED_CAPTION>'


    # Prepare the input prompt
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    # Process inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)

    # Generate the caption
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )

    # Decode and post-process the generated text
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )

    return parsed_answer
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


def generate_image_caption(image_path, text_prompt=None):
    """
    Generates a caption for an image using a pre-trained BLIP model.

    Args:
        image_path (str): Path to the image file to be processed.
        text_prompt (str, optional): A custom prompt for generating a detailed description. 
                                     If not provided, the captioning will be unconditional.

    Returns:
        str: The generated text caption for the image.
    """
    # Load the BLIP processor and model
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large").to("cuda")

    # Open and preprocess the image
    raw_image = Image.open(image_path).convert('RGB')

    # Prepare inputs based on whether a text prompt is provided
    if text_prompt:
        inputs = processor(raw_image, text_prompt,
                           return_tensors="pt").to("cuda")
    else:
        inputs = processor(raw_image, return_tensors="pt").to("cuda")

    # Generate the caption
    output = model.generate(
        **inputs,
        max_length=200,           # Increase maximum length for longer output
        num_beams=5,              # Use beam search for better quality
        repetition_penalty=1.5,   # Reduce repetition
        do_sample=True,           # Allow sampling to increase creativity
        temperature=0.7,          # Control the randomness
        top_p=0.9                 # Nucleus sampling for more diverse output
    )

    # Decode and return the generated caption
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

import requests
from io import BytesIO
import base64
import time
import json
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

def extract_key_elements(story_segment):
    """Extract key visual elements from the story segment."""
    lines = story_segment.split('\n')
    elements = {
        'scene': [],
        'character_actions': [],
        'atmosphere': [],
        'objects': [],
        'environment': [],
        'characters': []
    }
    
    for line in lines:
        line = line.lower()
        # Scene descriptions
        if any(x in line for x in ["scene", "setting", "chamber", "temple", "valley", "marketplace", "stones", "portal"]):
            elements['scene'].append(line)
        
        # Character actions and descriptions
        if any(name.lower() in line for name in ["alex", "morgan", "jamie"]):
            elements['characters'].append(line)
            if any(x in line for x in ["stands", "moves", "reveals", "discovers", "examines", "faces", "approaches", "searches"]):
                elements['character_actions'].append(line)
            
        # Atmosphere and magical elements
        if any(x in line for x in ["mood", "tension", "air", "atmosphere", "energy", "pulse", "glow", "shimmer", "magic"]):
            elements['atmosphere'].append(line)
            
        # Notable objects and artifacts
        if any(x in line for x in ["map", "artifact", "runes", "symbols", "mechanism", "stone", "markings"]):
            elements['objects'].append(line)
            
        # Environment and setting details
        if any(x in line for x in ["walls", "light", "shadows", "mist", "clouds", "spices", "aromas", "alleys", "passage"]):
            elements['environment'].append(line)
    
    return elements

def construct_image_prompt(elements):
    """Construct a detailed prompt for image generation."""
    prompt_parts = []
    
    # Start with the main scene
    if elements['scene']:
        prompt_parts.append(elements['scene'][0])
    
    # Add character descriptions and actions
    if elements['characters']:
        prompt_parts.append(elements['characters'][0])
    if elements['character_actions']:
        prompt_parts.append(elements['character_actions'][0])
    
    # Add environment and atmosphere
    if elements['environment']:
        prompt_parts.append(elements['environment'][0])
    if elements['atmosphere']:
        prompt_parts.append(elements['atmosphere'][0])
    
    # Add important objects
    if elements['objects']:
        prompt_parts.append(elements['objects'][0])
    
    # Base style and quality specifications with enhanced fantasy elements
    style_prompt = (
        "epic fantasy digital art, cinematic lighting, highly detailed, 4k UHD, "
        "professional fantasy artwork, dramatic composition, volumetric lighting, "
        "atmospheric perspective, golden hour lighting, intricate details, "
        "mystical atmosphere, ethereal glow"
    )
    
    # Combine all elements
    main_prompt = " ".join(prompt_parts)
    
    # Clean up the prompt
    clean_prompt = main_prompt.replace("the scene unfolds:", "")
    clean_prompt = clean_prompt.replace("the mood is", "with atmosphere:")
    clean_prompt = ' '.join(clean_prompt.split())  # Remove extra spaces
    
    # Add specific style elements based on the scene content
    if "marketplace" in clean_prompt.lower():
        style_prompt += ", vibrant colors, bustling crowd, exotic architecture"
    elif "stones" in clean_prompt.lower():
        style_prompt += ", ancient monoliths, mystical runes, ethereal energy"
    elif "temple" in clean_prompt.lower():
        style_prompt += ", sacred architecture, ancient wisdom, mystical symbols"
    
    return f"{clean_prompt}, {style_prompt}"

def generate_visual(story_segment):
    """Generate a visual representation of the story segment using Gemini 2.0 Flash Preview Image Generation, with fallback to Gemma 3n E4B."""
    def try_model(model_name, prompt):
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                contents=prompt,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 1,
                    "top_k": 32,
                }
            )
            image_data = None
            for part in response.parts:
                if hasattr(part, 'image'):
                    image_data = part.image.data
            if image_data:
                return f"data:image/png;base64,{image_data}"
        except Exception as e:
            logger.error(f"Image generation failed with {model_name}: {e}")
        return None

    # Use only the scene description or a summary for the image prompt
    elements = extract_key_elements(story_segment)
    visual_prompt = elements['scene'][0] if elements['scene'] else story_segment.split('\n')[0]
    style_prompt = (
        "cinematic, highly detailed, digital art, atmospheric, dramatic lighting, vivid colors, 4k, concept art"
    )
    prompt = f"{visual_prompt}, {style_prompt}"

    # Try primary model
    img = try_model('gemini-2.0-flash-preview-image-generation', prompt)
    if img:
        return img
    # Fallback to backup model
    img = try_model('gemma-3n-e4b-it', prompt)
    if img:
        return img
    # Fallback to placeholder
    return "/static/placeholder.png"

    # The AI image generation code is temporarily disabled
    # We'll enable it once we fix the PIL installation
    """
    prompt = f"Create an artistic scene: {story_segment}"
    try:
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(
            contents=prompt,
            generation_config={
                "temperature": 0.9,
                "top_p": 1,
                "top_k": 32,
            }
        )
        
        for part in response.parts:
            if hasattr(part, 'image'):
                return part.image.data
    except Exception as e:
        print("Image generation failed:", e)
    return None
    """

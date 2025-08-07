import torch
from transformers import CLIPProcessor, CLIPModel

# Global cache for models and processors
_model_cache = {}
_processor_cache = {}

def get_text_features(text, clip_name="openai/clip-vit-large-patch14", device=None):
    """
    Extract text features using CLIP model.
    
    Args:
        text (str): Input text to extract features from
        clip_name (str): CLIP model name to use
        device (str, optional): Device to run the model on. If None, uses CUDA if available, else CPU
        
    Returns:
        torch.Tensor: Normalized text features
    """
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Cache model and processor to avoid reloading
    cache_key = f"{clip_name}_{device}"
    
    if cache_key not in _model_cache:
        _processor_cache[cache_key] = CLIPProcessor.from_pretrained(clip_name)
        _model_cache[cache_key] = CLIPModel.from_pretrained(clip_name).to(device)
    
    processor = _processor_cache[cache_key]
    model = _model_cache[cache_key]
    
    # Process text and extract features
    inputs = processor(text=text, return_tensors="pt")
    text_embeds = model.get_text_features(input_ids=inputs['input_ids'].to(device))
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    
    return text_embeds.detach()
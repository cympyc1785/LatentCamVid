from open_clip import create_model_from_pretrained, get_tokenizer
import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
from urllib.request import urlopen
from PIL import Image

def text_global_pool(
        x: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        pool_type: str = 'argmax',
        eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    if pool_type == 'first':
        pooled = x[:, 0]
    elif pool_type == 'last':
        pooled = x[:, -1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled = x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)]
    elif pool_type == 'eos':
        # take features from tokenizer specific eos
        assert text is not None
        assert eos_token_id is not None
        idx = (text == eos_token_id).int().argmax(dim=-1)
        pooled = x[torch.arange(x.shape[0], device=x.device), idx]
    else:
        pooled = x

    return pooled

def get_model(repo_name, device="cuda:0"):
    clip, _ = create_model_from_pretrained(repo_name)
    tokenizer = get_tokenizer(repo_name)
    clip = clip.to(device)
    return clip, tokenizer

def encode_text_clip(model, tokenizer, text, normalize=False, device="cuda:0"):
    """
    model: CLIP-Family
    text: List of str
    """
    cast_dtype = model.transformer.get_cast_dtype()
    x = tokenizer(text, context_length=model.context_length)
    # y = tokenizer.tokenizer(text, padding='max_length', max_length=model.context_length, return_tensors="pt", return_attention_mask=True, add_special_tokens=False)
    # input_ids = y['input_ids']
    # attn_mask = y['attention_mask']
    x = x.to(device)
    x = model.token_embedding(x) # [batch_size, n_ctx, d_model]

    x = x + model.positional_embedding.to(cast_dtype)
    x = model.transformer(x, attn_mask=model.attn_mask)
    x_seq = model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
    x = text_global_pool(x_seq, text, model.text_pool_type, eos_token_id=getattr(model, "text_eos_id", None))
    if model.text_projection is not None:
        if isinstance(model.text_projection, nn.Linear):
            x = model.text_projection(x)
            x_seq = model.text_projection(x_seq)
        else:
            x = x @ model.text_projection
            x_seq = x_seq @ model.text_projection

    x_tok = F.normalize(x, dim=-1) if normalize else x
    x_tok = x_tok.unsqueeze(1) # [batch_size, 1, feat_dim] 
    x_tok_masks = torch.ones(x_tok.shape[:2], device=x_tok.device).bool()
    return x_tok, x_tok_masks

if __name__ == '__main__':
    repo_name = 'hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP'
    model, tokenizer = get_model(repo_name, device="cuda:0")

    text = ["The camera pans left and moves forward, capturing a wide view of the basketball court and surrounding trees."]
    x_seq, x_tok, decoded_toks = encode_text(model, tokenizer, text)

    image = Image.open('/data1/cympyc1785/SceneData/DL3DV/scenes/1K/006771db3c057280f9277e735be6daa24339657ce999216c38da68002a443fed/images/frame_00001.png')
    _, preprocess = create_model_from_pretrained(repo_name)
    image = preprocess(image).unsqueeze(0).cuda()
    image_features = model.encode_image(image)
    image_features = F.normalize(image_features, dim=-1)
    print(image_features.shape)
    text_features = F.normalize(x_tok[:, 0, :], dim=-1)
    text_probs = (100.0 * image_features @ text_features.T)
    print(text_probs)
    text_features = F.normalize(x_seq[0, :, :], dim=-1)
    text_probs = (100.0 * image_features @ text_features.T)
    assert len(decoded_toks) == len(text_probs[0])
    [print(decoded_toks[idx], text_probs[0][idx]) for idx in range(len(decoded_toks))]
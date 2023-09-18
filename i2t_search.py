import requests
from PIL import Image
from t2t_search import get_most_similar_indices


def get_IT_embeds(clip, clip_processor, images, text):
    inputs = clip_processor(text=text, images=images, return_tensors="pt", padding=True)

    outputs = clip(**inputs)
    # logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    # probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

    text_embeds = outputs.text_embeds.detach().numpy()
    image_embeds = outputs.image_embeds.detach().numpy()

    return image_embeds, text_embeds # , probs


def get_text_embeds(clip, clip_processor, text):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = clip_processor(text=text, images=[image], return_tensors="pt", padding=True)

    outputs = clip(**inputs)
    return outputs.text_embeds.detach().numpy()
    


def get_probs(text_embeds, image_embeds):
    logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
    logits_per_image = logits_per_text.t()
    probs = logits_per_image.softmax(dim=1)
    return probs


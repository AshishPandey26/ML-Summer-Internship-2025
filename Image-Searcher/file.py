import torch
import clip
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)

def image_to_embedding(img):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_input = preprocess(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(img_input)
    return embedding / embedding.norm(dim=-1, keepdim=True)

def sliding_window(image, step, window_size):
    for y in range(0, image.shape[0] - window_size[1], step):
        for x in range(0, image.shape[1] - window_size[0], step):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# Load images
# input_img = cv2.imread("vk.png")
input_img = cv2.imread("maxwell.png")
target_img = cv2.imread("rcb_team.png")

input_h, input_w = input_img.shape[:2]

# Get embedding for input image
input_embed = image_to_embedding(input_img)

# Slide over target image
step = 20
max_score = -1
best_loc = (0, 0)

for (x, y, window) in sliding_window(target_img, step, (input_w, input_h)):
    if window.shape[0] != input_h or window.shape[1] != input_w:
        continue
    window_embed = image_to_embedding(window)
    score = cosine_similarity(input_embed.cpu(), window_embed.cpu())[0][0]
    if score > max_score:
        max_score = score
        best_loc = (x, y)

# Draw rectangle around best match
output_img = target_img.copy()
cv2.rectangle(output_img, best_loc, (best_loc[0]+input_w, best_loc[1]+input_h), (0, 255, 0), 2)
cv2.imwrite("output.jpg", output_img)
print("Done! Best match score:", max_score)

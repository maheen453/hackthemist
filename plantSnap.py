from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
feature_extractor = ViTFeatureExtractor.from_pretrained('maheen453/invasive-species')
model = ViTForImageClassification.from_pretrained('maheen453/invasive-species')

with open('wis.jpg', 'rb') as file:
    image = Image.open(file)

    output = feature_extractor(image, return_tensors='pt')['pixel_values']

    sm = torch.nn.Softmax(dim=1)
    logits = model(output).logits

    result1 = sm(logits)[0].tolist()
    result = [round(x, 2) for x in result1]

    print(result)
plant_species = ['Barberry', 'English_Ivy', 'Phragmites', 'Purple_Loosestrife', 'Wisteria']
max_probability = 0.00

max_index = torch.argmax(logits, dim=1)
max_probability = sm(logits)[0, max_index].item()
for i, x in enumerate(result):
    if all(x < 0.50 for x in result):
        print("This plant does not belong to any of the 5 invasive species")
    elif x >= max_probability:
        max_probability = x


plant_species = plant_species[max_index]
print("The most probable plant species is {} with a probability of {}.".format(plant_species, max_probability))

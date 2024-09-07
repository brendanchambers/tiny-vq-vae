from datasets import load_dataset
from diffusers import VQModel


image_dataset = "jbarat/plant_species"
ds = load_dataset(image_dataset)
print(ds)

# sanity
print(ds['train'][0]['image'])  # {'image':, 'label':}
ds['train'][42]['image'].show()


# tokenize the dataset using a pretrained vqvae model
model = VQModel()



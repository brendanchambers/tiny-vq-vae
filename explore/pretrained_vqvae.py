from datasets import load_dataset
from diffusers import VQModel

# from timm.data.transforms_factory import create_transform
# from torchvision import transforms
import torch
from torchvision.transforms import v2
from dataclasses import dataclass

####################
# define some default values to get an exploration working

@dataclass
class QuickArgs:
    hresolution: int = 64
    wresolution: int = 64
    center_crop: float = None
    random_flip: bool = True

args = QuickArgs()
print(args)
print(f"x res: {args.wresolution}  y res: {args.hresolution}")


#####################
# quick arbitrary example to learn about typical preprocessing steps

image_dataset = "jbarat/plant_species"
dataset = load_dataset(image_dataset)
print(dataset)

# sanity
print("dataset on load...")
print(dataset['train'][0]['image'])  # {'image':, 'label':}
# ds['train'][42]['image'].show()
#    size of the plants images: 
        # 375 x 50


####### preprocess
# random augmentation taken from train_vqgan diffusers example

# Preprocessing the datasets.
example_transforms = v2.Compose([
    v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
    # v2.Resize(args.resolution, interpolation=v2.InterpolationMode.BILINEAR),
    v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
    # ...
    v2.RandomResizedCrop(size=(args.wresolution, args.hresolution), antialias=True),  # Or Resize(antialias=True)
    # ...
    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print('apply preprocessing transformations...')
print(dataset['train'][0])
dataset['train'][0]['image'].show()

train_dataset = dataset["train"].with_transform(example_transforms)   # hmmm todo look cloesr at this

print('after preprocessing...')
print(train_dataset)
print(train_dataset[0]['image'].size())
pil_image = v2.functional.to_pil_image(train_dataset[0]['image'])
pil_image.show()



######## inference

# tokenize the dataset using a default init'd vqvae model
model = VQModel()
# print('\n\n\n\n')
# print(model)
#     size of the model inputs: 64

i_img = 0
img = train_dataset[0]['image']
# outs = model(img)
# print(outs)
# print(type(outs))


'''
, tuple of (int, int), tuple of (int, int), tuple of (int, int), int)
 * (Tensor input, Tensor weight, Tensor bias = None, tuple of ints stride = 1, str padding = "valid", tuple of ints dilation = 1, int groups = 1)
      didn't match because some of the arguments have invalid types: (JpegImageFile, Parameter, Parameter, tuple of (int, int), tuple of (int, int), tuple of (int, int), int)
'''
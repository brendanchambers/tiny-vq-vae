# 0th learning attempt at vqvae 
# inference a randomly init'd vqvae on an example plant species dataset

from datasets import load_dataset
from diffusers import VQModel

# from timm.data.transforms_factory import create_transform
# from torchvision import transforms
import torch
from torchvision.transforms import v2
from dataclasses import dataclass
import time

####################
# define some default values to get an exploration working

@dataclass
class Quickconfig:
    hresolution: int = 64
    wresolution: int = 64
    center_crop: float = None
    random_flip: bool = True
    bsz: int = 128
    device: str = "cuda:0"

config = Quickconfig()
print(config)
print(f"x res: {config.wresolution}  y res: {config.hresolution}")


#####################
# quick arbitrary example to learn about typical preprocessing steps

torch.cuda.empty_cache()

image_dataset = "jbarat/plant_species"
dataset = load_dataset(image_dataset)
print(dataset)

# sanity
print("dataset on load...")
print(dataset['train'][0]['image'])  # {'image':, 'label':}
# ds['train'][42]['image'].show()
#    size of an example plants image: 
        # 375 x 50


####### preprocess
# random augmentation taken from train_vqgan diffusers example

# Preprocessing the datasets.
example_transforms = v2.Compose([
    v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
    v2.Resize((config.wresolution, config.hresolution), interpolation=v2.InterpolationMode.BILINEAR),   
    v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
    # ...
    v2.RandomResizedCrop(size=(config.wresolution, config.hresolution), antialias=True),  # Or Resize(antialias=True)  (todo how does this interact with resize?)
    # ...
    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print('apply preprocessing transformations...')
# print(dataset['train'][0])
# dataset['train'][0]['image'].show()
train_dataset = dataset["train"].with_transform(example_transforms)   # hmmm todo look closer at this

print('after preprocessing...')
# print(train_dataset)
print(train_dataset[0]['image'].size())
print(train_dataset[1]['image'].size())
pil_image = v2.functional.to_pil_image(train_dataset[0]['image'])
print(train_dataset[0]['image'].size())
print(train_dataset[1]['image'].size())
# pil_image.show()


######## inference

# tokenize the dataset using a default init'd vqvae model
model = VQModel().to(config.device)
print(f"\n\n\n\n{model}\n\n\n\n")
#     size of the model inputs: 64



def visualize_outs(outs, show=True):
    """ warning this will fail for batched outputs """
    print(outs)
    print(type(outs))
    print([k for k in outs])  # sample, commit_loss

    print("sample and commit_loss size:")
    print(f" sample dimensions: {outs['sample'].size()}")
    print(f" commit_loss: {outs['commit_loss']}")

    if show:
        print('reconstructed sample:')
        v2.functional.to_pil_image(outs['sample'].squeeze()).show()

########### sanity chceck

i_img = 42
img = train_dataset[i_img]['image']
batched_img = torch.unsqueeze(img, 0)  # add a batch dimension 
print(img.size())
print(batched_img.size())
outs = model(batched_img.to(config.device))
print('=============================')

try:
    
    visualize_outs(outs[0,:])  # show the sample and loss
                # tuple indexig error here - todo fix or delete
except:
    print('warning: todo fix visualize_outs function')


######################################

def yield_batches(itr, n=1):
    l = len(itr)
    for ndx in range(0, l, n):
        yield itr[ndx:min(ndx + n, l)]

def pack_batch(batch):

    # handle singleton batch
    if config.bsz==1:
        image = batch['image'][0]
        batched_tensor = image.unsqueeze(dim=0)
        batched_tensor = batched_tensor.to(config.device)
    else:
        images = batch['image']
        batched_tensor = torch.stack(images, dim=0)
        batched_tensor = batched_tensor.to(config.device)

    return batched_tensor

################## run inference over the dataset

t_s = time.time()

for i_batch, batch_entries in enumerate(yield_batches(train_dataset, n=config.bsz)):
    print(f"i batch: {i_batch}")
    batched_tensor = pack_batch(batch_entries)

    with torch.inference_mode():
        outs = model(batched_tensor)  # todo - VQ model likely has a n initializaiton problem

t_e = time.time()
print(f"elapsed (s): {t_e - t_s}")
print(f"num images: {len(train_dataset)}")
print(f"batchsize: {config.bsz}")
print(f"images / s: {len(train_dataset) / (t_e - t_s)}")   # ~11 img/s bsz=1 cpu tiny-laptop
                                                            # ~14 imgs/s bsz=2 cpu tiny-laptop
                                                            # ~39.9 second measurement; ~71 img/s bsz=2 gpu GeForce 1050 Ti 
                                                            # ~43 img/s bsz=32 gtx 1050 ti moble
                                                            #   performance seems to be dropping, may be heat or power limit
                                                            # 128 similar, bsz 1024 oom


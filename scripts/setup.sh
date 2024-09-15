

# notes on initial explorations

# currently working on local terminal, no gpu, tmp env
project_dir=/home/brch/personal_projects/github/brendanchambers/tiny-vq-vae
cd $project_dir
conda create -n vqvae python=3.9
source activate ~/anaconda3/envs/vqvae

pip install datasets
pip install Pillow
pip install --upgrade diffusers[torch]
pip install torchvision
pip install matplotlib

mkdir -p $project_dir/data/incoming

echo "data/**\nmodels/**\ncredentials/**\n" > $project_dir/.gitignore



# quick env management

# export env
conda activate pytorch-env
conda env export --no-builds --from-history | grep -v "prefix" > environment.yml

# restore env on new vm
conda create -n pytorch-env --file environment.yml
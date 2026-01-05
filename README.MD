# Project
This repo contains the CoPhaser model for single cell phase inference from scRNA-seq.

# Installation
0) Download this repo
1) Create a [conda environment](https://www.anaconda.com/docs/getting-started/miniconda/main)
```bash
conda create -n "CoPhaser_Env" python=3.13
conda activate CoPhaser_Env
```
2) Install PyTorch with CUDA support (if you want to use the GPU)
   **Note:** Check your CUDA driver version (`nvidia-smi`) and install the compatible PyTorch version. 
   Go to [the official page](https://pytorch.org/) for more information. The code was tested using the version 2.9.1.
   For instance for CUDA 12.6, with the latest PyTorch version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
3) Install the package
```bash
cd CoPhaser/
pip install .
```

# Usage
You can find a notebook as example.

1) Initialize a model, passing the cycling gene names, the context gene names (usually the 2000 most variable genes) and the dimension of z and the number of harmonics used for the rhythmic reconstruction.
```python
from CoPhaser import CoPhaser, Trainer, Loss
from CoPhaser import utils
from CoPhaser import gene_sets

# Define your gene sets for example
# rhythmic_genes = gene_sets.SMALL_CELL_CYCLE_GENE_SET
# variable_genes = utils.get_variable_genes(adata)

model = CoPhaser(rhythmic_gene_names=rhythmic_genes, context_genes=variable_genes, n_latent=10, n_harm=3)
```

2) Load the anndata containing the counts data
```python
model.load_anndata(adata, layer_to_use="spliced")
```

3) Initialize a trainer object with the desired weights of the different loss objects
```python
trainer = Trainer(model, Loss.compute_loss, noise_model="NB")
The model can be sensitive to the hyperparameters, a description of their impact and recommendation to set them can be found in the class description.
```

4) Train the model
```python
trainer.train_model(n_epochs=200, lr=1e-2, device="cuda", batch_size=2048, )
The batch_size can impact the training since it has a direct impact on the entropy inference. Usually set around 1/10th of the dataset.
```

5) Get for instance the inferred phases and context space
```python
model.to("cpu")
generative_outputs, space_outputs = model.get_outputs()
thetas = space_outputs["theta"].detach().numpy()
z_space = space_outputs["z"].detach().numpy()
```
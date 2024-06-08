# [Multivector Neurons: Better and Faster O(n)-Equivariant Clifford GNNs](https://arxiv.org/abs/2406.04052) 

## Abstract
Recent works have focused on designing deep learning models that are equivariant to the O(n) or SO(n) groups. These models either consider only scalar information such as distances and angles or have a very high computational complexity. In this work, we test a few novel message passing graph neural networks (GNNs) based on Clifford multivectors, structured similarly to prevalent equivariant models in geometric deep learning. Our approach leverages efficient invariant scalar features while simultaneously performing expressive learning on multivector representations, particularly through the use of the equivariant geometric product operator. By integrating these elements, our methods outperform established efficient baseline models on an N-Body simulation task and protein denoising task while maintaining a high efficiency. In particular, we push the state-of-the-art error on the N-body dataset to 0.0035 (averaged over 3 runs); an 8% improvement over recent methods.

## Requirement and Conda Environment
```
conda create -n mvn
conda activate mvn
conda install python=3.10
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install pyyaml wandb
conda install -c conda-forge openmm=8.1.1
pip install sidechainnet
```

Use following commands to set up the folder.
```
cd lib/engineer/
pip install -e .
cd ../../mvn
```

## Code Organization
* `mvn/`: contains the core code snippets.
  * `algebra/`: contains the Clifford Algebra implementation.
  * `configs/`: contains the configuration files. 
  * `data/`: contains necessary (simplicial) data modules.
  * `models/`: contains model and layer implementations.
* `engineer/`: contains the training and evaluation scripts.
* `lib/` contains commands and set up

## Datasets and Experiments

### Datasets
Run `mkdir ./datasets/` to generate a folder for storing datasets of experiments

#### NBody
Download / Generate nbody datasets and move to `./datasets/`

### Experiments
This implementation uses conda environment, change the path of `miniconda/` in `activate.sh` to your local `miniconda/` path and run `sh ../activate.sh`.

#### Instruction to run sweep_local for MVN on pNBody simulation task
* MVN: ```sweep_local configs/nbody_mvn.yaml```
* GVP: ```sweep_local configs/nbody_gvp.yaml```
* CVP: ```sweep_local configs/nbody_cvp.yaml```
* Clifford_EGNN: ```sweep_local configs/nbody_clifford_egnn.yaml```
* EGNN: ```sweep_local configs/nbody_egnn.yaml```


#### Instruction to run sweep_local for MVN on protein structure denoising task
* MVN: ```sweep_local configs/denoise_mvn.yaml```
* GVP: ```sweep_local configs/denoise_gvp.yaml```
* CVP: ```sweep_local configs/denoise_cvp.yaml```
* Clifford_EGNN: ```sweep_local configs/denoise_clifford_egnn.yaml```
* EGNN: ```sweep_local configs/denoise_egnn.yaml```

## Citation:
If you found this code useful, please cite our paper:

```
@misc{liu2024multivector,
      title={Multivector Neurons: Better and Faster O(n)-Equivariant Clifford Graph Neural Networks}, 
      author={Cong Liu and David Ruhe and Patrick Forré},
      year={2024},
      eprint={2406.04052},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

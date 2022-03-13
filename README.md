# 3D_PRoteins_Params

Execute from the directory you want the repo to be installed:

```
git clone https://github.com/aniton/3D_PRoteins_Params.git
cd 3D_PRoteins_Params
pip3 install -e .
pymol -R
```
Stylize from scratch:

```
python3 -m src.solver --compare_method ['ssim', 'neural']
```
Opimize existing parameters:

```
python3 -m src.reoptimize_solver --compare_method ['ssim', 'neural']
```

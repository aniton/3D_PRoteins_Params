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
python3 -m src.solver --solver solver  --compare_method ['ssim', 'neural'] --protein 1lmp --representation lines  --style_image_path st.png
```

Opimize existing parameters:

```
python3 -m src.solver --solver optimizer  --compare_method ['ssim', 'neural'] --protein 1lmp --representation lines  --style_image_path st.png
```

Try autoretrain with optimizer:

```
./src/autoretrain.sh
```

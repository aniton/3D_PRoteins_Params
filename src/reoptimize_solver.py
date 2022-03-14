import argparse
import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import xmlrpc.client as xlmrpclib
import torchvision.transforms as transforms
from ipymol import viewer as pymol
import numpy as np
from PIL import Image
from SSIM_PIL import compare_ssim
from itertools import product
from torchvision.utils import save_image
from typing import List, Tuple, TypedDict, Literal, Union, Dict, Optional
from .style_model import calculate_loss, weights_init_uniform_rule, image_loader, VGG
from .utils import render_image, merge_two_dicts, split_dict


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PymolTexture_opt:
    def __init__(
        self,
        compare_method: Literal["ssim", "neural"],
        style: Image,
        protein: str,
        repres: Literal["spheres", "sticks", "lines", "ribbons", "cartoon", "dots"],
        params: Dict,
        model: nn.Module,
    ) -> None:
        self.params = params
        self.protein = protein
        self.style = style
        self.model = model
        self.repres = repres
        self.method = compare_method

    """
    Arguments:
        style: image of the desired style
        protein: pdb name of the desired protein
        repres: representation
        params: EXISTING_PARAMS
        model: style transferring model
    """

    def run(self):
        text_params, common_params = split_dict(self.params)
        if self.repres == "spheres":
            texture_params = {
                "sphere_scale": np.arange(
                    text_params["sphere_scale"] - 0.1,
                    text_params["sphere_scale"] + 0.4,
                    0.1,
                ),
                "sphere_transparency": [text_params["sphere_transparency"]],
                "sphere_mode": [text_params["sphere_mode"]],
                "sphere_solvent": [text_params["sphere_solvent"]],
                "cull_spheres": [text_params["cull_spheres"]],
            }
        if self.repres == "sticks":
            texture_params = {
                "stick_radius": np.arange(
                    text_params["stick_radius"] - 0.1,
                    text_params["stick_radius"] + 0.4,
                    0.1,
                ),
                "stick_fixed_radius": [text_params["stick_fixed_radius"]],
                "stick_nub": [text_params["stick_nub"]],
                "stick_transparency": np.arange(
                    text_params["stick_transparency"] - 0.1,
                    text_params["stick_transparency"] + 0.3,
                    0.1,
                ),
            }
        if self.repres == "lines":
            texture_params = {
                "dynamic_width": [text_params["dynamic_width"]],
                "dynamic_width_max": [text_params["dynamic_width_max"]],
                "dynamic_width_min": [text_params["dynamic_width_min"]],
                "dynamic_width_factor": np.arange(
                    text_params["dynamic_width_factor"] - 0.1,
                    text_params["dynamic_width_factor"] + 0.4,
                    0.1,
                ),
                "line_radius": np.arange(
                    text_params["line_radius"] - 0.1,
                    text_params["line_radius"] + 0.4,
                    0.1,
                ),
                "line_smooth": [text_params["line_smooth"]],
                "line_width": [text_params["line_width"]],
            }

        if self.repres == "ribbons":
            texture_params = {
                "ribbon_nucleic_acid_mode": [text_params["ribbon_nucleic_acid_mode"]],
                "ribbon_power": np.arange(
                    text_params["ribbon_power"] - 1, text_params["ribbon_power"] + 3, 1
                ),
                "ribbon_power_b": np.arange(
                    text_params["ribbon_power_b"] - 0.1,
                    text_params["ribbon_power_b"] + 0.4,
                    0.1,
                ),
                "ribbon_radius": np.arange(
                    text_params["ribbon_radius"] - 0.1,
                    text_params["ribbon_radius"] + 0.3,
                    0.1,
                ),
                "ribbon_sampling": np.arange(
                    text_params["ribbon_power"] - 1, text_params["ribbon_power"] + 3, 1
                ),
                "ribbon_side_chain_helper": [text_params["ribbon_side_chain_helper"]],
                "ribbon_throw": [text_params["ribbon_throw"]],
                "ribbon_trace_atoms": [text_params["ribbon_trace_atoms"]],
                "ribbon_width": np.arange(
                    text_params["ribbon_width"] - 1, text_params["ribbon_width"] + 3, 1
                ),
                "trace_atoms_mode": [text_params["trace_atoms_mode"]],
            }
        if self.repres == "cartoon":
            texture_params = {
                "cartoon_cylindrical_helices": [
                    text_params["cartoon_cylindrical_helices"]
                ],
                "cartoon_debug": [text_params["cartoon_debug"]],
                "cartoon_dumbbell_length": np.arange(
                    text_params["cartoon_dumbbell_length"] - 1,
                    text_params["cartoon_dumbbell_length"] + 3,
                    1,
                ),
                "cartoon_dumbbell_radius": [text_params["cartoon_dumbbell_radius"]],
                "cartoon_dumbbell_width": [text_params["cartoon_dumbbell_width"]],
                "cartoon_fancy_helices": [text_params["cartoon_fancy_helices"]],
                "cartoon_fancy_sheets": [text_params["cartoon_fancy_sheets"]],
                "cartoon_flat_sheets": [text_params["cartoon_flat_sheets"]],
                "cartoon_loop_cap": [text_params["cartoon_loop_cap"]],
                "cartoon_nucleic_acid_mode": [text_params["cartoon_nucleic_acid_mode"]],
                "cartoon_oval_quality": np.arange(
                    text_params["cartoon_oval_quality"] - 1,
                    text_params["cartoon_oval_quality"] + 3,
                    1,
                ),
                "cartoon_ring_finder": [text_params["cartoon_ring_finder"]],
                "cartoon_smooth_cycles": [text_params["cartoon_smooth_cycles"]],
                "cartoon_transparency": [text_params["cartoon_transparency"]],
                "cartoon_tube_cap": [text_params["cartoon_tube_cap"]],
            }
        if self.repres == "dots":
            texture_params = {
                "dot_density": [text_params["dot_density"]],
                "dot_hydrogens": [text_params["dot_hydrogens"]],
                "dot_lighting": [text_params["dot_lighting"]],
                "dot_normals": [text_params["dot_normals"]],
                "dot_radius": np.arange(
                    text_params["dot_radius"] - 0.1,
                    text_params["dot_radius"] + 0.3,
                    0.1,
                ),
                "dot_solvent": [text_params["dot_solvent"]],
                "trim_dots": [text_params["trim_dots"]],
            }
        z = merge_two_dicts(texture_params, common_params)
        param_names = list(z.keys())
        param_values = (zip(param_names, x) for x in product(*z.values()))
        self.model.apply(weights_init_uniform_rule)
        epoch = 1
        lr = 0.004
        if self.method == "neural":
            generated_image = self.style.clone().requires_grad_(True)
            optimizer = optim.Adam([generated_image], lr=lr)
        losses = []
        parms = []
        cmd = xlmrpclib.ServerProxy("http://localhost:9123/")
        cmd.fetch(self.protein)
        for paramset in param_values:
            start_time = time.time()
            kwargs = dict(paramset)
            cmd.do(
                f"""                   
                set ray_trace_gain, {kwargs['ray_trace_gain']}
                set ray_trace_mode, {kwargs['ray_trace_mode']}
                bg_color white
                set ray_trace_color, black
                """
            )
            if self.repres == "spheres":
                cmd.do(
                    f"""
                        as spheres
                        remove solvent
                        unset specular
                        set sphere_scale, {kwargs['sphere_scale']}
                        set sphere_transparency, {kwargs['sphere_transparency']}
                        set sphere_mode, {kwargs['sphere_mode']}
                        unset depth_cue
                        ray 256, 256
                        """
                )

            elif self.repres == "sticks":
                cmd.do(
                    f"""
                        as sticks
                        set stick_radius, {kwargs['stick_radius']}
                        set stick_fixed_radius, {kwargs['stick_fixed_radius']}
                        set stick_nub, {kwargs['stick_nub']}
                        set stick_transparency, {kwargs['stick_transparency']}
                        ray 256, 256
                        """
                )

            elif self.repres == "lines":
                cmd.do(
                    f"""
                        as lines
                        set dynamic_width, {kwargs['dynamic_width']}
                        set dynamic_width_max, {kwargs['dynamic_width_max']}
                        set dynamic_width_min,  {kwargs['dynamic_width_min']}
                        set dynamic_width_factor, {kwargs['dynamic_width_factor']}
                        set line_radius, {kwargs['line_radius']}
                        set line_smooth, {kwargs['line_smooth']}
                        set line_width, {kwargs['line_width']}
                        ray 256, 256
                        """
                )

            elif self.repres == "ribbons":
                cmd.do(
                    f"""
                        as ribbons
                        set ribbon_nucleic_acid_mode, {kwargs['ribbon_nucleic_acid_mode']}
                        set ribbon_power, {kwargs['ribbon_power']}
                        set ribbon_power_b, {kwargs['ribbon_power_b']}
                        set ribbon_radius, {kwargs['ribbon_radius']}
                        set ribbon_sampling, {kwargs['ribbon_sampling']}
                        set ribbon_side_chain_helper, {kwargs['ribbon_side_chain_helper']}
                        set ribbon_throw, {kwargs['ribbon_throw']}
                        set ribbon_trace_atoms, {kwargs['ribbon_trace_atoms']}
                        set ribbon_width,  {kwargs['ribbon_width']}
                        set trace_atoms_mode,  {kwargs['trace_atoms_mode']}
                        ray 256, 256
                        """
                )

            elif self.repres == "cartoon":
                cmd.do(
                    f"""
                        as cartoon
                        set cartoon_cylindrical_helices,  {kwargs['cartoon_cylindrical_helices']}
                        set cartoon_debug, {kwargs['cartoon_debug']}
                        set cartoon_dumbbell_length, {kwargs['cartoon_dumbbell_length']}
                        set cartoon_dumbbell_radius, {kwargs['cartoon_dumbbell_radius']}
                        set cartoon_dumbbell_width, {kwargs['cartoon_dumbbell_width']}
                        set cartoon_fancy_helices, {kwargs['cartoon_fancy_helices']}
                        set cartoon_fancy_sheets, {kwargs['cartoon_fancy_sheets']}
                        set cartoon_flat_sheets, {kwargs['cartoon_flat_sheets']}
                        set cartoon_loop_cap, {kwargs['cartoon_loop_cap']}
                        set cartoon_nucleic_acid_mode, {kwargs['cartoon_nucleic_acid_mode']}
                        set cartoon_oval_quality, {kwargs['cartoon_oval_quality']}
                        set cartoon_ring_finder, {kwargs['cartoon_ring_finder']}
                        set cartoon_smooth_cycles, {kwargs['cartoon_smooth_cycles']}
                        set cartoon_transparency, {kwargs['cartoon_transparency']}
                        set cartoon_tube_cap, {kwargs['cartoon_tube_cap']}
                        ray 256, 256
                        """
                )
            elif self.repres == "dots":
                cmd.do(
                    f"""
                        as dots
                        set dot_density, {kwargs['dot_density']}
                        set dot_hydrogens, {kwargs['dot_hydrogens']}
                        set dot_lighting, {kwargs['dot_lighting']}
                        set dot_normals, {kwargs['dot_normals']}
                        set dot_radius, {kwargs['dot_radius']}
                        set dot_solvent, {kwargs['dot_solvent']}
                        set trim_dots, {kwargs['trim_dots']}
                        ray 256, 256
                        """
                )
            # render_image('styled')
            cmd.png("styled.png")
            time.sleep(1)
            if self.method == "neural":
                styled = image_loader("styled.png")
                for e in range(epoch):
                    gen_features = self.model(generated_image)
                    orig_feautes = self.model(styled)
                    style_featues = self.model(self.style)
                    total_loss = calculate_loss(
                        gen_features, orig_feautes, style_featues
                    )
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    losses.append(total_loss)
                    parms.append(kwargs)
                    print(kwargs)
                    print("--- %s seconds ---" % (time.time() - start_time))
            elif self.method == "ssim":
                styled = cv2.imread("styled.png")
                resized = cv2.resize(
                    styled,
                    (style.shape[1], style.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
                ssim = compare_ssim(Image.fromarray(style), Image.fromarray(resized))
                losses.append(ssim)
        return parms[np.argmin(losses)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare_method", dest="compare_method", type=str)
    args = parser.parse_args()
    PARAMS = {
        "dynamic_width": 1,
        "dynamic_width_max": 5,
        "dynamic_width_min": 2.5,
        "dynamic_width_factor": 0.4,
        "line_radius": 1,
        "line_smooth": 1,
        "line_width": 2,
        "ray_trace_gain": [5],
        "ray_trace_mode": [1],
    }
    if args.compare_method == "neural":
        style = image_loader("st.png")
        model = PymolTexture_opt(
            compare_method="neural",
            style=style,
            protein="1cjy",
            repres="lines",
            params=PARAMS,
            model=VGG().to(DEVICE).eval(),
        )
    if args.compare_method == "ssim":
        style = cv2.imread("st.png")
        model = PymolTexture_opt(
            compare_method="ssim",
            style=style,
            protein="1cjy",
            repres="lines",
            params=PARAMS,
            model=VGG().to(DEVICE).eval(),
        )
    k = model.run()

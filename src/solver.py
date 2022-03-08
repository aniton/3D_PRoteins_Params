import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from ipymol import viewer as pymol
import numpy as np
from PIL import Image
from itertools import product
from pymol import cmd
from torchvision.utils import save_image
from typing import List, Tuple, TypedDict, Literal, Union, Dict, Optional
from .style_model import calculate_loss, weights_init_uniform_rule, image_loader, VGG
from .utils import render_image, merge_two_dicts

COMMON_PARAMS = {
    "ray_trace_gain": np.arange(0, 20, 5),
    "ray_trace_mode": np.arange(0, 3, 1),
}

SPHERES_TEXT_PARAMS = {
    "sphere_scale": np.arange(0.1, 1, 0.3),
    "sphere_transparency": [0, 0.5],
    "sphere_mode": np.arange(0, 8, 1),
    "sphere_solvent": [0, 1],
    "cull_spheres": [1, 5, 10],
}

STICKS_TEXT_PARAMS = {
    "stick_radius": np.arange(0.1, 1, 0.3),
    "stick_fixed_radius": [0, 1],
    "stick_nub": [0.1, 0.5],
    "stick_transparency": [0, 0.5],
}

LINES_TEXT_PARAMS = {
    "dynamic_width": [0, 1],
    "dynamic_width_max": [0.5, 5, 7.5],
    "dynamic_width_min": [0.5, 2.5, 5],
    "dynamic_width_factor": np.arange(0.1, 1, 0.3),
    "line_radius": [0.5, 1],
    "line_smooth": [0, 1],
    "line_width": [1, 2, 5],
}

RIBBONS_TEXT_PARAMS = {
    "ribbon_nucleic_acid_mode": np.arange(0, 4, 1),
    "ribbon_power": [1, 2, 5],
    "ribbon_power_b": [0.1, 0.5, 0.9],
    "ribbon_radius": [0, 0.5],
    "ribbon_sampling": [1, 5],
    "ribbon_side_chain_helper": [0, 1],
    "ribbon_throw": [1, 1.5],
    "ribbon_trace_atoms": [0, 1],
    "ribbon_width": [1, 3, 6],
    "trace_atoms_mode": [1, 5],
}

CARTOON_TEXT_PARAMS = {
    "cartoon_cylindrical_helices": [0, 1],
    "cartoon_debug": np.arange(0, 3, 1),
    "cartoon_dumbbell_length": [1, 2, 5],
    "cartoon_dumbbell_radius": [0.1, 0.2, 0.5],
    "cartoon_dumbbell_width": [0.1, 0.2, 0.3],
    "cartoon_fancy_helices": [0, 1],
    "cartoon_fancy_sheets": [0, 1],
    "cartoon_flat_sheets": [0, 1],
    "cartoon_loop_cap": np.arange(0, 2, 1),
    "cartoon_nucleic_acid_mode": np.arange(0, 4, 1),
    "cartoon_oval_quality": [1, 5, 10],
    "cartoon_ring_finder": np.arange(1, 4, 1),
    "cartoon_smooth_cycles": [1, 2, 5],
    "cartoon_transparency": [0, 0.5],
    "cartoon_tube_cap": [0, 1, 2],
}

DOTS_TEXT_PARAMS = {
    "dot_density": [1, 2, 5],
    "dot_hydrogens": [0, 1],
    "dot_lighting": [0, 1],
    "dot_normals": [0, 1],
    "dot_radius": [0, 0.2, 0.7],
    "dot_solvent": [0, 1],
    "trim_dots": [0, 1],
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PymolTexture:
    def __init__(
        self,
        style: Image,
        protein: str,
        repres: Literal["spheres", "sticks", "lines", "ribbons", "cartoon", "dots"],
        texture_params: Dict,
        common_params: Dict,
        model: nn.Module,
    ) -> None:
        self.texture_params = texture_params
        self.common_params = common_params
        self.protein = protein
        self.style = style
        self.model = model
        self.repres = repres

    """
    Arguments:
        style: image of the desired style
        protein: pdb name of the desired protein
        repres: representation
        texture_params: EXISTING_PARAMS / SPHERES_TEXT_PARAMS / STICKS_TEXT_PARAMS / LINES_TEXT_PARAMS / RIBBONS_TEXT_PARAMS / DOTS_TEXT_PARAMS
        common_params: COMMON PARAMS for all representations
        model: style transferring model
    """

    def run(self):
        z = merge_two_dicts(self.texture_params, self.common_params)
        param_names = list(z.keys())
        param_values = (zip(param_names, x) for x in product(*z.values()))
        self.model.apply(weights_init_uniform_rule)
        epoch = 1
        lr = 0.004
        generated_image = self.style.clone().requires_grad_(True)
        optimizer = optim.Adam([generated_image], lr=lr)
        losses = []
        parms = []
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
            render_image("styled")
            styled = image_loader("styled.png")
            for e in range(epoch):
                gen_features = self.model(generated_image)
                orig_feautes = self.model(styled)
                style_featues = self.model(self.style)
                total_loss = calculate_loss(gen_features, orig_feautes, style_featues)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                losses.append(total_loss)
                parms.append(kwargs)
                print("--- %s seconds ---" % (time.time() - start_time))
        return parms[argmin(losses)]


if __name__ == "__main__":
    style = image_loader("st.png")
    model = PymolTexture(
        style=style,
        protein="1lmp",
        repres="dots",
        texture_params=DOTS_TEXT_PARAMS,
        common_params=COMMON_PARAMS,
        model=VGG().to(DEVICE).eval(),
    )
    k = model.run()
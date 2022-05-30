import time
import torch
import os
import ast
import torch.nn as nn
from PIL import ImageColor

import xmlrpc.client as xlmrpclib
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from ipymol import viewer as pymol
import numpy as np
import json
import cv2
from SSIM_PIL import compare_ssim
from PIL import Image
from itertools import product
import argparse
from torchvision.utils import save_image
from typing import List, Tuple, TypedDict, Literal, Union, Dict, Optional
from style_model import calculate_loss, weights_init_uniform_rule, image_loader, VGG
from utils import render_image, merge_two_dicts, split_dict
from collections import Counter
from sklearn.cluster import KMeans
from matplotlib import colors

COMMON_PARAMS = {
    "ray_trace_gain": np.arange(0, 10, 5),
    "ray_trace_mode": np.arange(0, 3, 1),
}

SPHERES_TEXT_PARAMS = {
    "sphere_scale": np.arange(0.1, 1, 0.4),
    "sphere_transparency": [0, 0.5],
    "sphere_mode": np.arange(0, 8, 1),
    "sphere_solvent": [0, 1],
    "cull_spheres": [1, 5, 10],
    "sphere_point_max_size": 10,
    "sphere_point_size": 1,
    "sphere_quality": 1,
    "sphere_scale ": 1,
    "nb_spheres_quality": 1,
    "nb_spheres_size": 0.25,
}

STICKS_TEXT_PARAMS = {
    "stick_radius": np.arange(0.1, 1, 0.3),
    "stick_fixed_radius": [0, 1],
    "stick_nub": [0.1, 0.5],
    "stick_transparency": [0, 0.5],
    "stick_ball": 0,
    "stick_ball_ratio": 1,
    "stick_color": -1,
    "stick_overlap": 0.2,
    "stick_quality": 8,
    "stick_valence_scale ": 1,
}

LINES_TEXT_PARAMS = {
    "dynamic_width": [0, 1],
    "dynamic_width_max": [0.5, 5, 7.5],
    "dynamic_width_min": [0.5, 2.5, 5],
    "dynamic_width_factor": np.arange(0.1, 1, 0.3),
    "line_radius": [0.5, 1],
    "line_smooth": [0, 1],
    "line_width": [1, 2, 5],
    "line_stick_helper": 1,
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


def rgb_to_hex(rgb_color):
    hex_color = "#"
    for i in rgb_color:
        i = int(i)
        hex_color += "{:02x}".format(i)
    return hex_color


def prep_image(raw_img):
    modified_img = cv2.resize(raw_img, (900, 600), interpolation=cv2.INTER_AREA)
    modified_img = modified_img.reshape(
        modified_img.shape[0] * modified_img.shape[1], 3
    )
    return modified_img


def color_analysis(img):
    clf = KMeans(n_clusters=2)
    color_labels = clf.fit_predict(img)
    center_colors = clf.cluster_centers_
    counts = Counter(color_labels)
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
    return hex_colors


class PymolTexture:
    def __init__(
        self,
        solver: str,
        params: Dict,
        model: nn.Module,
        color_1: Tuple[int, int, int],
        color_2: Tuple[int, int, int],
        style: Image,
        protein: str,
        cartoon_geis_mode: bool,
        compare_method: Literal["ssim", "neural"],
        repres: Literal["spheres", "sticks", "lines", "ribbons", "cartoon", "dots"],
        view: Optional[Tuple[float, ...]] = None,
    ) -> None:
        self.params = params
        self.protein = protein
        self.cartoon_geis_mode = cartoon_geis_mode
        self.style = style
        self.model = model
        self.color_1 = color_1
        self.color_2 = color_2
        self.repres = repres
        self.method = compare_method
        self.view = view
        self.solver = solver

    """
    Arguments:
        style: image of the desired style
        protein: pdb name of the desired protein
        repres: representation
        params: EXISTING PARAMS / SPHERES_TEXT_PARAMS / STICKS_TEXT_PARAMS / LINES_TEXT_PARAMS / RIBBONS_TEXT_PARAMS / DOTS_TEXT_PARAMS
        common_params: COMMON PARAMS for all representations
        model: style transferring model
    """

    def run(self):
        if self.solver == "optimizer":
            text_params, common_params = split_dict(self.params)
            if self.repres == "spheres":
                texture_params = {
                    "sphere_scale": np.arange(
                        text_params["sphere_scale"] - 0.1,
                        text_params["sphere_scale"] + 0.4,
                        0.1,
                    ),
                    "sphere_transparency": np.arange(
                        text_params["sphere_transparency"],
                        text_params["sphere_transparency"] + 5,
                        1,
                    ),
                    "sphere_mode": [text_params["sphere_mode"]],
                    "sphere_solvent": [text_params["sphere_solvent"]],
                    "cull_spheres": [text_params["cull_spheres"]],
                    "ambient": [text_params["ambient"]],
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
                    "stick_ball": [text_params["stick_ball"]],
                    "stick_ball_ratio": [text_params["stick_ball_ratio"]],
                    "stick_color": [text_params["stick_color"]],
                    "stick_overlap": [text_params["stick_overlap"]],
                    "stick_quality": [text_params["stick_quality"]],
                    "stick_valence_scale ": [text_params["stick_valence_scale"]],
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
                    "ribbon_nucleic_acid_mode": [
                        text_params["ribbon_nucleic_acid_mode"]
                    ],
                    "ribbon_power": np.arange(
                        text_params["ribbon_power"] - 1,
                        text_params["ribbon_power"] + 3,
                        1,
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
                        text_params["ribbon_power"] - 1,
                        text_params["ribbon_power"] + 3,
                        1,
                    ),
                    "ribbon_side_chain_helper": [
                        text_params["ribbon_side_chain_helper"]
                    ],
                    "ribbon_throw": [text_params["ribbon_throw"]],
                    "ribbon_trace_atoms": [text_params["ribbon_trace_atoms"]],
                    "ribbon_width": np.arange(
                        text_params["ribbon_width"] - 1,
                        text_params["ribbon_width"] + 3,
                        1,
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
                    "cartoon_nucleic_acid_mode": [
                        text_params["cartoon_nucleic_acid_mode"]
                    ],
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
        elif self.solver == "solver":
            z = self.params
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
        if os.path.exists(self.protein + ".pdb") and self.view:
            cmd.load(self.protein + ".pdb")
            cmd.set_view(self.view)
        else:
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
                        set sphere_solvent, {kwargs['sphere_solvent']}
                        set ambient,  {kwargs['ambient']}
                        set_color basic, {self.color_1}
                        set_color ligg, {self.color_2}
                        color basic, org
                        color ligg, lig
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
                        set stick_overlap,{kwargs['stick_overlap']}
                        set stick_ball, {kwargs['stick_ball']}
                        srt stick_color,  {kwargs['stick_color']}
                        set stick_quality,  {kwargs['stick_quality']}
                        set stick_ball_ratio,{kwargs['stick_ball_ratio']}
                        set stick_valence_scale, {kwargs['stick_valence_scale']}
                        set_color basic, {self.color_1}
                        set_color ligg, {self.color_2}
                        color basic, org
                        color ligg, lig
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
                        set_color basic, {self.color_1}
                        set_color ligg, {self.color_2}
                        color basic, org
                        color ligg, lig
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
                        set_color basic, {self.color_1}
                        set_color ligg, {self.color_2}
                        color basic, org
                        color ligg, lig
                        ray 256, 256
                        """
                )

            elif self.repres == "cartoon":
                cmd.do(
                    """
                        as cartoon"""
                )
                if self.cartoon_geis_mode:
                    cmd.do(
                        f"""cartoon tube
                        show surface
                        set_color basic, {self.color_1}
                        set surface_color, basic
                        set surface_mode,  {kwargs['surface_mode']}
                        set transperency_mode,  {kwargs['surface_mode']}
                        set transparency,  {kwargs['transparency']}
                        set ray_transparency_oblique,  {kwargs['ray_transparency_oblique']}
                        set ray_transparency_oblique_power,  {kwargs['ray_transparency_oblique_power']}
                    """
                    )
                cmd.do(
                    f"""
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
                        set_color basic, {self.color_1}
                        set_color bg, {self.color_2}
                        color basic
                        bg_color, bg
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
                        set_color basic, {self.color_1}
                        set_color ligg, {self.color_2}
                        color basic, org
                        color ligg, lig
                        ray 256, 256
                        """
                )
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
                parms.append(kwargs)
        result = parms[np.argmin(losses)]
        result["color_1"] = self.color_1
        result["color_2"] = self.color_2
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", dest="solver", type=str)
    parser.add_argument("--compare_method", dest="compare_method", type=str)
    parser.add_argument("--protein", dest="protein", type=str)
    parser.add_argument("--representation", dest="representation", type=str)
    parser.add_argument("--style_image_path", dest="style_image_path", type=str)
    parser.add_argument("--view", dest="view", type=str)
    parser.add_argument("--cartoon_geis_mode", dest="cartoon_geis_mode", type=bool)
    parser.add_argument("--params_txt", dest="params_txt", type=str)

    args = parser.parse_args()
    if args.solver == "optimizer":
        file = open(args.params_txt, "r")
        cont = file.read()
        params = ast.literal_eval(cont)
    elif args.solver == "solver":
        if args.representation == "dots":
            params = merge_two_dicts(DOTS_TEXT_PARAMS, COMMON_PARAMS)
        elif args.representation == "spheres":
            params = merge_two_dicts(SPHERES_TEXT_PARAMS, COMMON_PARAMS)
        elif args.representation == "sticks":
            params = merge_two_dicts(STICKS_TEXT_PARAMS, COMMON_PARAMS)
        elif args.representation == "lines":
            params = merge_two_dicts(LINES_TEXT_PARAMS, COMMON_PARAMS)
        elif args.representation == "ribbons":
            params = merge_two_dicts(RIBBONS_TEXT_PARAMS, COMMON_PARAMS)
        elif args.representation == "cartoon":
            params = merge_two_dicts(CARTOON_TEXT_PARAMS, COMMON_PARAMS)

    # extract 2 main colors from the style image
    style = cv2.imread(args.style_image_path)
    style_image = cv2.cvtColor(style, cv2.COLOR_BGR2RGB)
    modified_image = prep_image(style_image)
    hex = color_analysis(modified_image)
    color_1 = ImageColor.getcolor(hex[0], "RGB")
    color_2 = ImageColor.getcolor(hex[1], "RGB")

    if args.compare_method == "neural":
        style = image_loader(args.style_image_path)
        model = PymolTexture(
            solver=args.solver,
            compare_method="neural",
            style=style,
            cartoon_geis_mode=False,
            protein=args.protein,
            repres=args.representation,
            params=params,
            model=VGG().to(DEVICE).eval(),
            color_1=color_1,
            color_2=color_2,
            view=args.view,
        )
    elif args.compare_method == "ssim":
        style = cv2.imread(args.style_image_path)
        model = PymolTexture(
            solver=args.solver,
            compare_method="ssim",
            style=style,
            cartoon_geis_mode=False,
            protein=args.protein,
            repres=args.representation,
            params=params,
            model=VGG().to(DEVICE).eval(),
            color_1=color_1,
            color_2=color_2,
            view=args.view,
        )
    k = model.run()
    with open("texture_solver_params.txt", "w") as f:
        file.write(json.dumps(k))

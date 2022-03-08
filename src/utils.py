#%pylab inline
from ipymol import viewer as pymol
from pymol import cmd
from IPython.display import Image


def render_image(name):
    img_path = name + ".png"
    cmd.png(img_path, 500, 500)
    return Image(filename=img_path)


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

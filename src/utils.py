#%pylab inline
from ipymol import viewer as pymol
from IPython.display import Image
import xmlrpc.client as xlmrpclib



def render_image(name):
    img_path = name + ".png"
    cmd = xlmrpclib.ServerProxy("http://localhost:9123/")
    cmd.png(img_path, 500, 500)
    return Image(filename=img_path)


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def split_dict(dict1):
    k = []
    dict2 = {}
    keys = dict1.keys()
    for key in keys:
        if key == "ray_trace_gain" or key == "ray_trace_mode":
            dict2[key] = dict1[key]
            k.append(key)

    for key in k:
        del dict1[key]
    return dict1, dict2

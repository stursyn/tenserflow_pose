import tensorflow as tf

def gaussian_2d_normalized(height, width, sigma = 1, center=None):
    x, y = tf.meshgrid(tf.range(0, 1, 1/height), tf.range(0, 1, 1/width), indexing='xy')
    x0 = center[0]
    y0 = center[1]
    return tf.math.exp(-(tf.square(x - x0) + tf.math.square(y - y0))/ tf.math.square(sigma))

g = gaussian_2d_normalized(height=3, width=5, sigma=1.0/15, center=[2.0/3.0, 3.0/5.0])
a = [g, g]

b = tf.stack(a, axis=-1)
print(b)


def gaussian_2d(height, width, sigma = 1, center=None):
    x, y = tf.meshgrid(tf.range(0, height, 1), tf.range(0, width, 1), indexing='xy')
    print(x)
    print(y)
    x0 = center[0]
    y0 = center[1]
    return tf.math.exp(-(tf.square(x - x0) + tf.math.square(y - y0))/ tf.math.square(sigma))

g = gaussian_2d(height=3, width=5, sigma=1, center=[2, 3])


a = tf.constant([0, 1, 2])
b = tf.constant([3, 4, 5])
c = tf.constant([6, 7, 8])
tf.stack([a, b, c], axis=-1)[0]

import numpy as np

size = 7
x = np.arange(0, size, 1, float)
y = x[:, np.newaxis]
print(x)
print(y)

x0 = y0 = size // 2
g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * 1 ** 2))
print(g)

sigma = 1.0
size = 6 * sigma + 1
x, y = tf.meshgrid(tf.range(0, 6*sigma+1, 1), tf.range(0, 6*sigma+1, 1), indexing='xy')
x = tf.cast(x, dtype=tf.float32)
y = tf.cast(y, dtype=tf.float32)

# the center of the gaussian patch should be 1
x0 = size // 2
y0 = size // 2

# generate this 7x7 gaussian patch
gaussian_patch = tf.math.exp(-(tf.square(x - x0) + tf.math.square(y - y0))/ (tf.math.square(sigma) * 2))
print(gaussian_patch)


@tf.function
def generate_2d_guassian(height, width, y0, x0, sigma=1):
    """
    "The same technique as Tompson et al. is used for supervision. A MeanSquared Error (MSE) loss is
    applied comparing the predicted heatmap to a ground-truth heatmap consisting of a 2D gaussian
    (with standard deviation of 1 px) centered on the joint location."

    https://github.com/princeton-vl/pose-hg-train/blob/master/src/util/img.lua#L204
    """
    heatmap = tf.zeros((height, width))

    # this gaussian patch is 7x7, let's get four corners of it first
    xmin = x0 - 3 * sigma
    ymin = y0 - 3 * sigma
    xmax = x0 + 3 * sigma
    ymax = y0 + 3 * sigma
    # if the patch is out of image boundary we simply return nothing according to the source code
    if xmin >= width or ymin >= height or xmax < 0 or ymax < 0:
        return heatmap

    size = 6 * sigma + 1
    x, y = tf.meshgrid(tf.range(0, 6 * sigma + 1, 1), tf.range(0, 6 * sigma + 1, 1), indexing='xy')

    # the center of the gaussian patch should be 1
    center_x = size // 2
    center_y = size // 2

    # generate this 7x7 gaussian patch
    gaussian_patch = tf.cast(
        tf.math.exp(-(tf.square(x - center_x) + tf.math.square(y - center_y)) / (tf.math.square(sigma) * 2)),
        dtype=tf.float32)

    # part of the patch could be out of the boundary, so we need to determine the valid range
    # if xmin = -2, it means the 2 left-most columns are invalid, which is max(0, -(-2)) = 2
    patch_xmin = tf.math.maximum(0, -xmin)
    patch_ymin = tf.math.maximum(0, -ymin)
    # if xmin = 59, xmax = 66, but our output is 64x64, then we should discard 2 right-most columns
    # which is min(64, 66) - 59 = 5, and column 6 and 7 are discarded
    patch_xmax = tf.math.minimum(xmax, width) - xmin
    patch_ymax = tf.math.minimum(ymax, height) - ymin

    # also, we need to determine where to put this patch in the whole heatmap
    heatmap_xmin = tf.math.maximum(0, xmin)
    heatmap_ymin = tf.math.maximum(0, ymin)
    heatmap_xmax = tf.math.minimum(xmax, width)
    heatmap_ymax = tf.math.minimum(ymax, height)

    # finally, insert this patch into the heatmap
    indices = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)

    count = 0
    print(patch_ymin, patch_xmin)
    for j in tf.range(patch_ymin, patch_ymax):
        for i in tf.range(patch_xmin, patch_xmax):
            indices = indices.write(count, [heatmap_ymin + j, heatmap_xmin + i])
            updates = updates.write(count, gaussian_patch[j][i])
            count += 1
    heatmap = tf.tensor_scatter_nd_update(heatmap, indices.stack(), updates.stack())

    # unfortunately, the code below doesn't work because
    # tensorflow.python.framework.ops.EagerTensor' object does not support item assignment
    # heatmap[heatmap_ymin:heatmap_ymax, heatmap_xmin:heatmap_xmax] = gaussian_patch[patch_ymin:patch_ymax,patch_xmin:patch_xmax]

    return heatmap

m = generate_2d_guassian(64, 64, 5, 5)
tf.print(m, summarize=-1)

heatmap = tf.zeros((height, width))
y0, x0 = y[i], x[i]
# this gaussian patch is 7x7, let's get four corners of it first
xmin = x0 - 3 * sigma
ymin = y0 - 3 * sigma
xmax = x0 + 3 * sigma
ymax = y0 + 3 * sigma
# if the patch is out of image boundary we simply return nothing according to the source code
if not (xmin >= width or ymin >= height or xmax < 0 or ymax < 0):
    size = 6 * sigma + 1
    x, y = tf.meshgrid(tf.range(0, 6 * sigma + 1, 1), tf.range(0, 6 * sigma + 1, 1), indexing='xy')

    # the center of the gaussian patch should be 1
    center_x = size // 2
    center_y = size // 2

    # generate this 7x7 gaussian patch
    gaussian_patch = tf.cast(
        tf.math.exp(-(tf.square(x - center_x) + tf.math.square(y - center_y)) / (tf.math.square(sigma) * 2)),
        dtype=tf.float32)

    # part of the patch could be out of the boundary, so we need to determine the valid range
    # if xmin = -2, it means the 2 left-most columns are invalid, which is max(0, -(-2)) = 2
    patch_xmin = tf.math.maximum(0, -xmin)
    patch_ymin = tf.math.maximum(0, -ymin)
    # if xmin = 59, xmax = 66, but our output is 64x64, then we should discard 2 right-most columns
    # which is min(64, 66) - 59 = 5, and column 6 and 7 are discarded
    patch_xmax = tf.math.minimum(xmax, width) - xmin
    patch_ymax = tf.math.minimum(ymax, height) - ymin

    # also, we need to determine where to put this patch in the whole heatmap
    heatmap_xmin = tf.math.maximum(0, xmin)
    heatmap_ymin = tf.math.maximum(0, ymin)
    heatmap_xmax = tf.math.minimum(xmax, width)
    heatmap_ymax = tf.math.minimum(ymax, height)

    # finally, insert this patch into the heatmap
    indices = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)

    count = 0
    for j in tf.range(patch_ymin, patch_ymax):
        for i in tf.range(patch_xmin, patch_xmax):
            indices = indices.write(count, [heatmap_ymin + j, heatmap_xmin + i])
            updates = updates.write(count, gaussian_patch[j][i])
            count += 1
    heatmap = tf.tensor_scatter_nd_update(heatmap, indices.stack(), updates.stack())
heatmaps.append(heatmap)

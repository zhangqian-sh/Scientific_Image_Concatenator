import numpy as np
import os
from skimage import io, transform

conca_x, conca_y = 3, 2
sub_image_len_x, sub_image_len_y = 128, 128
space = int(sub_image_len_x / 32)

input_path = "./Demo/Data/"
image_list = [[None for j in range(conca_x)] for i in range(conca_y)]
image_list[0][0] = input_path + "1.png"
image_list[0][1] = input_path + "2.png"
image_list[0][2] = input_path + "3.png"
image_list[1][0] = input_path + "4.png"
image_list[1][1] = input_path + "5.png"
image_list[1][2] = input_path + "6.png"
channel_num = 4

output_path = "./Demo/output/"
os.makedirs(output_path, exist_ok=True)
output_name = "output.png"

canvas = np.ones(
    (
        conca_y * sub_image_len_y + (conca_y - 1) * space,
        conca_x * sub_image_len_x + (conca_x - 1) * space,
        channel_num,
    )
)

for i in range(conca_y):
    for j in range(conca_x):
        image = io.imread(image_list[i][j])
        image = transform.resize(image, (sub_image_len_y, sub_image_len_x, channel_num))
        anchor_y, anchor_x = (
            i * sub_image_len_y + i * space,
            j * sub_image_len_x + j * space,
        )
        canvas[
            anchor_y : anchor_y + sub_image_len_y,
            anchor_x : anchor_x + sub_image_len_x,
            :,
        ] = image[:, :, :]

io.imsave(output_path + output_name, canvas)

import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

from marching_squares import marching_squares


def draw_demo_image(size=512):
    x = torch.arange(0, size).repeat(size, 1)
    y = x.T
    r1 = size // 3
    cx1 = size // 5
    cy1 = size // 5
    cx2 = 3 * size // 5
    cy2 = 2 * size // 5
    r2 = size // 4
    c1 = (x - cx1) ** 2 + (y - cy1) ** 2 < r1**2
    c2 = (x - cx2) ** 2 + (y - cy2) ** 2 < r2**2
    return c1 | c2


def main():
    fig, ax = plt.subplots(figsize=(15, 7))
    image = draw_demo_image()
    ax.imshow(image, cmap="gray")
    contours = marching_squares(image)
    for contour in contours:
        contour_cartesian_coords = contour[:, [1, 0]]
        ax.add_patch(
            Polygon(contour_cartesian_coords, fill=None, linewidth=3, color="green")
        )
    fig.savefig("demo.png")


if __name__ == "__main__":
    main()

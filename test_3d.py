import matplotlib.pyplot as plt
from matplotlib import animation

from blog_utils import *


def make_cloth_sample_video(fname, pmf=None):

    sample = sample_cloth(2000, pmf=pmf)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # blender uses the y component (axis=1) as the height
    def init():
        ax.scatter(sample[:, 0], sample[:, 2], sample[:, 1], s=2, c='m')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        return fig,

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig,

    # Animate
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=20, blit=True)
    # Save
    anim.save(fname, fps=30, extra_args=['-vcodec', 'libx264'])
    plt.clf()


make_cloth_sample_video('cloth_images/cloth_uniform_sample.mp4')

test_pmf = (0.07
    + make_gaussian([0.65, 0.75], 0.05, 200)
    + make_gaussian([0.25, 0.55], [0.10, 0.07], 200)
    + make_gaussian([0.85, 0.15], [0.05, 0.2], 200)
)
test_pmf /= test_pmf.sum()

make_cloth_sample_video('cloth_images/cloth_nonuniform_sample.mp4', pmf=test_pmf)
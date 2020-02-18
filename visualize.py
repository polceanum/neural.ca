from __future__ import division
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
 
class AnimatedGif:
    def __init__(self, size=(640, 480)):
        self.fig = plt.figure()
        self.fig.set_size_inches(size[0] / 100, size[1] / 100)
        ax = self.fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
        ax.set_xticks([])
        ax.set_yticks([])
        self.images = []
 
    def add(self, image, label=''):
        plt_im = plt.imshow(image, vmin=0, vmax=1, animated=True)
        plt_txt = plt.text(10, 310, label, color='red')
        self.images.append([plt_im, plt_txt])
 
    def save(self, filename):
        animation = anim.ArtistAnimation(self.fig, self.images)
        animation.save(filename, writer='imagemagick', fps=16)
        plt.close()

def saveVideo(seq, filename, height=40, width=40, batch_index=0):
    animated_gif = AnimatedGif(size=(width, height))
    for i in range(len(seq)):
        img = seq[i].detach().cpu()[batch_index, :, :, :4].clamp(0,1).numpy()
        animated_gif.add(img, label=str(i))
     
    animated_gif.save(filename)
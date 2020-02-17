import numpy as np
import imageio
import matplotlib.pyplot as plt

# specific function for loading data for a single emoji (40x40)
def load_emoji(path, index):
	im = imageio.imread(path)
	emoji = np.array(im[:, index*40:(index+1)*40].astype(np.float32))
	# normalize
	emoji /= 255.0

	return emoji


if __name__ == '__main__':
	# plot each emoji to test
	for i in range(8):
		emoji = load_emoji('./data/emoji.png', i)
		plt.imshow(emoji[:, :, 3])
		plt.show()
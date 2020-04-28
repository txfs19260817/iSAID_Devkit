'''
script helps preview the annotated ground truth
'''

from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mplcursors
import sys

image = Image.open(sys.argv[1])
data = np.array(image)
img = plt.imshow(data, cmap=matplotlib.cm.plasma, norm=matplotlib.colors.Normalize(vmin=3, vmax=100))

points = []

cursor = mplcursors.cursor(img, hover=False)
@cursor.connect("add")
def cursor_clicked(sel):
    # sel.annotation.set_visible(False)
    sel.annotation.set_text(
        f'Clicked on\nx: {sel.target[0]:.2f} y: {sel.target[1]:.2f}\nindex: {sel.target.index}')
    points.append(sel.target.index)
    print("Current list of points:", points)

plt.title(sys.argv[1])
plt.show()
print("Selected points:", points)
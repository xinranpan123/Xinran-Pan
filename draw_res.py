
import numpy as np
from PIL import Image

"""
"""
W = np.loadtxt('W_matrix.txt')

W = (W-W.min())/(W.max()-W.min())
X = W*255
X_all = []
for i in range(8):
    X_all.append(X[i].reshape(7, 8))
    im = Image.fromarray(X_all[i])
    im = im.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
    im.save('outfile_{}.png'.format(i))
# im.show()
print('hello world')
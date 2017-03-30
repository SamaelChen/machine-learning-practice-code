import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from PIL import ImageDraw


tmp = Image.open('weibo.cn2.png')
tmp = tmp.convert('L')
plt.imshow(tmp, cmap='gray')
tmp = np.array(tmp)
tmp[tmp >= 140] = 255
tmp[tmp < 140] = 0
plt.imshow(tmp, cmap='gray')
draw = ImageDraw.Draw(tmp)
for i in range(0, 100):
    draw.point([random.randint(0, 100), random.randint(0, 20)], 255)
# help(draw.point)
plt.imshow(tmp, cmap='gray')
draw.text([80, 0], 'A', fill=0)
plt.imshow(tmp, cmap='gray')

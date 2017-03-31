from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import random
import math
import string
import codecs
import matplotlib.pyplot as plt
import numpy as np


class RandomChar():
    """用于随机生成汉字"""
    @staticmethod
    def Unicode():
        val = random.randint(0x4E00, 0x9FBF)
        return chr(val)

    @staticmethod
    def GB2312():
        head = random.randint(0xB0, 0xCF)
        body = random.randint(0xA, 0xF)
        tail = random.randint(0, 0xF)
        # val = (head &lt; &lt; 8) | (body & lt; & lt; 4) | tail
        val = (head << 8) | (body << 4) | tail
        str = "%x" % val
        str = codecs.decode(str, 'hex')
        str = str.decode('gb2312')
        # return str.decode('hex').decode('gb2312')
        return str, val


class ImageChar():

    def __init__(self, fontColor=(255, 255, 255),
                 size=(100, 20),
                 fontPath='wqy-microhei.ttc',
                 bgColor=(0, 0, 0),
                 fontSize=20):
        self.size = size
        self.fontPath = fontPath
        self.bgColor = bgColor
        self.fontSize = fontSize
        self.fontColor = fontColor
        self.font = ImageFont.truetype(self.fontPath, self.fontSize)
        self.image = Image.new('RGB', size, bgColor)

    # def rotate(self):
    #     self.image = self.image.rotate(10, expand=0)

    def drawText(self, pos, txt, fill):
        draw = ImageDraw.Draw(self.image)
        draw.text(pos, txt, font=self.font, fill=fill)
        del draw

    def drawTextV2(self, pos, txt, fill):
        image = Image.new('RGB', (20, 20), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.text((0, -3), txt, font=self.font, fill=fill)
        w = image.rotate(random.randint(-10, 10), expand=1)
        self.image.paste(w, box=pos)
        del draw

    # def randRGB(self):
    #     return (random.randint(0, 255),
    #             random.randint(0, 255),
    #             random.randint(0, 255))

    def randPoint(self, num):
        (width, height) = self.size
        draw = ImageDraw.Draw(self.image)
        for i in range(0, num):
            draw.point([random.randint(0, width),
                        random.randint(0, height)], (255, 255, 255))
        # return (random.randint(0, width), random.randint(0, height)
        del draw

    # def randLine(self, num):
    #     draw = ImageDraw.Draw(self.image)
    #     for i in range(0, num):
    #         draw.line([self.randPoint(), self.randPoint()], self.randRGB())
    #     del draw

    def randChinese(self, num):
        gap = 5
        start = 0
        label = []
        for i in range(0, num):
            char, val = RandomChar().GB2312()
            x = start + self.fontSize * i + random.randint(0, gap) + gap * i
            self.drawTextV2((x, random.randint(-3, 2)),
                            char, (255, 255, 255))
            # self.image.rotate(180)
            # self.rotate()
            label.append(val)
        self.randPoint(18)
        return label

    def save(self, path):
        self.image.save(path)


RandomChar().GB2312()
ic = ImageChar(fontPath='ukai.ttc')
ic.randChinese(3)
plt.imshow(ic.image, cmap='gray')
plt.imshow(np.array(ic.image), cmap='gray')
tmp = np.array(ic.image.convert('L'))
tmp.shape
plt.imshow(tmp, cmap='gray')
tmp = 255 - tmp
tmp[tmp >= 200] = 255
tmp[tmp < 200] = 0
plt.imshow(tmp, cmap='gray')
ic.rotate()
ic.image

# import cv2
import numpy as np
import PIL
from PIL import Image
import os

os.chdir(r'C:')
img = Image.open('dog.jpg')
width,height = img.size
dst = img.resize((32, 32))
dst.save('doggy.jpg')

# 008
# Gad Lidror
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

def question1NonVec(imgarr):
  red1, green1 = 0 ,0
  for l in range(len(imgarr)):
    for c in range(len(imgarr[0])):
      curr_r = imgarr[l][c][0]
      curr_g = imgarr[l][c][1]
      if(curr_r > curr_g):
        red1 += 1
      elif(curr_g > curr_r):
        green1 += 1

  return red1, green1

def question1Vec(imgarr):
  all_r = imgarr[:,:,0]
  all_g = imgarr[:,:,1]

  red1 = np.sum(all_r > all_g)
  green1 = np.sum(all_g > all_r)

  return red1, green1

print (" --- Targil 008.1 ---")

raccoon = Image.open(r'unit10\Raccoon.png')
plt.imshow(raccoon)
plt.show()
array = np.array(raccoon)
tic = time.time()
red1, green1 = question1NonVec(array)
toc = time.time()
print("Non Vectorized version: red = " + str(red1) + ", green = " + str(green1) + ". It took " + str(1000*(toc-tic)) + "ms")
tic = time.time()
red1, green1 = question1Vec(array)
toc = time.time();
print("Vectorized version: red = " + str(red1) + ", green = " + str(green1) + ". It took " + str(1000*(toc-tic)) + "ms")


print (" --- Targil 008.2 ---")
array = np.array([[[0,1,2],[10,11,12],[20,21,22]],[[100,101,102],[110,111,112],[120,121,122]]])
print("flat shape: " + str(array.flatten(order='F')))
print("reshape array: " + str(np.reshape(array, -1)))
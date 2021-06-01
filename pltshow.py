import matplotlib.pyplot as plt

plt.figure()
img = plt.imread('dog.jpg')
fig,ax=plt.subplots()
ax.imshow(img,extent=[1, 10, 2, 11])
plt.text(5,5,"pred_class",fontsize = 15)
plt.show()

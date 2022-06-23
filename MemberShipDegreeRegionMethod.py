image = cv2.imread("Fabric23.jpg")

rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.imshow(rgb)
plt.show()

hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
plt.imshow(hsv)
plt.show()

h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

plt.imshow(v, cmap="gray") 
plt.show()


blr = cv2.blur(v,(5, 5))
plt.imshow(blr, cmap="gray")
plt.show()


dst_blr = cv2.fastNlMeansDenoising(blr,None,10,7,21)
plt.imshow(dst_blr, cmap="gray")
plt.show()

_,binary = cv2.threshold(dst_blr, 127, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(binary, cmap="gray")
plt.show()

kernel = np.ones((5, 5), np.uint8)

dilation = cv2.dilate(binary, kernel,iterations = 1) 
plt.imshow(dilation, cmap="gray")
plt.show()

erode = cv2.erode(binary, kernel,iterations = 1) 
plt.imshow(erode, cmap="gray")
plt.show()

if (dilation==0).sum() >1:
        print("Bad Fabric")
        contours,_ = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for i in contours:
            if cv2.contourArea(i)< 5000.0:
                cv2.drawContours(rgb, i,-1,(0,0,255), 3)
else:
        print("Good Fabric")

plt.imshow(rgb)
plt.show()

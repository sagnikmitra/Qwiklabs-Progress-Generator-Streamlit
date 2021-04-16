from PIL import Image

img1 = Image.open(r"test.jpg")
  
# Opening the secondary image (overlay image)
img2 = Image.open(r"first.png")
  
# Pasting img2 image on top of img1 
# starting at coordinates (0, 0)
img1.paste(img2, (0,0), mask = img2)
  
# Displaying the image
img1.show()
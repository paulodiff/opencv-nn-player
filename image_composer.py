# open an image and insert another image into

from PIL import Image
img = Image.open('C:/nn/dataset/tmp/10Q.jpg', 'r')
img_w, img_h = img.size
print(img.size)
basewidth = 200
hsize = 300
img = img.resize((basewidth,hsize), Image.ANTIALIAS)
img_w, img_h = img.size
print(img.size)

background = Image.open('C:/nn/sample_imgs/panno-verde-900x900.jpg', 'r')
bg_w, bg_h = background.size
print(background.size)
offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
print(offset)

background.paste(img, offset)
background.save('C:/nn/sample_imgs/out.jpg')
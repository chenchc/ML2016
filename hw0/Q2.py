from PIL import Image
import sys

filename = sys.argv[1]
image = Image.open(filename)
width, height = image.size

outputImage = Image.new(image.mode, (width, height), 'white')
for i in range(0, height, 1):
	for j in range(0, width, 1):
		pix = image.getpixel((width - j - 1, height - i - 1))
		outputImage.putpixel((j, i), pix)

OUTPUT_FILENAME = 'ans2.png'
outputImage.save(OUTPUT_FILENAME)

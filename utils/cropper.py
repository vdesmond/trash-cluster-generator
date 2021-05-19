from PIL import Image, ImageChops
import glob
from tqdm import tqdm

total = glob.glob("/home/vigneshdesmond/Desktop/taco-dataset/pngtest/*")
counter = 0

for i, filename in enumerate(tqdm(total)):
    img = Image.open(filename)
    pixels = img.load()

    xlist = []
    ylist = []
    for y in range(0, img.size[1]):
        for x in range(0, img.size[0]):
            if pixels[x, y] != (0, 0, 0, 0):
                xlist.append(x)
                ylist.append(y)
    left = min(xlist)
    right = max(xlist)
    top = min(ylist)
    bottom = max(ylist)

    img = img.crop((left - 10, top - 10, right + 10, bottom + 10))
    img.save(filename[:-4] + "-cropped-" + str(i) + ".png")
print("All pngs cropped to fit.\n")

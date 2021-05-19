from PIL import Image, ImageChops
import glob
from tqdm import tqdm
import concurrent.futures
import os


def cropper(filename):
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
    img.save(filename[:-4] + "-cropped.png")
    return True


if __name__ == "__main__":
    images_list = glob.glob(os.getcwd() + "/pngs/*")
    print("\nStarting process ...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        _ = list(tqdm(executor.map(cropper, images_list), total=len(images_list)))
    print("Done.")

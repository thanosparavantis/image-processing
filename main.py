import os
from PIL import Image, ImageCms


def rgb2lab(img):
    srgb_profile = ImageCms.createProfile('sRGB')
    lab_profile = ImageCms.createProfile('LAB')

    transform = ImageCms.buildTransformFromOpenProfiles(
        srgb_profile,
        lab_profile,
        'RGB',
        'LAB'
    )

    lab = ImageCms.applyTransform(
        img,
        transform
    )

    return lab.split()


if __name__ == '__main__':
    if not os.path.isdir('temp'):
        os.mkdir('temp')

    img = Image.open('image.jpg')
    L, a, b = rgb2lab(img)

    L.save('temp/L.jpg')
    a.save('temp/a.jpg')
    b.save('temp/b.jpg')

    L.show()
    a.show()
    b.show()

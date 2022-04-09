import cv2 as cv

def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


img = cv.imread('persons/person01/p1-16.jpg')

print("img : {}".format(img.shape))

W = 128
H = 256

img2 = img.copy()
img3 = img.copy()

ri2 = image_resize(img2, W, H)
ri3 = cv.resize(src=img3, dsize=(W, H))

ri4 = cv.resize(src=ri2, dsize=(W, H))

cv.imshow('origin', img)

cv.imshow('resize_2', ri2)
print("resize_2 : {}".format(ri2.shape))

cv.imshow('resize_3', ri3)
print("resize_3 : {}".format(ri3.shape))

cv.imshow('resize_4', ri4)
print("resize_4 : {}".format(ri4.shape))

if cv.waitKey(0) == ord('q'):  # q to quit
    raise StopIteration

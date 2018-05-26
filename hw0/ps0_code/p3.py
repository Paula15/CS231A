# CS231A Homework 0, Problem 3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import misc


def normalize(img):
    mi, mx = np.min(np.abs(img)), np.max(np.abs(img))
    return (img - mi) / (mx - mi)


def main():
    # ===== Problem 3a =====
    # Read in the images, image1.jpg and image2.jpg, as color images.

    img1, img2 = None, None

    # BEGIN YOUR CODE HERE
    img1 = misc.imread('image1.jpg')
    img2 = misc.imread('image2.jpg')
    # END YOUR CODE HERE

    # ===== Problem 3b =====
    # Convert the images to double precision and rescale them
    # to stretch from minimum value 0 to maximum value 1.

    # BEGIN YOUR CODE HERE
    img1 = normalize(img1.astype(np.float64))
    img2 = normalize(img2.astype(np.float64))
    # END YOUR CODE HERE

    # ===== Problem 3c =====
    # Add the images together and re-normalize them 
    # to have minimum value 0 and maximum value 1. 
    # Display this image.

    # BEGIN YOUR CODE HERE
    imgAdd = normalize(img1 + img2)
    misc.imsave('runs/3c.png', imgAdd)
    # END YOUR CODE HERE

    # ===== Problem 3d =====
    # Create a new image such that the left half of 
    # the image is the left half of image1 and the 
    # right half of the image is the right half of image2.

    newImage1 = None

    # BEGIN YOUR CODE HERE
    newImage1 = np.concatenate([img1[:, :img1.shape[1]/2, :], img2[:, img2.shape[1]/2:, :]], axis=1)
    misc.imsave('runs/3d.png', newImage1)
    # END YOUR CODE HERE

    # ===== Problem 3e =====
    # Using a for loop, create a new image such that every odd 
    # numbered row is the corresponding row from image1 and the 
    # every even row is the corresponding row from image2. 
    # Hint: Remember that indices start at 0 and not 1 in Python.

    newImage2 = None

    # BEGIN YOUR CODE HERE
    newImage2 = []
    nLines = img1.shape[0]
    for i in xrange(nLines):
        if i % 2 == 1:
            newImage2.append(img1[i])
        else:
            newImage2.append(img2[i])
    newImage2 = np.array(newImage2)
    misc.imsave('runs/3e.png', newImage2)
    # END YOUR CODE HERE

    # ===== Problem 3f =====
    # Accomplish the same task as part e without using a for-loop.
    # The functions reshape and repmat may be helpful here.

    newImage3 = None

    # BEGIN YOUR CODE HERE
    mask1 = np.tile(np.concatenate(
        [np.zeros([1, img1.shape[1], 1]), np.ones([1, img2.shape[1], 1])], axis=0), reps=[img1.shape[0]/2, 1, 1])
    mask2 = np.tile(np.concatenate(
        [np.ones([1, img1.shape[1], 1]), np.zeros([1, img2.shape[1], 1])], axis=0), reps=[img1.shape[0]/2, 1, 1])
    newImage3 = mask1 * img1 + mask2 * img2
    misc.imsave('runs/3f.png', newImage3)
    # END YOUR CODE HERE

    # ===== Problem 3g =====
    # Convert the result from part f to a grayscale image. 
    # Display the grayscale image with a title.

    # BEGIN YOUR CODE HERE
    img = np.dot(newImage3[...,:3], [0.299, 0.587, 0.114])
    fig = plt.figure('3g')
    fig.suptitle('title')
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()
    # END YOUR CODE HERE


if __name__ == '__main__':
    main()
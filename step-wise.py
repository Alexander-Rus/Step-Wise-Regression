import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt


# This function loads in the datasets, and normalizes them to be 24x24
def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels


def measureAccuracyOfPredictors(trainingFaces, trainingLabels, param):
    # We need to keep track of the heighest FPC and which pixles(highestBoys) get us that value
    highestFPC = 0
    heightestBoys = []
    #We are going to do four 4 loops to go through every possible combinatino of pixle pairs
    for x in range(24):
        for y in range(24):
            for z in range(24):
                for a in range(24):

                    # Get the FPC in a vectorized way depending on how many pixles we have so far
                    FPC = classifySmile(trainingFaces, trainingLabels, param, x, y, z, a)
                    if FPC > highestFPC:
                        highestFPC = FPC
                        heightestBoys = [x, y, z, a]
    print("Highest FPC was: ", highestFPC)
    print("Highest Values were: ", heightestBoys)
    return ((heightestBoys[0], heightestBoys[1]), (heightestBoys[2], heightestBoys[3])), highestFPC

# Given an array of guesses and an array of labels, return how accurate the guesses are to the labels
def fPC(y, yhat):
    return np.sum(y == yhat) / len(yhat)

# In this function we have to return if we think the image is smiling or not based on the comparision between pixels
def classifySmile(trainingFaces, trainingLabels, param, x, y, z, a):
    # If we have no pixle paris so far, we will find our first pair by going through all possible pairs and finding the highest fpc they yeild
    if len(param) == 0:
        pixel1 = trainingFaces[:, x][:, y]
        pixel2 = trainingFaces[:, z][:, a]
        comarisonMatrix = pixel1 > pixel2
        fpc = fPC(comarisonMatrix, trainingLabels)
        return fpc
    #If we already have one pixle pair, then we need to keep it in our algorithm(greedy algorithm) and then find the next best pair
    elif len(param) == 1:
        pixel1 = trainingFaces[:, param[0][0][0]][:, param[0][0][1]]
        pixel2 = trainingFaces[:, param[0][1][0]][:, param[0][1][1]]
        pixel3 = trainingFaces[:, x][:, y]
        pixel4 = trainingFaces[:, z][:, a]
        comarisonMatrix = pixel1 > pixel2
        comarisonMatrix2 = pixel3 > pixel4

        #This is block of code looks complicated, but it really just says if at least one of the pixel comparisons says true, then we assume it is a smile
        # Due to the fact tha we know ~54% of the faces in the training set are smiling, it means we should assume it is a smile when there is a draw 
        largeBoy = np.array([comarisonMatrix, comarisonMatrix2])
        largerBoy = largeBoy.T.sum(axis=1)
        resultMatrix = largerBoy >= 1
        fpc = fPC(resultMatrix, trainingLabels)
        return fpc
    elif len(param) == 2:
        pixel1 = trainingFaces[:, param[0][0][0]][:, param[0][0][1]]
        pixel2 = trainingFaces[:, param[0][1][0]][:, param[0][1][1]]
        pixel3 = trainingFaces[:, param[1][0][0]][:, param[1][0][1]]
        pixel4 = trainingFaces[:, param[1][1][0]][:, param[1][1][1]]
        pixel5 = trainingFaces[:, x][:, y]
        pixel6 = trainingFaces[:, z][:, a]
        comarisonMatrix = pixel1 > pixel2
        comarisonMatrix2 = pixel3 > pixel4
        comarisonMatrix3 = pixel5 > pixel6

        largeBoy = np.array([comarisonMatrix, comarisonMatrix2, comarisonMatrix3])
        largerBoy = largeBoy.T.sum(axis=1)
        resultMatrix = largerBoy >= 2
        fpc = fPC(resultMatrix, trainingLabels)
        return fpc
    elif len(param) == 3:
        pixel1 = trainingFaces[:, param[0][0][0]][:, param[0][0][1]]
        pixel2 = trainingFaces[:, param[0][1][0]][:, param[0][1][1]]
        pixel3 = trainingFaces[:, param[1][0][0]][:, param[1][0][1]]
        pixel4 = trainingFaces[:, param[1][1][0]][:, param[1][1][1]]
        pixel5 = trainingFaces[:, param[2][0][0]][:, param[2][0][1]]
        pixel6 = trainingFaces[:, param[2][1][0]][:, param[2][1][1]]
        pixel7 = trainingFaces[:, x][:, y]
        pixel8 = trainingFaces[:, z][:, a]
        comarisonMatrix = pixel1 > pixel2
        comarisonMatrix2 = pixel3 > pixel4
        comarisonMatrix3 = pixel5 > pixel6
        comarisonMatrix4 = pixel7 > pixel8

        largeBoy = np.array([comarisonMatrix, comarisonMatrix2, comarisonMatrix3, comarisonMatrix4])
        largerBoy = largeBoy.T.sum(axis=1)
        resultMatrix = largerBoy >= 2
        fpc = fPC(resultMatrix, trainingLabels)
        return fpc
    elif len(param) == 4:
        pixel1 = trainingFaces[:, param[0][0][0]][:, param[0][0][1]]
        pixel2 = trainingFaces[:, param[0][1][0]][:, param[0][1][1]]
        pixel3 = trainingFaces[:, param[1][0][0]][:, param[1][0][1]]
        pixel4 = trainingFaces[:, param[1][1][0]][:, param[1][1][1]]
        pixel5 = trainingFaces[:, param[2][0][0]][:, param[2][0][1]]
        pixel6 = trainingFaces[:, param[2][1][0]][:, param[2][1][1]]
        pixel7 = trainingFaces[:, param[3][0][0]][:, param[3][0][1]]
        pixel8 = trainingFaces[:, param[3][1][0]][:, param[3][1][1]]
        pixel9 = trainingFaces[:, x][:, y]
        pixel10 = trainingFaces[:, z][:, a]
        comarisonMatrix = pixel1 > pixel2
        comarisonMatrix2 = pixel3 > pixel4
        comarisonMatrix3 = pixel5 > pixel6
        comarisonMatrix4 = pixel7 > pixel8
        comarisonMatrix5 = pixel9 > pixel10

        largeBoy = np.array([comarisonMatrix, comarisonMatrix2, comarisonMatrix3, comarisonMatrix4, comarisonMatrix5])
        largerBoy = largeBoy.T.sum(axis=1)
        resultMatrix = largerBoy >= 3
        fpc = fPC(resultMatrix, trainingLabels)
        return fpc

# In this function we know what the pixel comparisions should be, so we can test them with 3/5 being considered a smile
def testModel(optimalFeatures, testingFaces, testingLabels):
    # There are five optimal pixel pairs, here we test to see just how optimal they really are
    pixel1 = testingFaces[:, optimalFeatures[0][0][0]][:, optimalFeatures[0][0][1]]
    pixel2 = testingFaces[:, optimalFeatures[0][1][0]][:, optimalFeatures[0][1][1]]
    pixel3 = testingFaces[:, optimalFeatures[1][0][0]][:, optimalFeatures[1][0][1]]
    pixel4 = testingFaces[:, optimalFeatures[1][1][0]][:, optimalFeatures[1][1][1]]
    pixel5 = testingFaces[:, optimalFeatures[2][0][0]][:, optimalFeatures[2][0][1]]
    pixel6 = testingFaces[:, optimalFeatures[2][1][0]][:, optimalFeatures[2][1][1]]
    pixel7 = testingFaces[:, optimalFeatures[3][0][0]][:, optimalFeatures[3][0][1]]
    pixel8 = testingFaces[:, optimalFeatures[3][1][0]][:, optimalFeatures[3][1][1]]
    pixel9 = testingFaces[:, optimalFeatures[4][0][0]][:, optimalFeatures[4][0][1]]
    pixel10 = testingFaces[:, optimalFeatures[4][1][0]][:, optimalFeatures[4][1][1]]

    comarisonMatrix = pixel1 > pixel2
    comarisonMatrix2 = pixel3 > pixel4
    comarisonMatrix3 = pixel5 > pixel6
    comarisonMatrix4 = pixel7 > pixel8
    comarisonMatrix5 = pixel9 > pixel10


    largeBoy = np.array([comarisonMatrix, comarisonMatrix2, comarisonMatrix3, comarisonMatrix4, comarisonMatrix5])
    largerBoy = largeBoy.T.sum(axis=1)
    resultMatrix = largerBoy >= 3
    fpc = fPC(resultMatrix, testingLabels)
    return fpc


#This function simply shows visually on a sample face where the optimal pixels are. 
def stepwiseRegression (testingFaces, features):
    show = True
    if show:
        # Show an arbitrary test image in grayscale
        im = testingFaces[0,:,:]
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        # Show r1,c1
        rect = patches.Rectangle((features[0][0][0] - 0.5, features[0][0][1] - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Show r2,c2
        rect = patches.Rectangle((features[0][1][0] - 0.5, features[0][1][1] - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Display the merged result
        rect = patches.Rectangle((features[1][0][0] - 0.5, features[1][0][1] - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        # Show r2,c2
        rect = patches.Rectangle((features[1][1][0] - 0.5, features[1][1][1] - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        rect = patches.Rectangle((features[2][0][0] - 0.5, features[2][0][1] - 0.5), 1, 1, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        # Show r2,c2
        rect = patches.Rectangle((features[2][1][0] - 0.5, features[2][1][0] - 0.5), 1, 1, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

        rect = patches.Rectangle((features[3][0][0] - 0.5, features[3][0][1] - 0.5), 1, 1, linewidth=2, edgecolor='y', facecolor='none')
        ax.add_patch(rect)
        # Show r2,c2
        rect = patches.Rectangle((features[3][1][0] - 0.5, features[3][1][0] - 0.5), 1, 1, linewidth=2, edgecolor='y', facecolor='none')
        ax.add_patch(rect)

        rect = patches.Rectangle((features[4][0][0] - 0.5, features[4][0][1] - 0.5), 1, 1, linewidth=2, edgecolor='k', facecolor='none')
        ax.add_patch(rect)
        # Show r2,c2
        rect = patches.Rectangle((features[4][1][0] - 0.5, features[4][1][0] - 0.5), 1, 1, linewidth=2, edgecolor='k', facecolor='none')
        ax.add_patch(rect)

        plt.show()
 

if __name__ == "__main__":
    #Load the training and testing data
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")

    # Here a simple loop will test various sizes of training set {400, 800, 1200, 1600, 2000}
    for val in [400, 800, 1200, 1600, 2000]:

        print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        print("Traing data using size of -> ", val)
        print("...\n")
        features = []
        highestFPC = 0
        for x in range(5):
            print("Feature number " + str(x + 1) + " ==>")
            param, highestFPC = measureAccuracyOfPredictors(trainingFaces[:val], trainingLabels[:val], features)
            features.append(param)
            print("Calculating next pair of pixles ...\n")
        print("These are the final featuers: ", features)
        print("The final FPC from the training set was : ", highestFPC)
        print("\nNow we will calculate the FPC on the testing set ...\n")
        # When we test, regardless of the training set's size, we need to use the full testing matrix
        stepwiseRegression(testingFaces, features)
        accuracy = testModel(features, testingFaces, testingLabels)
        print("The FPC for the testing set is : ", accuracy)

import os
import collections
import time
import cv2
import numpy as np

MODE = "MPI"

if MODE == "MPI":
    protoFile = r"C:\Users\sharm\Downloads\Human-Pose-Estimation-master\mpi-pose_deploy_linevec.prototxt"
    weightsFile = r"C:\Users\sharm\Downloads\Human-Pose-Estimation-master\pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10],
                  [14, 11], [11, 12], [12, 13]]

# Directory paths for input and output images
input_folder = r"C:\Users\sharm\Downloads\Human-Pose-Estimation-master\Human-Pose-Estimation-master\Image_Pose\inputImage"
output_folder = r"C:\Users\sharm\Downloads\Human-Pose-Estimation-master\Human-Pose-Estimation-master\Image_Pose\outputImage"

# List all files in the input folder
input_files = os.listdir(input_folder)

# Process each input image
for input_file in input_files:
    # Input image path
    userImageInput = os.path.join(input_folder, input_file)
    
    # Read the input image
    # frame = cv2.imread(userImageInput)
    img = r"C:\Users\sharm\Downloads\Human-Pose-Estimation-master\Human-Pose-Estimation-master\White.jpg"

    frame = cv2.imread(userImageInput)
    print(frame.shape)
    frameWhite = cv2.imread(img)

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    frameWidth1 = frameWhite.shape[1]
    frameHeight1 = frameWhite.shape[0]
    frame = cv2.resize(frame, (480, 640))
    frameWhite = cv2.resize(frameWhite, (520, 740))
    threshold = 0.1

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    t = time.time()
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    output = net.forward()

    print("time taken by network : {:.3f}".format(time.time() - t))

    H = output.shape[2]
    W = output.shape[3]
    points = []
    points1 = []

    for i in range(nPoints):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H
        x1 = (frameWidth1 * point[0]) / W
        y1 = (frameHeight1 * point[1]) / H
        if prob > threshold:
            cv2.circle(frame, (int(x), int(y)), 8 , (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frameWhite, (int(x1) , int(y1)+40),8,(0, 0, 255), thickness=-1, lineType=cv2.FILLED)

            points.append((int(x), int(y)))
            points1.append((int(x1) , int(y1)+40))
        else:
            points.append(None)
            points1.append(None)

    dictAngle = {'leftHand': [5, 6, 7], 'leftLeg': [11, 12, 13], 'rightHand': [2, 3, 4], 'rightLeg': [8, 9, 10]}
    dictAngle = collections.OrderedDict(sorted(dictAngle.items()))

    usernameDict = []
    userangleDict = []
    heightAngle = [350, 370, 390, 410]
    j = 0
    # print(dictAngle)
    for i in dictAngle:
        dictPoint1 = np.array(points[dictAngle[i][0]])
        dictPoint2 = np.array(points[dictAngle[i][1]])
        dictPoint3 = np.array(points[dictAngle[i][2]])
        # print(dictAngle[i])
        if str(dictPoint1) != 'None' and str(dictPoint2) != 'None' and str(dictPoint3) != 'None':
            ba = dictPoint1 - dictPoint2
            bc = dictPoint3 - dictPoint2
            tup1 = points1[dictAngle[i][1]]

            if i == 'leftHand' or i == 'leftLeg':
                pointAngle = (tup1[0] + 15, tup1[1] + 18)

            if i == 'rightHand' or i == 'rightLeg':
                pointAngle = (tup1[0] - 50, tup1[1])
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            ang = str(np.degrees(angle))
            angleFloat = float(ang)
            ang1 = round(angleFloat, 2)
            cv2.putText(frameWhite, "Right Side", (15, 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frameWhite, "Left Side", (500, 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frameWhite, "User-Image", (700, 420), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frameWhite, str(ang1), pointAngle, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frameWhite, str(i + ": "), (15, heightAngle[j]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
                    cv2.LINE_AA)
            cv2.putText(frameWhite, str(ang), (90, heightAngle[j]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1,
                    cv2.LINE_AA)

            usernameDict.append(i)
            userangleDict.append(ang)
        else:
            usernameDict.append(i)
            userangleDict.append('0')
        j += 1

    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            # cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            cv2.line(frameWhite, points1[partA], points1[partB], (0, 255, 255), 2)

    userDict = dict(zip(usernameDict, userangleDict))
    athletesOutputSecond = r"C:\Users\sharm\Downloads\Human-Pose-Estimation-master\Human-Pose-Estimation-master\Image_Pose\outputImage\output_2.jpg"
    cv2.imwrite(athletesOutputSecond, frameWhite)

    AngleDict = str(
        userDict['rightHand'] + "," + userDict['leftLeg'] + "," + userDict['leftHand'] + "," + userDict['rightLeg'])


    
    # Save the output image
    output_file = "output_" + os.path.splitext(input_file)[0] + ".jpg"
    output_path = os.path.join(output_folder, output_file)
    cv2.imwrite(output_path, frameWhite)

    print("Processed:", input_file, "Output saved as:", output_file)

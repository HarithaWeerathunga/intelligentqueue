import argparse
from datetime import datetime
import datetime
import time
from pathlib import Path
import math

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from db import insertQueueLength


def countPeopleROI(countInside, image):
    cv2.putText(image, "Queue pople: " + str(countInside) , (0,300), cv2.FONT_HERSHEY_SIMPLEX, fontScale= 1, color=(0, 255, 255), thickness=3)

def countExitedPeople(numberOfPeopleExited, image):
    cv2.putText(image, "Exited People: " + str(numberOfPeopleExited) , (0,350), cv2.FONT_HERSHEY_SIMPLEX, fontScale= 1, color=(0, 255, 255), thickness=3)

def printCashierAvailability(cashierID, image):
    cv2.putText(image, "CashierID" + str(cashierID) + "available", (0,400), cv2.FONT_HERSHEY_SIMPLEX, fontScale= 1, color=(0, 255, 255), thickness=3)

def displayExit(image):
    cv2.putText(image, "One is at the exit", (0,500), cv2.FONT_HERSHEY_SIMPLEX, fontScale= 1, color=(0, 255, 255), thickness=3)



def checkExit(centroidX,centroidY,x1,y1,x2,y2,image):
    print('exit')

    dist = abs((y2-y1)*centroidX - (x2-x1)*centroidY + x2*y1 - y2*x1)/math.sqrt((y2-y1)**2 + (x2-x1)**2)
    if dist < 10:
        print("near enough")
        cv2.putText(image, "Near: " , (0,500), cv2.FONT_HERSHEY_SIMPLEX, fontScale= 1, color=(0, 255, 255), thickness=3)
        


# # points that define the line
# p1 = [1, 6]
# p2 = [3, 2]
# x1, y1 = p1
# x2, y2 = p2

# centroid = [2,4]
# x3, y3 = centroid

# # distance from centroid to line
#  # to calculate square root
# dist = abs((y2-y1)*x3 - (x2-x1)*y3 + x2*y1 - y2*x1)/math.sqrt((y2-y1)**2 + (x2-x1)**2)

# if dist < some_value:
#     print("Near enough")



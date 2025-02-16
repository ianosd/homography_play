import numpy as np 
import cv2 as cv
import sys
from collections import defaultdict
import os
from os.path import join as join_path
from json import dump, load


WINDOW_NAME = "Image"
if len(sys.argv) < 2 :
    print("Usage: select_points <dir>")

directory = sys.argv[1]
file_list = os.listdir(directory)
file_list.sort()
file_index = 2

def current_file():
    return join_path(directory, file_list[file_index])

window = cv.namedWindow(WINDOW_NAME)

points = defaultdict(list) 
if len(sys.argv) > 2:
    with open(sys.argv[2], "r") as fd:
        data = load(fd)
        points.update({k: [tuple(point) for point in l] for k, l in data.items()})

# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN:
        points[current_file()].append((x, y))
        cv.circle(img,(x,y),5,(0,0,255),-1)

last_file_index = file_index
last_show_all_points = None
cv.namedWindow(WINDOW_NAME)
cv.setMouseCallback(WINDOW_NAME, draw_circle)
show_all_points = False
while(1):
    if file_index != last_file_index or last_show_all_points != show_all_points:
        last_file_index = file_index
        last_show_all_points = show_all_points
        print(current_file())
        img = cv.imread(current_file())
        if show_all_points:
            for point in (point for l in points.values() for point in l):
                cv.circle(img,point,5,(0,0,255),-1)
        else:
            for x, y in points[current_file()]:
                cv.circle(img,(x,y),5,(0,0,255),-1)
    cv.imshow(WINDOW_NAME,img)
    k = cv.waitKey(1) & 0xFF
    if k == ord('n'):
        file_index = (file_index + 1) % len(file_list)
    elif k == ord('p'):
        file_index = (file_index - 1) % len(file_list)
    elif k == ord('m'):
        file_index = (file_index + 10) % len(file_list)
    elif k == ord('['):
        file_index = (file_index - 10) % len(file_list)
    elif k == ord('d'):
        l = points[current_file()]
        if l :
            p = l.pop()
            last_file_index = None
    elif k == ord('a'):
        show_all_points = not show_all_points
    elif k == 27:
        break

cv.destroyAllWindows()

save = None
while save not in ["y", "n"]:
    save = input("Save?(y/n)")

if save == "y":
    result_fn = input("Enter filename to save points to:")
    with open(result_fn, "w") as fp:
        dump({k:v for k, v in points.items() if v}, fp)
    print(f"Saved points to {result_fn}")

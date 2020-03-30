import cv2
import os 

video_path = "/Users/prashanga/Downloads/ML projects/Face Attendence/Face_video/Prashanga_Pokharel.mp4"
name1 = "Prashanga_Pokharel"
current_dir = os.getcwd()
dir = os.path.join(current_dir,"Face_images",name1)
print(dir)
if not os.path.exists(dir):
    os.mkdir(dir)
currentframe = 0
# for video in os.listdir(video_path):
    # print(video)
    # break
frame = cv2.VideoCapture(video_path)
while(True):
    ref, image = frame.read()
    # print(image)
    # break
    if ref:
        name = "/Users/prashanga/Downloads/ML projects/Face Attendence/Face_images/Prashanga_Pokharel/" + name1 + "--" + str(currentframe) + '.jpg'
        print('creating...'+ name)

        cv2.imwrite(name, image)
        currentframe += 1
    else:
        break

frame.release()
cv2.destroyAllWindows()
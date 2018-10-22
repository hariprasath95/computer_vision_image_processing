import cv2
import numpy as np
import glob
import imutils

    
def match_show_images(images_orig,template):
    print("matching new template")
    images = images_orig
    blur_images = [cv2.GaussianBlur(image,(3,3),1) for image in images]
    laplacian_images = [cv2.Laplacian(image,cv2.CV_32F) for image in blur_images]
    w, h = template.shape[::-1]
    for i,image in enumerate(laplacian_images):
        current_max = 0
        gmin_loc=(0,0)
        gv =0
        gh=0
        flag = False
        for scale in np.linspace(0.6, 1.0, 20)[::-1]:
            resized = imutils.resize(template, width = int(template.shape[1] * scale))
            
            template_gaussian = cv2.GaussianBlur(resized,(3,3),1)
            template_laplacian = cv2.Laplacian(template_gaussian,cv2.CV_32F)
            
            
            r = template_laplacian.shape[1] / float(resized.shape[1])
            result=cv2.matchTemplate(image,template_laplacian,cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if(current_max < max_val and max_val > 0.50):
                current_max =max_val
                gmin_loc = max_loc
                gv,gh =resized.shape[::-1]
                flag = True
        if(flag):
            cv2.rectangle(images[i], gmin_loc, (gmin_loc[0] + gv, gmin_loc[1] + gh), (0,0,0), 2)
        cv2.putText(images[i],str(flag),(100,100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("image",images[i])
        cv2.waitKey(0)

images = [cv2.imread(file,0) for file in glob.glob("task3/*.jpg")]
images_bonus = [cv2.imread(file,0) for file in glob.glob("task3/bonus/*.jpg")]
template = cv2.imread("task3/template/template.png",0)
bonus_template_1 = cv2.imread("task3/template/template_bonus_1.png",0)
bonus_template_2 = cv2.imread("task3/template/template_bonus_2.png",0)

#apply blur and laplacian on main images

match_show_images(images,template)
match_show_images(images_bonus,bonus_template_1)
match_show_images(images_bonus,bonus_template_2)


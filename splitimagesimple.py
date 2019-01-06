import cv2
import os
import uuid

basepath = './simple/'
if not os.path.exists(basepath):
    os.makedirs(basepath)


def img_process(img_path, img_file):
    img = cv2.imread(img_path + "/" + img_file)
    im1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h3 = cv2.adaptiveThreshold(im1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 6)

    for i in range(0, len(h3) - 2):
        for j in range(0, len(h3[0])):
            if h3[i, j] == 0:
                if h3[i + 1, j] == 255:
                    h3[i, j] = 255
                elif h3[i + 1, j] == 0:
                    if h3[i + 2, j] == 255:
                        h3[i: i + 2, j] = 255
    # reduce_noise(h3, 1)
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # h3=cv2.erode(h3, kernel)

    h3[:, 0] = 255
    h3[:, len(h3[0]) - 1] = 255
    h3[1, :] = 255
    h3[len(h3) - 1, :] = 255

    image, contours, hierarchy = cv2.findContours(h3, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    image = cv2.bitwise_not(image)
    # cv2.imshow('img', image)
    remain = []
    remain_map = {}
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])

        # print(area)
        if area < 100:
            cv2.drawContours(image, [contours[i]], 0, 0, -1)
        else:
            x, y, w, h = cv2.boundingRect(contours[i])
            # print(x, y ,w, h)
            # newimage = image[y:y + h, x:x + w]
            # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            # newimage=cv2.dilate(newimage, kernel)
            # cv2.imwrite(str(x) + ".jpg", newimage)
            if x > 0:
                # if not need_split:
                #     need_split = w > 35
                remain.append(x)
                remain_map[str(x)] = (x, y, w, h)
    cv2.imwrite(img_file, image)
    remain.sort()

    rcount = 0
    w_last = 0
    remove_list = []
    for i in range(len(remain)):
        x, y, w, h = remain_map[str(remain[i])]
        if rcount > x + w:
            remove_list.append(remain[i])
        else:
            rcount = x + w

    for obj in remove_list:
        remain.remove(obj)

    remain_list = []
    remove_list = []
    add_map = {}

    for x in remain:
        remain_list.append(remain_map[str(x)])
    print("-------------------------------------------------")

    print(remain_list)

    # if len(remain_list) != 6 or need_split:
    ind = 0
    if True:
        while ind < len(remain_list):
            x, y, w, h = remain_list[ind]
            if ind < len(remain_list) - 1:
                xx, yy, ww, hh = remain_list[ind + 1]
                # merge
                if w + ww <= 36 and x + w >= xx:
                    remain_list[ind] = (x, y, w + ww, h)
                    remove_list.append((xx, yy, ww, hh))
                    ind = ind + 1
                # adjust size
                if x + w - xx >= 7:
                    remain_list[ind] = (x, y, xx - x, h)

            ind = ind + 1

    for i in remove_list:
        if i in remain_list:
            remain_list.remove(i)

    if len(remain_list) != 6:
        index = 0
        for i in range(len(remain_list)):
            x, y, w, h = remain_list[i]
            # if i < len(remain_list) - 1:
            #     xx, yy, ww, hh = remain_list[i + 1]
            #
            #     if w + ww <= 35 and abs(x + w - xx) <= 4:
            #         remain_list[i] = (x, y, w + ww, h)
            #         remove_list.append((xx, yy, ww, hh))

            if w > 70:
                wn = int(w / 3)
                add_map[str(i + index)] = (x, y, wn, h)
                add_map[str(i + index + 1)] = (x + wn, y, wn, h)
                add_map[str(i + index + 2)] = (x + 2 * wn, y, wn, h)
                remove_list.append((x, y, w, h))
                index = index + 3

            if (w > 35) and (w <= 70):
                wn = int(w / 2)
                add_map[str(i + index)] = (x, y, wn, h)
                add_map[str(i + index + 1)] = (x + wn, y, wn, h)
                index = index + 2
                remove_list.append((x, y, w, h))

    i = 0

    index = add_map.keys()
    sorted(index)

    for k in index:
        remain_list.insert(int(k), add_map.get(k))

    for i in remove_list:
        if i in remain_list:
            remain_list.remove(i)

    # for k, v in remain_map.items():
    #     x, y, w, h = cv2.boundingRect()
    remain = None
    remove_list = None
    remove_list = None
    print(remain_list)

    if len(remain_list) == 6:
        sub_file = []
        for i in range(len(remain_list)):
            # print(i)
            x, y, w, h = remain_list[i]
            # print(x, y, w, h)
            newimage = image[y:y + h, x:x + w]
            if h > 35 or w > 30:
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                newimage = cv2.erode(newimage, kernel)
            imgname = img_file.split('.')[0]
            img_true_path = basepath + "/" + imgname + "/" + imgname[i]
            if not os.path.exists(img_true_path):
                os.makedirs(img_true_path)

            sfile = img_true_path + "/" + img_file.split('.')[0] + "_" + str(uuid.uuid4()) + ".jpg"
            cv2.imwrite(sfile, newimage)
            sub_file.append(sfile)
            # cv2.imwrite()
        return sub_file
    else:
        print(img_file)
        # print(remain)
        for i in range(len(remain_list)):
            # print(i)
            x, y, w, h = remain_list[i]
            # print(x, y, w, h)
            # if w > 35:

            newimage = image[y:y + h, x:x + w]
            # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            # newimage=cv2.dilate(newimage, kernel)
            img_true_path = basepath + "/" + img_file.split('.')[0]
            if not os.path.exists(img_true_path):
                os.makedirs(img_true_path)
            cv2.imwrite(img_true_path + "/" + str(uuid.uuid4()) + ".jpg", newimage)
        raise Exception("split img: " + len(remain_list))

def reduce_line(h3):
    for i in range(0, len(h3)):
        for j in range(0, len(h3[0])):
            if h3[i, j] == 0:
                if h3[i + 1, j] == 255:
                    h3[i, j] = 255
                elif h3[i + 1, j] == 0:
                    if h3[i + 2, j] == 255:
                        h3[i: i + 2, j] = 255


def reduce_noise(img, ths):
    for i in range(0, len(img)):
        if len(img[i]) - sum(img[i]) / 255 <= ths:
            img[i, :] = 255

    for i in range(0, len(img[0])):
        if len(img) - sum(img[:, i]) / 255 <= ths:
            img[:, i] = 255
import sys

if __name__ == '__main__':
    try:
        filepath = sys.argv[1]
        print(filepath)
        img_process(os.path.dirname(filepath), os.path.basename(filepath))
    except Exception as e:
        print(e)

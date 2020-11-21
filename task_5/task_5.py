import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

""" CONSTS """
PATH = os.path.dirname(os.path.abspath(__file__))

""" 1 задание """
def one():
    image = cv2.imread(f'{PATH}/first.jpg')
    canny_1 = cv2.Canny(image, 1.5, 1)
    canny_2 = cv2.Canny(image, 2.75, 1)
    canny_4 = cv2.Canny(image, 4, 1)
    cv2.imshow("Default", image)
    cv2.imshow("1.5 and 1", canny_1)
    cv2.imshow("2.75 and 1", canny_2)
    cv2.imshow("4 and 1", canny_4)
    cv2.waitKey(0)
    """ a """
    canny_50 = cv2.Canny(image, 0, 5)
    cv2.imshow("<50", canny_50)
    """ b """
    canny_50 = cv2.Canny(image, 5, 10)
    cv2.imshow("50-100", canny_50)
    """ c """
    canny_50 = cv2.Canny(image, 10, 15)
    cv2.imshow("100-150", canny_50)
    """ d """
    canny_50 = cv2.Canny(image, 15, 20)
    cv2.imshow("150-200", canny_50)
    """ e """
    canny_50 = cv2.Canny(image, 20, 25)
    cv2.imshow("200-250", canny_50)
    cv2.waitKey(0)
    """ f """
    """ 
    С каждым разом все меньше пикселей на изображении попадает в диапазон от верхнего порога до нижнего.
    Если быть точнее то Edge Gradient пикселя. 
    """


""" 2 задание """
def two():
    image = cv2.imread(f'{PATH}/bicycle.jpg')
    image_lines = image.copy()
    image_circle = cv2.imread(f'{PATH}/bicycle.jpg', 0)
    gray = cv2.cvtColor(image_lines, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 160)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(image_lines, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
    circles = cv2.HoughCircles(image_circle, cv2.HOUGH_GRADIENT, 1, image.shape[1]/20, param1=90, param2=60, minRadius=30, maxRadius=200)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(image_circle, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(image_circle, (i[0], i[1]), 2, (0, 255, 0), 3)
    cv2.imshow('Circles', image_circle)
    cv2.imshow("Lines", image_lines)
    cv2.waitKey(0)


""" 4 задание """
def four():
    image = cv2.imread(f'{PATH}/first.jpg', 0)
    mu, sigma = 1, 2
    rows, cols = image.shape
    s = np.random.normal(mu, sigma, (rows, cols, 2))
    dft_image = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_s = dft_image * s
    image_back = cv2.idft(dft_s)
    image_back = cv2.magnitude(image_back[:, :, 0], image_back[:, :, 1])
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(image_back, cmap='gray')
    plt.title('After DFT'), plt.xticks([]), plt.yticks([])
    plt.show()
    """ 
    Пиксели размазываются по оси x. Чем больше разница между мат ожиданием и средним отклонением, тем сильнее размытие.
    """


""" 5 задание """
def five():
    image = cv2.imread(f'{PATH}/first.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    integral_image = cv2.integral(gray)
    max_format_value = 255
    c = max_format_value / integral_image[integral_image.shape[0] - 1][integral_image.shape[1] - 1]
    integral = integral_image * c
    np_integral = np.asarray(integral)
    print(image.shape)
    print("Take a part of image (x < 854 | y < 1280):")
    x = int(input())
    y = int(input())
    diff_image = cv2.Sobel(np_integral[x:, :y], -1, 1, 1)
    plt.subplot(121)
    plt.plot(diff_image)
    plt.subplot(122)
    plt.imshow(diff_image)
    plt.show()
    cv2.waitKey(0)


""" 6 задание """
def six():
    image = cv2.imread(f'{PATH}/first.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    num_rows, num_cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), 45, 1)
    image = cv2.warpAffine(image, rotation_matrix, (num_cols, num_rows))
    print("Input x1,y1,x2,y2:")
    y1 = int(input())
    x1 = int(input())
    y2 = int(input())
    x2 = int(input())
    new_image = image[y1:y2, x1:x2]
    cv2.imshow("New image with new x and y", new_image)
    cv2.waitKey(0)
    _, _, integral = cv2.integral3(new_image)
    max_format_value = 250
    c = max_format_value / integral[integral.shape[0] - 1][integral.shape[1] - 1]
    integral = integral * c
    print(sum(sum(integral)))
    cv2.imshow("Integral", integral)
    cv2.waitKey(0)


""" 9 задание """
def nine():
    image = cv2.imread(f'{PATH}/first.jpg')
    cv2.imshow("Human", image)
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (395, 310, 520, 605)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 10, cv2.GC_INIT_WITH_RECT)
    mask_new = np.where((mask == 2) | (mask == 0), 1, 0).astype('uint8')
    new_image = image * mask_new[:, :, np.newaxis]
    dst = cv2.inpaint(image, mask, 2, cv2.INPAINT_TELEA)
    cv2.imshow('dst', dst)
    cv2.imshow('Without Human', new_image)
    cv2.waitKey(0)


""" 10 задание """
def ten():
    foreground = cv2.imread(f'{PATH}/ten.jpg')
    foreground = cv2.pyrMeanShiftFiltering(foreground, 10, 20, 1)
    mask = np.zeros([foreground.shape[:2][0] + 2, foreground.shape[:2][1] + 2], np.uint8)
    cv2.floodFill(foreground, mask, (483, 788), (0, 0, 0), (10,)*3, (135,)*3, flags=cv2.FLOODFILL_FIXED_RANGE)
    cv2.floodFill(foreground, mask, (184, 84), (0, 0, 0), (10,)*3, (130,)*3, flags=cv2.FLOODFILL_FIXED_RANGE)
    foreground[np.where((foreground != [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    background = cv2.imread(f'{PATH}/ten.jpg')
    background = cv2.GaussianBlur(background, (25, 25), cv2.BORDER_DEFAULT)
    res = cv2.copyTo(background, foreground)
    alpha = cv2.imread(f'{PATH}/ten.jpg')
    res[np.where((res == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    cv2.imshow("outmg", res)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            if (res[i][j] == [255, 255, 255]).all():
                res[i][j] = alpha[i][j]
    cv2.imshow("outImg", res)
    cv2.waitKey(0)


if __name__ == '__main__':
    while True:
        print("Введите номер упражнения.")
        flag = input('>>')
        if flag == '1':
            one()
        if flag == '2':
            two()
        if flag == '4':
            four()
        if flag == '5':
            five()
        if flag == '6':
            six()
        if flag == '9':
            nine()
        if flag == '10':
            ten()
        if flag == 'exit':
            break

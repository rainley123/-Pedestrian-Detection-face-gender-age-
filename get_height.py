'''
导入后，调用 cal(p_top, p_botm) 函数
返回身高
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def cal(p_top, p_botm):
    '''
    p_top:  人顶部坐标点（x1, y1）
    p_botm: 人底部坐标点（x2, y2）
    坐标可以是元组、或者列表
    x轴为横向轴，y轴为纵向轴
    坐标原点为图像左上角
    '''
    door_top = (661, 2)
    door_botm = (628, 263)
    vcross = (-3, -264)
    hcross = (2672, 115)
    
    pt_a = p_botm
    pt_b = door_botm
    pt_c = line_cross(vcross, hcross, pt_a, pt_b)
    pt_d = line_cross(p_top, p_botm, pt_c, door_top)
    pt_e = door_top
    pt_f = p_top
    pt_g = line_cross(pt_e, pt_b, pt_a, pt_f)

    ad = distance(pt_a, pt_d)
    af = distance(pt_a, pt_f)
    gd = distance(pt_g, pt_d)
    gf = distance(pt_g, pt_f)

    door_height = 210   # 210 cm
    if (ad/af)/(gd/gf) != 0:
        result = door_height / ((ad/af)/(gd/gf))
    else:
        result = random.randint(160, 180)
    #print(result)
    return int(result)

def door_line(img):
    '''
    手动标点，找出门框
    '''
    plt.imshow(img)
    print("# Check two points of door: ")
    print("# >>> TOP point first! <<< ")
    door = plt.ginput(2)
    pt1 = door[0]
    pt2 = door[1]
    plt.plot( [pt1[0],pt2[0]],  [pt1[1],pt2[1]],  marker = 'o')
    return [pt1, pt2]

def floor_line(img):
    '''
    手动标点，找出地平线
    '''
    plt.imshow(img)
    #vertical line1 inputs
    print("check two points of vertical line1: ")
    vline1 = plt.ginput(2)
    v11 = vline1[0]
    v12 = vline1[1]
    plt.plot( [v11[0],v12[0]],  [v11[1],v12[1]],  marker = 'o')

    #vertical line2 inputs
    print("check two points of vertical line2: ")
    vline2 = plt.ginput(2)
    v21 = vline2[0]
    v22 = vline2[1]
    plt.plot( [v21[0],v22[0]],  [v21[1],v22[1]],  marker = 'o')

    #horizontal line1 inputs
    print("check two points of horizontal line1: ")
    hline1 = plt.ginput(2)
    h11 = hline1[0]
    h12 = hline1[1]
    plt.plot( [h11[0],h12[0]],  [h11[1],h12[1]],  marker = 'o')

    #horizontal line2 inputs
    print("check two points of horizontal line2: ")
    hline2 = plt.ginput(2)
    h21 = hline2[0]
    h22 = hline2[1]
    plt.plot( [h21[0],h22[0]],  [h21[1],h22[1]],  marker = 'o')
    
    #得到地平线上的2点 vcross hcross
    vcross = line_cross(v11, v12, v21, v22)
    hcross = line_cross(h11, h12, h21, h22)
    #print("two pts: ", vcross, hcross)

    return [vcross, hcross]


def line_cross(pt11, pt12, pt21, pt22):
    '''
    pt11, pt12 为直线1上的2点；
    pt21, pt22 为直线2上的2点；
    返回两直线的交点。
    '''
    # #line 1
    # k1 = (pt11[1]-pt12[1]) / (pt11[0]-pt12[0])
    # b1 = pt11[1]-k1*pt11[0]
    # #line 2
    # k2 = (pt21[1]-pt22[1]) / (pt21[0]-pt22[0])
    # b2 = pt21[1]-k2*pt21[0]
    # # cross point
    # x0 = int( (b2-b1)/(k1-k2) )
    # y0 = int( k1*x0+b1 )
    # return [x0, y0]
    a1 = pt11[1] - pt12[1]
    b1 = pt12[0] - pt11[0]
    c1 = pt11[0] * pt12[1] - pt12[0] * pt11[1]

    a2 = pt21[1] - pt22[1]
    b2 = pt22[0] - pt21[0]
    c2 = pt21[0] * pt22[1] - pt22[0] * pt21[1]

    x0 = (c2 * b1 - c1 * b2) / (a1 * b2 - a2 * b1)
    y0 = (a1 * c2 - a2 * c1) / (a2 * b1 - a1 * b2)
    return [x0, y0]

def distance(pt1, pt2):
    xx = pt1[0]-pt2[0]
    yy = pt1[1]-pt2[1]
    return int( np.sqrt(xx*xx+yy*yy) )

    
if __name__ == "__main__":
    res=cal((946, 181), (871, 416))
    print("height: ", res)
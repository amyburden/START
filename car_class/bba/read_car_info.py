import csv
import os
from collections import defaultdict
import json
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
import glob


def read_car_info(filepath):
    car_info = []
    with open(filepath, 'r', encoding = 'gbk') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if row[0] == '' or row[6]=='' or row[5]=='':
                continue
            car_info.append({"car_id":row[1], "pp_brand_id":row[2], "pp_genre_id":row[3], "left_behind":row[5], "left_front": row[6], 'chinese':row[4]})
    return car_info
def read_car_pic(code):
    url = 'http://icdn.startcarlife.com/img/'+ code[:4]+'/'+code[4]+'/'+code+'.jpg'
    response = requests.get(url)
    try:
        img = Image.open(BytesIO(response.content))
    except:
        print (code," failed")
        img = None
    return img

info_list = glob.glob('*.csv')
car_info = []
for item in info_list:
    car_info += read_car_info(item)
#car_info = [json.loads(line, ) for line in open('10car.json')]
i = 0
for item in car_info:
    path = os.path.join('../',item['pp_brand_id'],item['pp_genre_id'])
    if not os.path.exists(path):
        os.makedirs(path)
        print (path)
    if not os.path.exists(os.path.join(path,item['car_id']+'_b.jpg')):
        img = read_car_pic(item['left_front'])
        img2 = read_car_pic(item['left_behind'])
        if img and img2:
            img.save(os.path.join(path,item['car_id']+'.jpg'))
            img2.save(os.path.join(path,item['car_id']+'_b.jpg'))
            i += 1
            if i % 50 == 0:
                print('save ',i,' pic')

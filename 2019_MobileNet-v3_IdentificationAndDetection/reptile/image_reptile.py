#!/usr/bin/python
# -*- coding:utf-8 -*-
import re
import os
import requests

# 如果文件夹不存在就创建文件夹
def existDir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass

# 定义爬虫函数
def reptile(reptileWord, savePath):
    # 如果文件夹不存在就创建文件夹
    existDir(savePath)
    # 获取爬虫页面url地址
    url = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + reptileWord + '&pn='
    # 定义一些要用到的变量
    num = 1
    # 定义headers参数
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36'
    }
    # 传入headers参数并得到响应
    response = requests.Session()
    response.headers = headers
    html = response.get(url, timeout=10, allow_redirects=False)
    # 正则表达式获取所有url
    pic_urls = re.findall('"objURL":"(.*?)",', html.text, re.S)
    # 遍历所有url
    for pic_url in pic_urls:
        # 获取下载信息
        download_value = '正在下载第' + str(num) + '张图片，图片url：' + str(pic_url)
        # 定义存储路径
        string = savePath + r'\\' + reptileWord + '+' + str(num) + '.jpg'
        # “wb”方式打开
        local_path = open(string, 'wb')
        photo = requests.get(pic_url, timeout=7)
        # 写入爬取成功的图片
        local_path.write(photo.content)
        local_path.close()
        num += 1
        # yield return 语句可一次返回一个元素
        yield download_value

if __name__ == '__main__':
    word = 'hello world'
    path = 'F:/images'
    values = reptile(word, path)
    for value in range(60):
        print(next(values))

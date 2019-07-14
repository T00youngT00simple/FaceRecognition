import numpy as np  # 数据处理的库 Numpy
import cv2          # 图像处理的库 OpenCv
import os
import shutil
import _thread
import wx
import csv
from importlib import reload
from skimage import io as iio
import sys
import datetime
from PIL import Image
from MysqlDB import MyDB
# 创建 cv2 摄像头对象
#    C++: VideoCapture::VideoCapture(int device);

#API:http://www.opencv.org.cn/opencvdoc/2.3.2/html/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture

# 保存
path = "/home/pi/test/data"

path_make_dir = path + '/dataset/'

path_feature_all = "data/feature_all.csv"

info = '/home/pi/test/icon/info.png'






#register ui
class   RegisterUi(wx.Frame):
    def __init__(self,superion):
        wx.Frame.__init__(self,parent=superion,title="人脸录入",size=(800,590),style=wx.DEFAULT_FRAME_STYLE|wx.STAY_ON_TOP)
        self.SetBackgroundColour('white')
        self.Center()

        self.NewButton =  wx.Button(parent=self,pos=(50,120),size=(80,50),label='新建录入')

        self.ShortCutButton = wx.Button(parent=self,pos=(50,320),size=(80,50),label='截图保存')

        self.SaveButton =  wx.Button(parent=self,pos=(50,220),size=(80,50),label='完成录入')

        # 封面图片
        self.image_info = wx.Image(info, wx.BITMAP_TYPE_ANY).Scale(600, 480)
        # 显示图片
        self.bmp = wx.StaticBitmap(parent=self, pos=(180,20), bitmap=wx.Bitmap(self.image_info))

        self.Bind(wx.EVT_BUTTON,self.OnShortCutButtonClicked,self.ShortCutButton)
        self.Bind(wx.EVT_BUTTON,self.OnNewButtonClicked,self.NewButton)
        self.ShortCutButton.Enable(enable=False)
        self.SaveButton.Enable(False)

        self.Bind(wx.EVT_BUTTON,self.OnSaveButtonClicked,self.SaveButton)


        self.Bind(wx.EVT_CLOSE,self.OnClose)

        self.sc_number = 0
        self.register_flag = 0

        self.count = 0
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        self.detector = cv2.CascadeClassifier("/home/pi/test/each_code/Cascades/lbpcascade_frontalface.xml")



        #用日期的字符拼接作为人脸录入的id
        self.face_id = ''
        #wx框获得facename
        self.face_name = ""
        #录入姓名查重
        self.face_names_exist = []
        #id 
        self.face_ids_exist = []
        #人脸模型
        self.faceSamples=[]
        #人脸模型列表
        self.ids = []
        #人脸与id对应关系
        self.name_id = {}
    
    
    #yiwai guan bi
    def OnClose(self,event):
        r = wx.MessageBox("Close?",'True',wx.CANCEL|wx.OK|wx.ICON_QUESTION)
        if r == wx.OK:
            try :
                self.cap.release()
                print('f')
            except:
                print('f')
            finally:
                print ('f')
                self.Destroy()

    
    #新建录入
    def OnNewButtonClicked(self, event):
        imagePaths = [os.path.join(path_make_dir, f) for f in os.listdir(path_make_dir)]

        for imagePath in imagePaths:
            face_name_exist = os.path.split(imagePath)[-1].split(".")[0]
            face_id_exist = os.path.split(imagePath)[-1].split(".")[1]
            if face_name_exist not in self.face_names_exist:
                self.face_names_exist.append(face_name_exist)
                self.face_ids_exist.append(face_id_exist)




        while self.face_name == '':
            self.face_name = wx.GetTextFromUser(message="请先输入录入者的姓名", caption="温馨提示",
                                                 default_value = "",parent = None)
            if self.face_name in self.face_names_exist:
                wx.MessageBox(message="姓名已存在，请重新输入", caption="警告")
                self.face_name = ''


        while self.face_id == '':
            self.face_id = wx.GetTextFromUser(message="请先输入录入者的ID", caption="温馨提示",
                                                 default_value = "",parent = None)
            if self.face_id in self.face_ids_exist:
                wx.MessageBox(message="学号已存在，请重新输入", caption="警告")
                self.face_id = ''
                

        self.NewButton.Enable(enable=False)
        self.SaveButton.Enable(enable=False)
         #使用多线程，子线程运行后台的程序，主线程更新前台的UI，这样不会互相影响
        # 创建子线程，按钮调用这个方法，



        
        print("f")
        # cap.set(propId, value)
        # 设置视频参数，propId 设置的视频参数，value 设置的参数值
        # 
##        self.face_id = int(
##            str(datetime.datetime.now().day) + str(datetime.datetime.now().hour) +
##            str(datetime.datetime.now().minute) +
##            str(datetime.datetime.now().second))
        print(self.face_id)
        # self.cap.set(cv2.CAP_PROP_FPS,5)
        self.get_NameID_dict()
        print(self.name_id)
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('/home/pi/test/trainer/trainer.yml')
        cascadePath = "/home/pi/test/each_code/Cascades/lbpcascade_frontalface.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath);
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.cap = cv2.VideoCapture(0)
        minW = 0.1 * self.cap.get(3)
        minH = 0.1 * self.cap.get(4)
        while True:
                
            ret, img = self.cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text = "Entering face..............."
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH))
            )
            if (len(faces) >= 2):
                
                text = "Only one face can be entered at the same time"
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = img[y:y + h, x:x + w]                                                                     
            else:
                self.get_NameID_dict()
                # get name_id first
                for (x, y, w, h) in faces:                    
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    id, confi = recognizer.predict(gray[y:y + h, x:x + w])
                    # 检查是否小于100 ==>“0”是完美匹配
##                  self.DeleteRepeat()    
                    name = "Entering"
                    confidence = "  {0}%".format(round(100 - confi))
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = img[y:y + h, x:x + w]
##                    cv2.imwrite("/home/pi/test/dataset/"+str(face_id)+"."+str(count)+"."+"/face.jpg",gray[y:y+h,x:x+w])
                    text = "Faces: " + str(len(faces))
                    if (confi > 100):
                        self.count += 1
                        name = 'Entering'
                        text = "Faces: " + str(len(faces))
                        cv2.imwrite(path_make_dir + str(self.face_name) + '.' + str(self.face_id) + '.' + str(self.count) + ".jpg",
                                    gray[y:y + h, x:x + w])
                    else :
                        self.DeleteRepeat()
                        self.count = 0
                        text = "Face has been got,please give me a true face"
                        name = 'exist'
                    if self.count >= 10:
                        with MyDB('localhost','root','123456','student') as cs:
            
                            sql = "INSERT INTO user(name,password,chidao,kuangke,qingjia,qiandao) VALUES ('%s','%s','%d','%d','%d','%d')"%(self.face_name,self.face_id,0,0,0,0)
                            try:
                                cs.execute(sql)
    
                            except:
                                cs.rollback()
                                
                        wx.MessageBox(message="luru chenggong ", caption="提示")
                        self.cap.release()
                        self.SaveButton.Enable(enable=True)    
                        return 0
  
                    cv2.putText(img, str(name), (x+5,y-5), font, 1, (255,255,255), 2)
                    cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)                   
            cv2.putText(img,text, (50, 80), font, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, "Warning: face can not exceed one", (50, 140), font, 0.4, (0, 0, 255), 1,
                        cv2.LINE_AA)
            height, width = img.shape[:2]
            image1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pic = wx.Bitmap.FromBuffer(width, height, image1)
            self.bmp.SetBitmap(pic)
            cv2.waitKey(1)

    def get_NameID_dict(self):
        names = []
        imagePaths = [os.path.join(path_make_dir, f) for f in os.listdir(path_make_dir)]
        for imagePath in imagePaths:
            face_id = os.path.split(imagePath)[-1].split(".")[1]
            name = os.path.split(imagePath)[-1].split(".")[0]
            if name not in names:
                names.append(name)
                self.name_id[face_id] = name

  
            # cap.read()
            # 返回两个值：
            #    一个布尔值 true/false，用来判断读取视频是否成功/是否到视频末尾
            #    图像对象，图像的三维矩阵q
##            print("ff")
##            ret, img = self.cap.read()
##            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##            faces = faceCascade.detectMultiScale(
##                gray,
##                scaleFactor=1.2,
##                minNeighbors=5,
##                minSize=(int(minW), int(minH))
##            )
##            if (len(faces) >= 1):
##                self.DeleteRepeat()
##                for (x, y, w, h) in faces:
##                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
##                    roi_gray = gray[y:y + h, x:x + w]
##                    roi_color = img[y:y + h, x:x + w]
##                    # cv2.putText(img, "Only one face can be entered at the same time", (50, 80), font, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
##                    text = "Only one face can be entered at the same time"
##
##            else:
##                for (x, y, w, h) in faces:
##                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
##                    id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
##                    # 检查是否小于100 ==>“0”是完美匹配
##                    if (confidence < 100):
##                        roi_gray = gray[y:y + h, x:x + w]
##                        roi_color = img[y:y + h, x:x + w]
##                        text = "Face has been entered"
##                    else :
##                        self.count += 1
##                        roi_gray = gray[y:y + h, x:x + w]
##                        roi_color = img[y:y + h, x:x + w]
##                        ##        cv2.imwrite("/home/pi/test/dataset/"+str(face_id)+"."+str(count)+"."+"/face.jpg",gray[y:y+h,x:x+w])
##                        cv2.imwrite(path + str(self.face_name) + '.' + str(self.face_id) + '.' + str(self.count) + ".jpg",
##                                    gray[y:y + h, x:x + w])
##                        text = "Faces: " + str(len(faces))
##                cv2.putText(img,"fcuk", (50, 80), font, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
##                cv2.imshow('video', img)
##                cv2.waitKey(100)



    #截图录入
    def OnShortCutButtonClicked(self,event):
        self.SaveButton.Enable(True)
        if len(self.rects) !=0:
            # 计算矩形框大小,保证同步
            height = self.rects[0].bottom() - self.rects[0].top()
            width = self.rects[0].right() - self.rects[0].left()
            self.sc_number += 1
            im_blank = np.zeros((height, width, 3), np.uint8)
            for ii in range(height):
                for jj in range(width):
                    im_blank[ii][jj] = self.im_rd[self.rects[0].top() + ii][self.rects[0].left() + jj]
            cv2.imencode('.jpg', im_blank)[1].tofile(path_make_dir+self.name + "/img_face_" + str(self.sc_number) + ".jpg") #正确方法
            print("写入本地：", str(path_make_dir+self.name) + "/img_face_" + str(self.sc_number) + ".jpg")

        else:
            print("未检测到人脸，识别无效，未写入本地")




    #完成录入  删重 模型检测
    def  DeleteRepeat(self):
        #当出现录入意外时，删除已经录入的照片并将照片计数器清零
        self.count = 0
        
        for imagePath in os.listdir(path_make_dir):
            if str(self.face_id) in imagePath:
                os.remove(path_make_dir+imagePath)

    def OnSaveButtonClicked(self,event):


        self.image_info
        self.bmp

        self.NewButton.Enable(True)
        self.SaveButton.Enable(False)
        self.ShortCutButton.Enable(False)


        # 释放摄像头
        self.cap.release()



        
        print('f')
        # 删除建立的窗口
        #cv2.destroyAllWindows()

        self.imagePaths = os.listdir(path_make_dir)
        os.chdir(path_make_dir)
        for imagePath in self.imagePaths:

            PIL_img = Image.open(imagePath).convert('L')
            # 将其转换为灰度
            img_numpy = np.array(PIL_img, 'uint8')
            #用image库处理图片灰度处理
            id = int(imagePath.split(".")[1])
            faces = self.detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                self.faceSamples.append(img_numpy[y:y + h, x:x + w])
                self.ids.append(id)

        if len(self.ids) == 0:
            wx.MessageBox(message=face_name + "未检测到人脸，请重新录入", caption="警告")
            self.DeleteRepeat()
        else:
            self.recognizer.train(self.faceSamples, np.array(self.ids))
        # 将模型保存到trainer/trainer.yml中
            self.recognizer.write('/home/pi/test/trainer/trainer.yml')  # save在电脑上工作，树莓派上不工作
            wx.MessageBox(message=" End，共{0}张人脸信息".format(len(np.unique(self.ids))), caption="提示")
        # 输出人脸数
            
            self.count = 0
            self.face_name = ''
            self.face_id =''
            return 
        # if self.register_flag == 1:
        #     if os.path.exists(path_make_dir+self.name):
        #         shutil.rmtree(path_make_dir+self.name)
        #         print("重复录入，已删除姓名文件夹", path_make_dir+self.name)

        # if self.sc_number == 0 and len(self.name)>0:
        #     if os.path.exists(path_make_dir+self.name):
        #         shutil.rmtree(path_make_dir+self.name)
        #         print("您未保存截图，已删除姓名文件夹", path_make_dir+self.name)
        # if self.register_flag==0 and self.sc_number!=0:
        #     pics = os.listdir(path_make_dir+self.name)
        #     feature_list = []
        #     feature_average = []
        #     for i in range(len(pics)):
        #         pic_path = path_make_dir+self.name + "/" + pics[i]
        #         print("正在读的人脸图像：", pic_path)
        #         img = iio.imread(pic_path)
        #         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #         dets = face_recognize_punchcard.detector(img_gray, 1)
        #         if len(dets) != 0:
        #             shape = face_recognize_punchcard.predictor(img_gray, dets[0])
        #             face_descriptor = face_recognize_punchcard.facerec.compute_face_descriptor(img_gray, shape)
        #             feature_list.append(face_descriptor)
        #         else:
        #             face_descriptor = 0
        #             print("未在照片中识别到人脸")
        #     if len(feature_list)>0:
        #         for j in range(128):
        #             feature_average.append(0)
        #             for i in range(len(feature_list)):
        #                 feature_average[j] += feature_list[i][j]
        #             feature_average[j] = (feature_average[j])/len(feature_list)
        #         feature_average.append(self.name)
        #
        #         with open(path_feature_all, "a+", newline="") as csvfile:
        #             writer = csv.writer(csvfile)
        #             print('写入一条特征人脸入库',feature_average)
        #             writer.writerow(feature_average)

        
        # self.register_flag = 0
        # self.sc_number = 0





##
##app = wx.App()
##
##frame = RegisterUi(None)
##frame.Show()
##app.MainLoop()

#cap.isOpened（） 返回 true/false 检查初始化是否成功

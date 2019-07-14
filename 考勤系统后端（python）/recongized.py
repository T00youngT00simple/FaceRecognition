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
import save_photo_new









loading = '/home/pi/test/icon/loading.png'
pun_fail = '/home/pi/test/icon/pun_fail.png'
pun_repeat = '/home/pi/test/icon/pun_repeat.png'
pun_success = '/home/pi/test/icon/pun_success.png'

path_csv = "/home/pi/test/data/datacsv/logcat.csv"




path_make_dir = '/home/pi/test/data/dataset'


path_feature_known_csv = "/home/pi/test/data/feature_all.csv"

# path_features_known_csv= "/media/con/data/code/python/P_dlib_face_reco/data/csvs/features_all.csv"
##csv_rd = pd.read_csv(path_feature_known_csv, header=None,encoding='gbk')

# 存储的特征人脸个数
#print(csv_rd.shape)
#（2，129）

# 用来存放所有录入人脸特征的数组
features_known_arr = []

#print("s0",csv_rd.shape[0],"s1",csv_rd.shape[1])
##for i in range(csv_rd.shape[0]):
##    features_someone_arr = []
##    for j in range(0, len(csv_rd.ix[i, :])):
##        features_someone_arr.append(csv_rd.ix[i, :][j])
##    #    print(features_someone_arr)
##    features_known_arr.append(features_someone_arr)
##print("数据库人脸数:", len(features_known_arr))

class   PunchcardUi(wx.Frame):
    def __init__(self,superion):
        wx.Frame.__init__(self,parent=superion,title="刷脸签到",size=(800,590),style=wx.DEFAULT_FRAME_STYLE|wx.STAY_ON_TOP)
        self.SetBackgroundColour('white')
        self.Center()

        self.OpenCapButton =  wx.Button(parent=self,pos=(50,120),size=(90,60),label='开始/继续签到')

        self.resultText = wx.StaticText(parent=self,style=wx.ALIGN_CENTER_VERTICAL,pos=(50,320),size=(90,60),label="签到天数:0")

        self.resultText.SetBackgroundColour('white')
        self.resultText.SetForegroundColour('blue')
        font = wx.Font(14, wx.DECORATIVE, wx.ITALIC, wx.NORMAL)
        self.resultText.SetFont(font)

        self.pun_day_num = 0

        # 封面图片
        self.image_loading = wx.Image(loading, wx.BITMAP_TYPE_ANY).Scale(600, 480)

        self.image_fail = wx.Image(pun_fail, wx.BITMAP_TYPE_ANY).Scale(600, 480)
        self.image_repeat = wx.Image(pun_repeat, wx.BITMAP_TYPE_ANY).Scale(600, 480)
        self.image_success = wx.Image(pun_success, wx.BITMAP_TYPE_ANY).Scale(600, 480)

        # 显示图片
        self.bmp = wx.StaticBitmap(parent=self, pos=(180,20), bitmap=wx.Bitmap(self.image_loading))

        self.Bind(wx.EVT_BUTTON,self.OnOpenCapButtonClicked,self.OpenCapButton)
        self.Bind(wx.EVT_CLOSE,self.OnClose)
        
        self.recoders = []
        self.count = 0
        self.name_id = {}
        self.chidao = 0
        self.kuangke = 0
        self.qingjia = 0
        self.qiandao = 0

        self.uptime =0
    def get_NameID_dict(self):
        names = []
        imagePaths = [os.path.join(path_make_dir, f) for f in os.listdir(path_make_dir)]
        for imagePath in imagePaths:
            face_id = os.path.split(imagePath)[-1].split(".")[1]
            name = os.path.split(imagePath)[-1].split(".")[0]
            if name not in names:
                names.append(name)
                self.name_id[face_id] = name


    def read_csv(self):
        self.recoders = []
        if os.path.exists(path_csv):
            with open(path_csv,'r',newline = "") as f:
                reader = csv.reader(f)
                for row in reader:
                    self.recoders.append(row)
        else:
            with open(path_csv,'w',newline = "") as f:
                writer = csv.writer(f)
                header = ['FACE_NAME','FACE_ID','CHECK_DATE','CHECK_TIME']
                writer.writerow(header)
        

    def OnClose(self,event):
        r = wx.MessageBox("Close?",'True',wx.CANCEL|wx.OK|wx.ICON_QUESTION)
        if r == wx.OK:
            try :
                self.cam.release()
                print('f')
            except:
                print('f')
            finally:
                print ('f')
                self.Destroy()
    def get_uptime(self):
        with MyDB('localhost','root','123456','student') as cs:
            sql = "select * from shijian where id=1"
            try:
                cs.execute(sql)
                result = cs.fetchall()
                for row in result:
                    self.uptime = int(row[1])
            except:
                cs.rollback()

    def get_cishu(self):
        with MyDB('localhost','root','123456','student') as cs:
            sql = "select * from user where name='%s'"%(self.face_name)
            try:
                cs.execute(sql)
                result = cs.fetchall()
                for row in result:
                    self.chidao = row[2]
                    self.kuangke = row[3]
                    self.qingjia = row[4]
                    self.qiandao = row[5]
                    
            except:
                cs.rollback()
    
    def OnOpenCapButtonClicked(self,event):

        """使用多线程，子线程运行后台的程序，主线程更新前台的UI，这样不会互相影响"""
        # 创建子线程，按钮调用这个方法，
##        _thread.start_new_thread(self._open_cap, (event,))
##        self.NewButton.Enable(enable=False)
##        self.SaveButton.Enable(enable=False)
         #使用多线程，子线程运行后台的程序，主线程更新前台的UI，这样不会互相影响
        # 创建子线程，按钮调用这个方法，

        self.cam = cv2.VideoCapture(0)
       
         
        # Define min window size to be recognized as a face
        minW = 0.1*self.cam.get(3)
        minH = 0.1*self.cam.get(4)
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('/home/pi/test/trainer/trainer.yml')
        cascadePath = "/home/pi/test/each_code/Cascades/lbpcascade_frontalface.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath);
        self.get_uptime()
        while True:
            ret, img =self.cam.read()
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            font = cv2.FONT_HERSHEY_SIMPLEX

            self.get_NameID_dict()
            faces = faceCascade.detectMultiScale( 
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
               )
            if (len(faces) >= 2):
                text = "Only one face can be entered at the same time"
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = img[y:y + h, x:x + w]
            else:
                text = "Faces: " + str(len(faces))
                for (x, y, w, h) in faces:          
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    face_id, confi = recognizer.predict(gray[y:y + h, x:x + w])
                    print(face_id)
                    confidence = "  {0}%".format(round(100 - confi))
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = img[y:y + h, x:x + w]
                    text = "Faces: " + str(len(faces))
                    if(confi<100):
                        self.count += 1
                        name ='Checking...'
                        text = "Faces: " + str(len(faces))
                        self.face_name = self.name_id.get(str(face_id),'unknown')
                        print(face_id)
                        recoder = []
                        recoder.append(self.face_name)
                        recoder.append(str(face_id))
                        localtime = datetime.datetime.now()
                        date = str(localtime.year)+'-'+str(localtime.month)+'-'+str(localtime.day)
                        
                    else:
                        self.count = 0
                        text = "The face information is not entered"
                        name = 'Unknown'
                    if self.count >5:
                        time = str(localtime.hour)+ ':'+str(localtime.minute)+':'+str(localtime.second)
                        check_time = int(str(localtime.hour)+str(localtime.minute)+str(localtime.second))
                        recoder.append(date)
                        recoder.append(time)
                        self.read_csv()
                        for item in self.recoders:
                            if item[0] ==  recoder[0]:
                                self.pun_day_num += 1
                        for item in self.recoders:
                            if item[0] ==  recoder[0] and item[2] == recoder[2]:
                                wx.MessageBox(message="this face has checked today", caption="警告")
                                text = 'this face has checked today'
                                self.count = 0
                        if text != 'this face has checked today':
                            with MyDB('localhost','root','123456','student') as cs:
                                sql = "select *from user WHERE name='%s'"%(self.face_name)
                                try:
                                    cs.execute(sql)
                                    result = cs.fetchall()
                                    if len(result) == 0:
                                        wx.MessageBox(message="信息被管理员删除，清联系管理员", caption="警告")
                                        self.count = 0
                                        self.cam.release()
                                        return 0
                                except:
                                    cs.rollback()
                                    
                            self.get_cishu()                                
                            if (self.uptime>check_time and self.uptime-check_time<1500):
                            
                                wx.MessageBox(message=str(self.face_name)+'完成签到,签到时间'+str(date+time), caption="警告")
                               
                                self.qiandao +=1;
                                sql = "INSERT INTO CHECK_LOG(FACE_NAME,FACE_ID,CHECK_DATE,CHECK_TIME) VALUES ('%s','%s','%s','%s')"%(recoder[0],recoder[1],recoder[2],recoder[3])
                                sql_update = "UPDATE user SET qiandao=%d WHERE name='%s'"%(self.qiandao,self.face_name)
                                print("get true")
                            
                            if (self.uptime<check_time and check_time-self.uptime< 2000 ):
                            
                                wx.MessageBox(message=str(self.face_name)+'完成签到,迟到,签到时间'+str(date+time), caption="警告")
                                self.chidao +=1;
                                sql = "INSERT INTO CHECK_LOG(FACE_NAME,FACE_ID,CHECK_DATE,CHECK_TIME) VALUES ('%s','%s','%s','%s')"%(recoder[0],recoder[1],recoder[2],recoder[3])
                                sql_update = "UPDATE user SET chidao=%d WHERE name='%s'"%(self.qiandao,self.face_name)
                                print("get true")
                            
                            if (self.uptime<check_time and check_time-self.uptime>2000 and check_time-self.uptime<4500):
                            
                                wx.MessageBox(message=str(self.face_name)+'完成签到,旷课,签到时间'+str(date+time), caption="警告")
                                print('wancheng qiandao,'+str(self.face_name)+',qiandaoriqi,'+str(date+time))
                                self.kuangke +=1;
                                sql = "INSERT INTO CHECK_LOG(FACE_NAME,FACE_ID,CHECK_DATE,CHECK_TIME) VALUES ('%s','%s','%s','%s')"%(recoder[0],recoder[1],recoder[2],recoder[3])
                                sql_update = "UPDATE user SET kuangke=%d WHERE name='%s'"%(self.qiandao,self.face_name)
                                print("get true")
                            
                            else:
                                wx.MessageBox(message='非签到时间!', caption="警告")
                                self.count = 0
                                self.cam.release()
                                return 0
                            
                            with open(path_csv,'a+',newline = "") as f:
                                writer = csv.writer(f)
                                writer.writerow(recoder)
                            with MyDB('localhost','root','123456','student') as cs:
                                print(recoder)
                                try:
                                    cs.execute(sql)
                                    cs.execute(sql_update)
                                except:
                                    cs.rollback()

                                
                            self.count = 0
                            self.cam.release()
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



##
###
##app = wx.App()
##frame = PunchcardUi(None)
##frame.Show()
##app.MainLoop()



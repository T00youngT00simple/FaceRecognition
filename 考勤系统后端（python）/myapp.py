import wx
import os
from importlib import reload
import webbrowser
import save_photo_new
import sys
import recongized
from PIL import Image
import cv2
import numpy as np
from MysqlDB import MyDB



main ="/home/pi/test/icon/main.png"
path = "/home/pi/test/data/dataset/"
path_restart = '/home/pi/test/data/datarestart/'
file_path = '/home/pi/test/data/datacsv/logcat.csv'




class   Mainui(wx.Frame):
    def __init__(self,superion):
        wx.Frame.__init__(self,parent=superion,title="员工考勤系统",size=(800,590))
        self.SetBackgroundColour('white')
        self.Center()

        self.frame = ''
        self.RegisterButton = wx.Button(parent=self, pos=(50, 120), size=(80, 50), label='人脸录入')

        self.PunchcardButton = wx.Button(parent=self, pos=(50, 220), size=(80, 50), label='刷脸签到')

        self.LogcatButton = wx.Button(parent=self, pos=(50, 320), size=(80, 50), label='日志导出')
        
        self.RestartButton = wx.Button(parent=self, pos=(50, 420), size=(80, 50), label='重置系统')

        self.InstructButton =  wx.Button(parent=self,pos=(210,460),size=(80,50),label='操作说明')

        self.ForkButton =  wx.Button(parent=self,pos=(385,460),size=(80,50),label='Frok me')

        self.AboutButton =  wx.Button(parent=self,pos=(560,460),size=(80,50),label='完成人员')

        self.Bind(wx.EVT_BUTTON,self.OnRegisterButtonClicked,self.RegisterButton)
        self.Bind(wx.EVT_BUTTON,self.OnPunchCardButtonClicked,self.PunchcardButton)
        self.Bind(wx.EVT_BUTTON,self.OnLogcatButtonClicked,self.LogcatButton)
        self.Bind(wx.EVT_BUTTON,self.OnInstructButtonClicked,self.InstructButton)
        self.Bind(wx.EVT_BUTTON,self.OnForkButtonClicked,self.ForkButton)
        self.Bind(wx.EVT_BUTTON,self.OnAboutButtonClicked,self.AboutButton)
        


        self.Bind(wx.EVT_BUTTON,self.OnRestartButtonClicked,self.RestartButton)
        # 封面图片
        self.image_cover = wx.Image(main, wx.BITMAP_TYPE_ANY).Scale(520, 360)
        # 显示图片
        self.bmp = wx.StaticBitmap(parent=self, pos=(180,80), bitmap=wx.Bitmap(self.image_cover))

##    def Onclose(frame,event):
##        r = wx.MessageBow('Close?','True',wx.CANCEL|wx.OK|wx.ICON_QUESTION)
##        if r == wx.ID_OK:
##            frame.Destroy()
    def OnRegisterButtonClicked(self,event):
        reload(save_photo_new)
        #del sys.modules['save_photo_new']
        #import save_photo_new
        #runpy.run_path("face_img_register.py")
        #frame = save_photo_new.RegisterUi(None)
        app = wx.App()

        frame = save_photo_new.RegisterUi(None)
##        panel = wx.Panel(frame)
##        button = wx.Button(panel,label = "close")
##        button.Bind(wx.EVT_BUTTON,self.Onclose,frame)
##        frame.Bind(wx.Evt_CLOSE,Onclose)
        frame.Show()
        app.MainLoop()









    def OnPunchCardButtonClicked(self,event):
        #del sys.modules['face_recognize_punchcard']
        reload(recongized)
        #import face_recognize_punchcard
        app = wx.App()

        frame = recongized.PunchcardUi(None)
##        panel = wx.Panel(frame)
##        button = wx.Button(panel,label = "close")
##        button.Bind(wx.EVT_BUTTON,self.Onclose,frame)
##        frame.Bind(wx.Evt_CLOSE,Onclose)
        frame.Show()
        app.MainLoop()

    def OnLogcatButtonClicked(self,event):
        if os.path.exists(file_path):
            #调用系统默认程序打开文件
            os.startfile(file_path)
        else:
            wx.MessageBox(message="要先运行过一次刷脸签到系统，才有日志", caption="警告")
        pass

    def OnRestartButtonClicked(self,event):

        
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("/home/pi/test/each_code/Cascades/lbpcascade_frontalface.xml")

        if os.path.exists(file_path):
            os.remove(file_path)
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        for imagePath in imagePaths:
            os.remove(imagePath)
        imagePaths = [os.path.join(path_restart,f) for f in os.listdir(path_restart)]     
        faceSamples=[]
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        recognizer.train(faceSamples, np.array(ids))
        recognizer.write('/home/pi/test/trainer/trainer.yml')

        with MyDB('localhost','root','123456','student') as cs:
            
            sql_check_log = "TRUNCATE TABLE CHECK_LOG;"
            sql_user = "TRUNCATE TABLE user;"
            sql_qingjia = "TRUNCATE TABLE qingjia;"
            try:
                cs.execute(sql_check_log)
                cs.execute(sql_user)
                cs.execute(sql_qingjia)
            except:
                cs.rollback()
        
        wx.MessageBox(message="已重置系统", caption="警告")









    def OnForkButtonClicked(self,event):
        webbrowser.open("https://https://github.com/T00youngT00simple",new=1,autoraise=True)

    def OnInstructButtonClicked(self,event):
        wx.MessageBox(message="打开系统进行人脸录入，在完成签到后才会有日志生成，可对日志进行查看，导出为csv文件",caption="操作说明")
        pass

    def OnAboutButtonClicked(self,event):
        wx.MessageBox(message="技术支持:李洪浩     专业班级:物联B151"+
                              "\n联系qq:940512354     所在单位:华北科技学院", caption="关于我")

class MainApp(wx.App):
    def OnInit(self):
        self.frame = Mainui(None)
        self.frame.Show()
        return True

app = MainApp()
app.MainLoop()




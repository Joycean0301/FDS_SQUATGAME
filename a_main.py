import cv2
import mediapipe as mp
import numpy as np
import datetime
import time
from PyQt5 import QtCore, QtWidgets
import sys
import csv
from csv import writer
from PyQt5.QtWidgets import QApplication,QMainWindow
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from gui import Ui_MainWindow
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


NAME = ''
COUNTER = 0 
stage = 'up'
table_rank = {}
sorted_table_rank = {}
stage = 'up'

class MainWindow:
    def __init__(self):
        self.main_win = QMainWindow()
        self.uic = Ui_MainWindow()
        self.thread = {}
        self.uic.setupUi(self.main_win)
        self.uic.start.clicked.connect(self.clear)
        #self.uic.start.clicked.connect(self.start_worker_1)
        self.uic.enter.clicked.connect(self.name)
        self.uic.submmit.clicked.connect(self.score)
        self.loaddata()
        self.start_video()
        self.start_worker_1()
    
    # Set variable
    def clear(self):
        global COUNTER,stage
        COUNTER = 0 
        stage = 'up'
        self.uic.Name.setText('')
        self.uic.enter.setEnabled(True)
        self.uic.submmit.setEnabled(False)


    def name(self):
        global NAME
        NAME = self.uic.Name_input.text()
        self.uic.Name_input.setText('')
        self.uic.Name.setText(NAME)
        self.uic.enter.setEnabled(False)
        self.uic.submmit.setEnabled(True)
        
        print('NAME:',self.uic.Name.text())
        print('ENTER SUCCESS!!!')

    def score(self):
        global COUNTER,sorted_table_rank,table_rank
        COUNTER = 0
        self.uic.enter.setEnabled(True)
        self.uic.submmit.setEnabled(False)
        
        
        name = self.uic.Name.text()
        score = int(self.uic.Score.text())
        hour_time = time.strftime("%H:%M:%S", time.localtime())
        current_time = datetime.datetime.now()
        date_time = f'{current_time.day}/{current_time.month}/{current_time.year}'
        
        # update name and score to dic
        #table_rank.update({name:score})
        # sorted dict
        # temp = sorted(table_rank.items(), key=lambda x:x[1],reverse=True)
        # final = dict(temp)
        # sorted_table_rank = final
        
        List = [name, score, hour_time, date_time]
        table_rank = {}
        sorted_table_rank = {}
        with open('precord.csv', 'a+') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(List)
            f_object.close()

        print('-'*20)
        print(f'Name:{name} | Score:{score} | Date:{date_time} | Time:{hour_time} ADDED')
        print('-'*20)
        self.uic.Name.setText('Submit success!!!')
        self.loaddata()

    def loaddata(self):
        global sorted_table_rank,table_rank
        
        with open('precord.csv') as csv_file:
            csv_reader = csv.reader(csv_file) 
            for row in csv_reader:
                if len(row) != 0:
                    table_rank.update({row[0]:int(row[1])})
                # sorted dict
                    temp = sorted(table_rank.items(), key=lambda x:x[1],reverse=True)
                    final = dict(temp)
                    sorted_table_rank = final
            csv_file.close()


        row = 0
        key = list(sorted_table_rank.keys())
        value = list(sorted_table_rank.values())
        
        self.uic.tableWidget.setRowCount(len(sorted_table_rank))
        for i in range(len(sorted_table_rank)):
            self.uic.tableWidget.setItem(row,0,QtWidgets.QTableWidgetItem(str(key[i])))
            self.uic.tableWidget.setItem(row,1,QtWidgets.QTableWidgetItem(str(value[i])))
            if i == 0:
                self.uic.tableWidget.setItem(row,2,QtWidgets.QTableWidgetItem(str('🥇')))
            elif i == 1:
                self.uic.tableWidget.setItem(row,2,QtWidgets.QTableWidgetItem(str('🥈')))
            elif i == 2:
                self.uic.tableWidget.setItem(row,2,QtWidgets.QTableWidgetItem(str('🥉')))    
            row=row+1


    # Video from Open CV to Pyqt5
    def start_video(self):
        #self.uic.start.setEnabled(False)
        self.Work = Work()
        self.Work.start()
        self.Work.Imageupd.connect(self.Imageupd_slot)

    def Imageupd_slot(self, Image):
        self.uic.screen.setPixmap(QPixmap.fromImage(Image))

    # Update Score Realtime
    def start_worker_1(self):
        self.thread[1] = ThreadClass(index=1)
        self.thread[1].start()
        self.thread[1].signal.connect(self.my_function)

    def my_function(self, counter):
        m = counter
        self.uic.Score.setText(str(m))

    # SHOW
    def show(self):
        self.main_win.show()

# Class Update Score realtime
class ThreadClass(QtCore.QThread):
    signal = pyqtSignal(int)

    def __init__(self, index=0):
        super().__init__()
        self.index = index

    def run(self):
        global COUNTER
        while True:
            time.sleep(1)
            self.signal.emit(COUNTER)


# Class MediaPipe + OpenCV detect angle
stage = None
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle >180.0:
        angle = 360-angle
    return angle 

class Work(QThread):
    Imageupd = pyqtSignal(QImage)

    def run(self):
        global COUNTER,stage
        cap = cv2.VideoCapture('SQUAT.mp4')
        #cap = cv2.VideoCapture(0)
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:
                ret, frame = cap.read()
                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates
                    shoulder_L = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    elbow_L = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    wrist_L = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    shoulder_R = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    wrist_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    # Calculate angle
                    angle_L = calculate_angle(shoulder_L, elbow_L, wrist_L)
                    angle_R = calculate_angle(shoulder_R, elbow_R, wrist_R)

                    #PUT TEXT
                    cv2.putText(image, str(angle_L),tuple(np.multiply(elbow_L, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(angle_R),tuple(np.multiply(elbow_R, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                    # Rep data
                    cv2.putText(image, 'REPS', (15,12),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(COUNTER), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                    # Stage data
                    cv2.putText(image, 'STAGE', (65,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, stage, (60,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                    # Drawing landmark
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) )  

                    if (angle_L <= 90)and(angle_R <= 90):
                        stage = "down"
                    if ((angle_L > 165)and(angle_L > 165)) and stage =='down':
                        stage="up"
                        COUNTER +=1
                        #print(COUNTER)           
                except:
                    pass             

                convertir_QT = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                pic = convertir_QT.scaled(1000, 1000, Qt.KeepAspectRatio)
                self.Imageupd.emit(pic)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
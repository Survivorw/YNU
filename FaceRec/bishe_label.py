import cv2
import sys
from tkinter import *
import tkinter as tk
master = Tk()



def First_Scriptcallback():
    exec(open(rb'C:\\Users\\lu152\\opencvproject\\bishe\\get_face.py',encoding='UTF-8').read())


def second_Scriptcallback():
    exec(open(rb'C:\\Users\\lu152\\opencvproject\\bishe\\face_train.py', encoding='UTF-8').read())


def third_Scriptcallback():
    exec(open(rb'C:\\Users\\lu152\\opencvproject\\bishe\\Face_recognition.py', encoding='UTF-8').read())

master.title("人脸识别系统")
canvas = tk.Canvas(master, height=300, width = 400)
canvas.pack()

firstButton = Button(master, text="人脸获取及检测", command=First_Scriptcallback)
firstButton.pack()

secondButton = Button(master, text="训练人脸模板", command=second_Scriptcallback)
secondButton.pack()

thirdButton = Button(master, text="人脸识别", command=third_Scriptcallback)
thirdButton.pack()


mainloop()
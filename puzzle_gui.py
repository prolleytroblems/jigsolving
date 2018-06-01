from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from img_recog_tf import *
from img_recog_proto import *
from time import sleep
import cv2


class GUI(Tk):

    def __init__(self, functions):
        super().__init__()
        self.open(functions)
        self.mainloop()


    def open(self, functions):
        self.title("Image rebuild")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        mainframe=ttk.Frame(self, padding=(5,5,0,5))
        mainframe.columnconfigure(0, weight=1)
        mainframe.rowconfigure(0, weight=1)
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))

        sideframe=ttk.Frame(self)
        sideframe.columnconfigure(0, weight=1)
        sideframe.rowconfigure(0, weight=1)
        sideframe.grid(column=1, row=0, sticky=(N, W, E, S), padx=7)

        detailslabel=ttk.Label(sideframe)
        detailslabel.configure(text="Dimensions:\n\nNumber of pieces:")
        detailslabel.grid(column=0, row=0, sticky=(W,E))

        buttonframe=ttk.Frame(sideframe)
        buttonframe.grid(column=0, row=1, sticky=(N, W, E, S))

        sideframe.columnconfigure(0, weight=1)
        sideframe.rowconfigure(0, weight=1)
        sideframe.rowconfigure(1, weight=1)

        pathvariable=StringVar()
        pathentry=ttk.Entry(buttonframe, textvariable=pathvariable)
        pathentry.configure(state="disabled")
        pathentry.grid(column=0, row=0, pady=10)

        shufflebutton=ttk.Button(buttonframe, text="Shuffle", default="active")
        shufflebutton.grid(column=0, row=1, pady=10)
        #shufflebutton.configure(command=lambda x: functions[0](x))
        shufflebutton.configure(command=lambda: print(2))

        solvebutton=ttk.Button(buttonframe, text="Solve")
        solvebutton.configure(command=lambda: self.plot_image(cv2.imread('puzzle.jpg', 1)))
        solvebutton.grid(column=0, row=2, pady=10)


        buttonframe.columnconfigure(0, weight=1)

        self.canvas = Canvas(mainframe)
        self.canvas.configure(height=600, width=800)
        self.canvas.grid(column=0, row=0, sticky=(N, W, E, S))

    def plot_image(self, images, dims=(1,1)):
        print(images.shape)
        "plots an image or equally sized pieces of an image into a canvas object"
        self.canvas.create_image(100, 100, image=ImageTk.PhotoImage(Image.fromarray(images)), anchor=NW)
        if len(np.array(images.shape))==4:
            assert dims[0]*dims[1]==len(images)
        elif len(images.shape)==3:
            images=[images]
        else:
            raise Exception("Invalid image object")
        shape=images[0].shape
        images=[ImageTk.PhotoImage(Image.fromarray(image)) for image in images]

root=GUI([lambda x:x, lambda x:x])


#ttk.Label(mainframe, image=image).grid(column=0, row=1)

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
        shufflebutton.configure(command=lambda x: functions[0](x))
        shufflebutton.configure(command=lambda: print(2))

        solvebutton=ttk.Button(buttonframe, text="Solve")
        solvebutton.configure(command=lambda: self.plot_image(openimg('puzzle.jpg')))
        solvebutton.grid(column=0, row=2, pady=10)

        buttonframe.columnconfigure(0, weight=1)

        self.canvas = Canvas(mainframe)
        self.canvas.configure(height=600, width=800)
        self.canvas.grid(column=0, row=0, sticky=(N, W, E, S))

    def plot_image(self, images, dims=(1,1)):
        "plots an image or equally sized pieces of an image into a canvas object"

        if len(np.array(images.shape))==4:
            assert dims[0]*dims[1]==len(images)
        elif len(images.shape)==3:
            images=[images]
        else:
            raise Exception("Invalid image object")
        print(self.canvas['width'])
        center=(400, 300)
        shape=images[0].shape
        full_size_reversed=np.array((shape[1]*dims[1], shape[0]*dims[0]))

        centers=np.array([(x*shape[1], y*shape[0]) for y in range(dims[0]) for x in range(dims[1])])
        centers+=center-full_size_reversed//2+(shape[1]//2,shape[0]//2)
        self.canvas.images=[ImageTk.PhotoImage(Image.fromarray(image)) for image in images]

        for image, piece_center in zip(self.canvas.images, centers):
            id=self.canvas.create_image(piece_center[0], piece_center[1], image=image)

root=GUI([lambda x:x, lambda x:x])


#ttk.Label(mainframe, image=image).grid(column=0, row=1)

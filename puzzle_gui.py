from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from img_recog_tf import *
from time import sleep



class GUI(Tk):

    def __init__(self, functions):
        super().__init__()
        self.open(functions)

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

        shufflebutton=ttk.Button(buttonframe, text="Shuffle")
        shufflebutton.grid(column=0, row=1, pady=10)
        shufflebutton.bind(lambda x: functions[0](x))

        solvebutton=ttk.Button(buttonframe, text="Solve")
        solvebutton.grid(column=0, row=2, pady=10)
        solvebutton.bind(lambda x: function[1](x))

        buttonframe.columnconfigure(0, weight=1)


        image_pieces=img_split_cpu('puzzle.jpg', dims)
        shape=image_pieces[0].shape
        full_shape=(image_pieces[0].shape[0]*dims[0], image_pieces[0].shape[1]*dims[1])

        self.canvas = Canvas(mainframe)
        self.canvas.configure(height=full_shape[0], width=full_shape[1])
        self.canvas.grid(column=0, row=0, sticky=(N, W, E, S))






dims=(3,3)
def plot(event, image_pieces, shape):
    images=[ImageTk.PhotoImage(Image.fromarray(image)) for image in image_pieces]
    for i, image in enumerate(images):
        print(i%3, i//3)
        canvas.create_image(i%3*shape[1], i//3*shape[0], image=image, anchor=NW)
    #canvas.create_line((0, 0, 100, 100))
    #canvas.create_image(j*shape[1], i*shape[0], image=image, anchor=NW)

canvas

root.mainloop()


#ttk.Label(mainframe, image=image).grid(column=0, row=1)

from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from time import sleep
import cv2
import numpy as np
import re


class GUI(Tk):
    """A simple gui for prototyping"""

    def __init__(self, functions, dims=(1,1), **params):
        """Four functions as input, in a dictionary."""
        super().__init__()
        self.images=None
        self.dims=None
        self.start(self.decorate_functions(functions))
        self.mainloop()

    def start(self, functions):
        self.title("Image rebuild")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        mainframe=ttk.Frame(self, padding=(5,5,0,5))
        mainframe.columnconfigure(0, weight=1)
        mainframe.rowconfigure(0, weight=1)
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        mainframe.configure(borderwidth=2, relief="sunken")

        sideframe=ttk.Frame(self)
        sideframe.columnconfigure(0, weight=1)
        sideframe.rowconfigure(0, weight=1)
        sideframe.grid(column=1, row=0, sticky=(N, W, E, S), padx=1, pady=1)

        #----------------------------------

        openframe=ttk.Frame(sideframe, borderwidth=2, relief="groove")
        openframe.columnconfigure(0, weight=1)
        openframe.columnconfigure(1, weight=6)
        openframe.rowconfigure(0, weight=1)
        openframe.rowconfigure(1, weight=1)
        openframe.rowconfigure(2, weight=1)
        openframe.grid(column=0, row=0, sticky=(N, W, E, S), padx=2, pady=2)

        shuffleframe=ttk.Frame(sideframe, borderwidth=2, relief="groove")
        shuffleframe.columnconfigure(0, weight=1)
        shuffleframe.columnconfigure(1, weight=1)
        shuffleframe.rowconfigure(0, weight=1)
        shuffleframe.rowconfigure(1, weight=1)
        shuffleframe.grid(column=0, row=1, sticky=(N, W, E, S), padx=2, pady=2)

        distortframe=ttk.Frame(sideframe, borderwidth=2, relief="groove")
        distortframe.columnconfigure(0, weight=1)
        distortframe.rowconfigure(0, weight=1)
        distortframe.rowconfigure(1, weight=1)
        distortframe.rowconfigure(2, weight=1)
        distortframe.grid(column=0, row=2, sticky=(N, W, E, S), padx=2, pady=2)

        solveframe=ttk.Frame(sideframe, borderwidth=2, relief="groove")
        solveframe.columnconfigure(0, weight=1)
        solveframe.columnconfigure(1, weight=2)
        solveframe.rowconfigure(0, weight=1)
        solveframe.rowconfigure(1, weight=1)
        solveframe.grid(column=0, row=3, sticky=(N, W, E, S), padx=2, pady=2)

        fillerframe=ttk.Frame(sideframe)
        fillerframe.columnconfigure(0, weight=1)
        fillerframe.rowconfigure(0, weight=1)
        fillerframe.grid(column=0, row=4, sticky=(N, W, E, S), padx=1, pady=1)

        self.progress=ttk.Progressbar(sideframe, orient=HORIZONTAL, length=30, mode="determinate")
        self.progress.grid(column=0, row=5, sticky=(E,W), padx=3)

        sideframe.columnconfigure(0, weight=1)
        sideframe.rowconfigure(0, weight=5)
        sideframe.rowconfigure(1, weight=5)
        sideframe.rowconfigure(2, weight=5)
        sideframe.rowconfigure(3, weight=5)
        sideframe.rowconfigure(4, weight=15)
        sideframe.rowconfigure(5, weight=1)

        #-----------------------------

        pathlabel=ttk.Label(openframe)
        pathlabel.configure(text="Path:")
        pathlabel.grid(column=0, row=0, sticky=W, pady=2, padx=3)

        self.pathentry=ttk.Entry(openframe)
        self.pathentry.grid(column=1, row=0, pady=2, padx=5, sticky=(W,E))
        self.pathentry.insert(0,"puzzle.jpg")

        openbutton=ttk.Button(openframe, text="Open", width=20)
        openbutton.configure(command=lambda: functions["open"])
        openbutton.grid(column=0, row=1, columnspan=2, padx=3)

        self.detailslabel=ttk.Label(openframe)
        self.detailslabel.configure(text="Size:\nName:\nFormat:")
        self.detailslabel.grid(column=0, row=2, columnspan=2, padx=3, sticky=(N,W,E))

        #-------------------------

        xframe=ttk.Frame(shuffleframe)
        xframe.columnconfigure(0, weight=1)
        xframe.columnconfigure(1, weight=1)
        xframe.rowconfigure(0, weight=1)
        xframe.grid(column=0, row=0, sticky=(N, W, E, S))

        xlabel=ttk.Label(xframe)
        xlabel.configure(text="X splits:")
        xlabel.grid(column=0, row=0, pady=2, padx=3, sticky=E)

        xvar=StringVar()
        xspin=Spinbox(xframe, from_=1, to=10, width=4, textvariable=xvar)
        xspin.grid(column=1, row=0, pady=2, padx=3)
        xvar.set("4")

        yframe=ttk.Frame(shuffleframe)
        yframe.columnconfigure(0, weight=1)
        yframe.columnconfigure(1, weight=1)
        yframe.rowconfigure(0, weight=1)
        yframe.grid(column=1, row=0, sticky=(N, W, E, S))

        ylabel=ttk.Label(yframe)
        ylabel.configure(text="Y splits:")
        ylabel.grid(column=2, row=0, pady=2, padx=3, sticky=E)

        yvar=StringVar()
        yspin=Spinbox(yframe, from_=1, to=10, width=4, textvariable=yvar)
        yspin.grid(column=3, row=0, pady=2, padx=3)
        yvar.set("4")

        shufflebutton=ttk.Button(shuffleframe, text="Shuffle", default="active", width=20)
        shufflebutton.grid(column=0, row=1, columnspan=4, pady=2)
        shufflebutton.configure(command=lambda: functions["shuffle"](dims=(int(xvar.get()), int(yvar.get()))))

        #-----------------------

        deltalabel=ttk.Label(distortframe)
        deltalabel.configure(text="Delta:")
        deltalabel.grid(column=0, row=0, sticky=W, pady=2, padx=3)

        deltaentry=ttk.Entry(distortframe)
        deltaentry.grid(column=1, row=0, pady=2, padx=5, sticky=(W,E))
        deltaentry.insert(0,"10")

        self.disttypevar = StringVar()
        distortcombo = ttk.Combobox(distortframe, textvariable=self.disttypevar)
        distortcombo.configure(values=["Brightness", "Color", "Gradient"], state="readonly")
        distortcombo.grid(column=0, row=1, pady=2, padx=5, columnspan=2, sticky=(W,E))
        self.disttypevar.set("Brightness")

        distortbutton=ttk.Button(distortframe, text="Distort", width=20)
        distortbutton.configure(command=lambda: functions["distort"](delta=float(deltaentry.get()), self.disttypevar.get()))
        distortbutton.grid(column=0, row=2, pady=2, columnspan=2)

        #-------------------------

        poollabel=ttk.Label(solveframe)
        poollabel.configure(text="Pooling:")
        poollabel.grid(column=0, row=0, sticky=W, pady=2, padx=3)

        poolvar=StringVar()
        poolspin=Spinbox(solveframe, from_=1, to=10, textvariable=poolvar, width=10)
        poolspin.grid(column=1, row=0, pady=2, padx=5, sticky=(W))
        poolvar.set("5")

        solvebutton=ttk.Button(solveframe, text="Solve", width=20)
        solvebutton.configure(command=lambda: functions["solve"]())
        solvebutton.grid(column=0, row=1, columnspan=2, pady=2)

        #--------------------------

        self.canvas=Canvas(mainframe)
        self.canvas.configure(height=600, width=800)
        self.canvas.grid(column=0, row=0, sticky=(N, W, E, S))

    def plot_image(self, images, dims=(1,1)):
        "plots an image or equally sized pieces of an image into a canvas object"
        if len(np.array(images).shape)==4:
            assert dims[0]*dims[1]==len(images)
        elif len(np.array(images).shape)==3:
            assert dims[0]*dims[1]==1
            images=[images]
        else:
            raise Exception("Invalid image object")
        center=(400, 300)

        images, ratio=resize_for_canvas(images, (640, 480))
        shape=images[0].shape
        full_size_reversed=np.array((shape[1]*dims[1], shape[0]*dims[0]))

        centers=np.array([(x*shape[1], y*shape[0]) for y in range(dims[0]) for x in range(dims[1])])
        centers+=center-full_size_reversed//2+(shape[1]//2,shape[0]//2)

        self.images=images
        self.dims=dims
        self.canvas.tkimages=[ImageTk.PhotoImage(Image.fromarray(image)) for image in images]
        for image, piece_center in zip(self.canvas.tkimages, centers):
            id=self.canvas.create_image(piece_center[0], piece_center[1], image=image)

    @static
    def resize_for_canvas(images, size):
        if isinstance(images, np.ndarray):
            shape=images.shape
            if shape[1]/shape[0]=>1:
                ratio=640/dims[1]/shape[1]
                new_shape=(int(ratio*shape[1]), int(ratio*shape[0]))
            elif shape[1]/shape[0]<1:
                ratio=480//dims[0]/shape[0]
                new_shape=(int(ratio*shape[1]), int(ratio*shape[0]))
            return (cv2.resize(images, size), ratio)
        elif isinstance(images, list):
            shape=images[0].shape
            if shape[1]/shape[0]=>1:
                ratio=640/dims[1]/shape[1]
                new_shape=(int(ratio*shape[1]), int(ratio*shape[0]))
            elif shape[1]/shape[0]<1:
                ratio=480//dims[0]/shape[0]
                new_shape=(int(ratio*shape[1]), int(ratio*shape[0]))
            resized=[]
            for image in images
                resized.append(cv2.resize(images, size))
            return (resized, ratio)
        else:
            raise TypeError("Images must be an ndarray or list of ndarrays")

    def decorate_functions(self, functions):
        def open_image(self, path):
            image=functiond["open"](path)
            self.plot_image(image, dims=(1,1))
            self.detailslabel.configure(text="Size: " + str(image.shape[0])+"x" +
                                            str(image.shape[1]) + "\nName: " +
                                            re.spit("\\")[-1] + "\nFormat: "+re.spit(r".")[-1])
            self.image_path=path

        def shuffle_image(self, shuffle_function, dims):
            image=functions["shuffle"](self.images, dims=dims)
            self.plot_image(image, dims=dims)

        def distort_image(self, distort_function, delta, mode):
            image=functions["distort"](self.images, delta, mode)
            self.plot_image(image, dims=self.dims)

        def solve_puzzle(self, solve_function):
            image=functions["solve"](self.image_path, self.images, dims=self.dims,
                                    pooling=int(poolvar.get()))
            self.plot_image(image, dims=self.dims)

        new_functions={"open": open_image, "shuffle": shuffle_image, "distort": distort_image, "solve": solve_puzzle}

        return new_functions


@cuda.jit("(uint8[:,:,:], int32[:,:], )")
def max_pool_unit(image, pooling, stride, pooled_image):
    y,x=cuda.grid(2)

    if y>pooled_image.shape[0] or x>pooled_image.shape[1]:
        return
    window=image[y*stride[1]:y*stride[1]+pooling[1], x*stride[0]:x*stride[0]+pooling[0], :]
    FLATTEN WINDOW AND TAKE MAX
    pooled_image[y,x]=max()


def pool(image, pooling, stride):
    assert pooling>=stride
    def add_padding(image, axis, side="end"):
        if (image.shape[axis]-pooling+stride)%stride[axis]==0:
            return image
        else:
            #add a blank row/column
            IMPLEMENT THIS
            if side=="end":
                image=add_padding(image, axis, side="start")
            if side=="start":
                image=add_padding(image, axis, side="end")
            return image

    for axis in range(2):
        image=add_padding(image, axis, "end")






def main():
    window=GUI({"solve":lambda x:x, "shuffle":lambda x:x, "open":lambda x:x})

if __name__=="__main__":
    main()

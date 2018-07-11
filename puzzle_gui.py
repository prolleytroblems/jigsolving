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
        if not("decorate" in params):
            params["decorate"]=True
        super().__init__()
        self.images=None
        self.dims=None
        if params["decorate"]==True:
            self.start(self.decorate_functions(functions))
        else:
            self.start(functions)
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
        self.pathentry.insert(0,"images\puzzle.jpg")

        self.openbutton=ttk.Button(openframe, text="Open", width=20)
        self.openbutton.configure(command=lambda: functions["open"](self.pathentry.get()))
        self.openbutton.grid(column=0, row=1, columnspan=2, padx=3)

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

        self.shufflebutton=ttk.Button(shuffleframe, text="Shuffle", default="active", width=20)
        self.shufflebutton.grid(column=0, row=1, columnspan=4, pady=2)
        self.shufflebutton.configure(command=lambda: functions["shuffle"](dims=(int(yvar.get()), int(xvar.get()))))

        #-----------------------

        deltalabel=ttk.Label(distortframe)
        deltalabel.configure(text="Delta:")
        deltalabel.grid(column=0, row=0, sticky=W, pady=2, padx=3)

        deltaentry=ttk.Entry(distortframe)
        deltaentry.grid(column=1, row=0, pady=2, padx=5, sticky=(W,E))
        deltaentry.insert(0,"10")

        self.disttypevar = StringVar()
        distortcombo = ttk.Combobox(distortframe, textvariable=self.disttypevar)
        distortcombo.configure(values=["Noise", "Brightness", "Color", "Gradient", "Shape"], state="readonly")
        distortcombo.grid(column=0, row=1, pady=2, padx=5, columnspan=2, sticky=(W,E))
        self.disttypevar.set("Noise")

        self.distortbutton=ttk.Button(distortframe, text="Distort", width=20)
        self.distortbutton.configure(command=lambda: functions["distort"](delta=float(deltaentry.get()), mode=self.disttypevar.get()))
        self.distortbutton.grid(column=0, row=2, pady=2, columnspan=2)

        #-------------------------

        poollabel=ttk.Label(solveframe)
        poollabel.configure(text="Pooling:")
        poollabel.grid(column=0, row=0, sticky=W, pady=2, padx=3)

        poolvar=StringVar()
        poolspin=Spinbox(solveframe, from_=1, to=10, textvariable=poolvar, width=10)
        poolspin.grid(column=1, row=0, pady=2, padx=5, sticky=(W))
        poolvar.set("5")

        self.solvebutton=ttk.Button(solveframe, text="Solve", width=20)
        self.solvebutton.configure(command=lambda: functions["solve"](pooling=int(poolvar.get())))
        self.solvebutton.grid(column=0, row=1, columnspan=2, pady=2)

        #--------------------------

        self.canvas=Canvas(mainframe)
        self.canvas.configure(height=600, width=800)
        self.canvas.grid(column=0, row=0, sticky=(N, W, E, S))

        #--------------------------


        self.shufflebutton.configure(state="disabled")
        self.distortbutton.configure(state="disabled")
        self.solvebutton.configure(state="disabled")


    def plot_image(self, images, dims=(1,1), **params):
        "plots an image or equally sized pieces of an image into a canvas object"
        if not("clear" in params):
            params["clear"]=True

        if len(np.array(images).shape)==4:
            assert dims[0]*dims[1]==len(images)
        elif len(np.array(images).shape)==3:
            assert dims[0]*dims[1]==1
            images=[images]
        else:
            raise Exception("Invalid image object")

        if params["clear"]==True:
            self.canvas.delete(self.canvas.find_all())

        center=(400, 300)

        #This should go before resizing!
        self.images=images
        self.dims=dims

        images, ratio=GUI.resize_for_canvas(images, (640, 480), dims)
        shape=images[0].shape
        full_size_reversed=np.array((shape[1]*dims[1], shape[0]*dims[0]))

        centers=np.array([(x*shape[1], y*shape[0]) for y in range(dims[0]) for x in range(dims[1])])
        centers+=center-full_size_reversed//2+(shape[1]//2,shape[0]//2)

        self.canvas.tkimages=[ImageTk.PhotoImage(Image.fromarray(image)) for image in images]

        ids=[]
        for image, piece_center in zip(self.canvas.tkimages, centers):
            ids.append(self.canvas.create_image(piece_center[0], piece_center[1], image=image))

        return ids


    @staticmethod
    def resize_for_canvas(images, size, dims):
        if isinstance(images, np.ndarray):
            shape=images.shape
            if shape[1]/shape[0]>=1:
                ratio=640/dims[1]/shape[1]
                new_shape=(int(ratio*shape[1]), int(ratio*shape[0]))
            elif shape[1]/shape[0]<1:
                ratio=480//dims[0]/shape[0]
                new_shape=(int(ratio*shape[1]), int(ratio*shape[0]))
            return (cv2.resize(images, new_shape), ratio)

        elif isinstance(images, list):
            shape=images[0].shape
            if shape[1]/shape[0]>=1:
                ratio=640/dims[1]/shape[1]
                new_shape=(int(ratio*shape[1]), int(ratio*shape[0]))
            elif shape[1]/shape[0]<1:
                ratio=480//dims[0]/shape[0]
                new_shape=(int(ratio*shape[1]), int(ratio*shape[0]))
            resized=[]
            for image in images:
                resized.append(cv2.resize(image, new_shape))
            return (resized, ratio)
        else:
            raise TypeError("Images must be an ndarray or list of ndarrays")


<<<<<<< HEAD
    def move_image(self, id, delx, dely, time=None, r_rate=10):
        def diff_move(id, x0, y0, dx, dy, step, end):
            px=round(x0+dx-round(x0))
            py=round(y0+dy-round(y0))
            print(px, py)
            if abs(px)>0 or abs(py)>0:
                self.canvas.move(id, px, py)
            if step<end:
                self.canvas.after(r_rate, lambda : diff_move(id, x0+dx, y0+dy, dx, dy, step+r_rate, end))
=======
    def move_image(self, id, delx, dely, time=None):
        def diff_move(id, x0, y0, dx, dy, step, end):

            px=int(x0+dx-int(x0))
            py=int(y0+dy-int(y0))
            if px>0 or py>0:
                self.canvas.move(id, px, py)
            if step<end:
                self.canvas.after(1, lambda : diff_move(id, x0+dx, y0+dy, dx, dy, step+1, end))
>>>>>>> 81b28be04fa8f8858b6dd1983b5fa521ceaaba17

        if time==None:
            self.canvas.move(id, dx, dy)
            return

        elif isinstance(time, int):
<<<<<<< HEAD
            dx=delx/time*r_rate
            dy=dely/time*r_rate
            print(self.canvas.coords(id), dx, dy)
            x0, y0 = self.canvas.coords(id)
            diff_move(id, x0, y0, dx, dy, 0, time)
=======
            dx=delx/time
            dy=dely/time
            print(self.canvas.coords(id))
            x0, y0 = self.canvas.coords(id)
            diff_move(id,x0, y0, dx, dy, 0, time )

>>>>>>> 81b28be04fa8f8858b6dd1983b5fa521ceaaba17
        else:
            raise TypeError("time must be a positive integer")


<<<<<<< HEAD
=======





>>>>>>> 81b28be04fa8f8858b6dd1983b5fa521ceaaba17
    def decorate_functions(self, functions):
        def open_image(path):
            image=functions["open"](path)
            ids=self.plot_image(image, dims=(1,1))
            self.detailslabel.configure(text="Size: " + str(image.shape[0])+" x " +
                                            str(image.shape[1]) + " pixels \nName: " +
                                            re.split(r"\\", path)[-1] + "\nFormat: "+re.split(r"\.", path)[-1])
            self.image_path=path

            self.shufflebutton.configure(state="enabled")
            self.distortbutton.configure(state="enabled")

            return ids


        def shuffle_image(dims):
<<<<<<< HEAD
            image=functions["shuffle"](self.images, dims=dims)

            ids=self.plot_image(image, dims)
=======
            #image=functions["shuffle"](self.images, dims=dims)

            id=self.plot_image(self.images[0])[0]
            self.move_image(id, 30, 50, 2000)
>>>>>>> 81b28be04fa8f8858b6dd1983b5fa521ceaaba17

            self.distortbutton.configure(state="enabled")
            self.shufflebutton.configure(state="disabled")
            self.solvebutton.configure(state="enabled")

        def distort_image(delta, mode):
            modedict={"Noise":"n", "Brightness":"b", "Color":"c", "Gradient":"g", "Shape":"s"}
            image=functions["distort"](self.images, delta, modedict[mode])
            self.plot_image(image, dims=self.dims)

        def solve_puzzle(pooling=None):
            image=functions["solve"](self.image_path, self.images, dims=self.dims,
                                    pooling=pooling)
            self.plot_image(image)

            self.shufflebutton.configure(state="disabled")
            self.solvebutton.configure(state="disabled")
            self.shufflebutton.configure(state="enabled")

        new_functions={"open": open_image, "shuffle": shuffle_image, "distort": distort_image, "solve": solve_puzzle}

        return new_functions


def main():
    window=GUI({"solve":lambda x:x, "shuffle":lambda x:x, "open":lambda x:x})

if __name__=="__main__":
    main()

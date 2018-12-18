from tkinter import *
from tkinter import ttk
from puzzle_canvas import PuzzleCanvas
from time import sleep
import numpy as np
import re
from datetime import datetime
from image_obj import *
from utils import *
import cv2

class GUI(Tk):
    """A simple gui for prototyping"""

    def __init__(self, functions, dims=(1,1), **params):
        """Four functions as input, in a dictionary."""
        if not("decorate" in params):
            params["decorate"]=True
        super().__init__()
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

        detectframe=ttk.Frame(sideframe, borderwidth=2, relief="groove")
        detectframe.columnconfigure(0, weight=1)
        detectframe.rowconfigure(0, weight=1)
        detectframe.grid(column=0, row=1, sticky=(N, W, E, S), padx=2, pady=2)

        solveframe=ttk.Frame(sideframe, borderwidth=2, relief="groove")
        solveframe.columnconfigure(0, weight=1)
        solveframe.columnconfigure(1, weight=2)
        solveframe.rowconfigure(0, weight=1)
        solveframe.rowconfigure(1, weight=1)
        solveframe.rowconfigure(2, weight=1)
        solveframe.grid(column=0, row=2, sticky=(N, W, E, S), padx=2, pady=2)

        fillerframe=ttk.Frame(sideframe)
        fillerframe.columnconfigure(0, weight=1)
        fillerframe.rowconfigure(0, weight=1)
        fillerframe.grid(column=0, row=3, sticky=(N, W, E, S), padx=1, pady=1)

        self.progress=ttk.Progressbar(sideframe, orient=HORIZONTAL, length=30, mode="determinate")
        self.progress.grid(column=0, row=4, sticky=(E,W), padx=3)

        sideframe.columnconfigure(0, weight=1)
        sideframe.rowconfigure(0, weight=5)
        sideframe.rowconfigure(1, weight=5)
        sideframe.rowconfigure(2, weight=5)
        sideframe.rowconfigure(3, weight=20)
        sideframe.rowconfigure(4, weight=1)

        #-----------------------------

        pathlabel=ttk.Label(openframe)
        pathlabel.configure(text="Img. path:")
        pathlabel.grid(column=0, row=0, sticky=W, pady=2, padx=3)

        self.pathentry=ttk.Entry(openframe)
        self.pathentry.grid(column=1, row=0, pady=2, padx=5, sticky=(W,E))
        self.pathentry.insert(0,"images/samples/puzzle.jpg")

        solpathlabel=ttk.Label(openframe)
        solpathlabel.configure(text="Sol. path:")
        solpathlabel.grid(column=0, row=1, sticky=W, pady=2, padx=3)

        self.solpathentry=ttk.Entry(openframe)
        self.solpathentry.grid(column=1, row=1, pady=2, padx=5, sticky=(W,E))
        self.solpathentry.insert(0,"images/puzzle.jpg")

        self.openbutton=ttk.Button(openframe, text="Open", width=20)
        self.openbutton.configure(command=lambda: functions["open"](self.pathentry.get(), self.solpathentry.get()))
        self.openbutton.grid(column=0, row=2, columnspan=2, padx=3)

        self.detailslabel=ttk.Label(openframe)
        self.detailslabel.configure(text="Size:\nName:\nFormat:")
        self.detailslabel.grid(column=0, row=3, columnspan=2, padx=3, sticky=(N,W,E))

        #-------------------------

        self.detectbutton=ttk.Button(detectframe, text="Detect", default="active", width=20)
        self.detectbutton.grid(column=0, row=0, columnspan=4, pady=2)
        self.detectbutton.configure(command=lambda: functions["detect"]())

        #-----------------------


        poollabel=ttk.Label(solveframe)
        poollabel.configure(text="Pooling:")
        poollabel.grid(column=0, row=0, sticky=W, pady=2, padx=3)

        poolvar=StringVar()
        poolspin=Spinbox(solveframe, from_=1, to=10, textvariable=poolvar, width=10)
        poolspin.grid(column=1, row=0, pady=2, padx=5, sticky=(W))
        poolvar.set("5")

        comparelabel=ttk.Label(solveframe)
        comparelabel.configure(text="Method:")
        comparelabel.grid(column=0, row=1, sticky=W, pady=2, padx=3)

        self.comparevar=StringVar()
        comparecombo=ttk.Combobox(solveframe, textvariable=self.comparevar, width=10)
        comparecombo.configure(values=["xcorr", "square error"], state="readonly")
        comparecombo.grid(column=1, row=1, pady=2, padx=5, sticky=(W))
        self.comparevar.set("xcorr")

        self.solvebutton=ttk.Button(solveframe, text="Solve", width=20)
        self.solvebutton.configure(command=lambda: functions["solve"](pooling=int(poolvar.get()), method=self.comparevar.get()))
        self.solvebutton.grid(column=0, row=2, columnspan=2, pady=2)

        #--------------------------

        self.canvas=PuzzleCanvas(mainframe, size=(800,600))
        self.canvas.grid(column=0, row=0, sticky=(N, W, E, S))

        #--------------------------

        self.detectbutton.configure(state="disabled")
        self.solvebutton.configure(state="disabled")


    def decorate_functions(self, functions):
        def open_image(image_path, solution_path):
            print("Opening image. image_path: ", image_path)

            image = functions["open"](image_path)
            self.solution_path = solution_path

            collection=PieceCollection(image, (1,1))
            self.scaling=self.canvas.plot_by_order(collection, dims=(1,1), clear=True)
            self.detailslabel.configure(text="Size: " + str(image.shape[0])+" x " +
                                            str(image.shape[1]) + " pixels \nName: " +
                                            re.split(r"\\", image_path)[-1] + "\nFormat: "+re.split(r"\.", image_path)[-1])
            self.detectbutton.configure(state="enabled")
            self.solvebutton.configure(state="disabled")

        def detect_pieces():
            print("Detecting pieces.")

            boxes = functions["detect"](self.canvas.collection)
            piece=self.canvas.collection.get()[0]

            subimages=list(map(lambda box: piece.get_subimage(box), boxes))

            new_collection = PieceCollection(subimages)
            image_center=(piece.array.shape[1]/2, piece.array.shape[0]/2)
            centers=self.canvas.boxes_to_centers(boxes, image_center, self.scaling)
            new_collection.mass_set("location", centers)

            self.canvas.plot_by_location(new_collection, scaling=self.scaling)
            self.canvas.plot_rectangles(boxes, self.scaling)

            self.detectbutton.configure(state="disabled")
            self.solvebutton.configure(state="enabled")

        def solve_puzzle(pooling=None, method="xcorr"):
            print("Solving puzzle. Method: ", method, ". Pooling: ", pooling)

            id_slots = functions["solve"](self.solution_path, self.canvas.collection,
                                    pooling=pooling, iterator_mode=False, method=method)

            self.detectbutton.configure(state="disabled")
            self.solvebutton.configure(state="disabled")

            self.canvas.clear("rectangle")
            self.canvas.update(id_slots)

        new_functions = {"open": open_image, "detect": detect_pieces, "solve": solve_puzzle}

        return new_functions

def main():
    window=GUI({"solve":lambda x:x, "shuffle":lambda x:x, "open":lambda x:x})

if __name__=="__main__":
    main()

from tkinter import Canvas
from image_obj import Piece, PieceCollection
import numpy as np
from math import floor
from PIL import ImageTk, Image
from utils import resize

class PuzzleCanvas(Canvas):

    def __init__(self, master, size, usage=0.8):
        super().__init__(master)
        self.configure( width=size[0], height=size[1] )
        self.usage=(floor(size[0]*usage), floor(size[1]*usage))
        self.ids=None
        self.locations=None
        self.pieces=None
        self.tkimages=None


    def plot_image(self, images, dims=(1,1), **params):
        "plots Piece objects"
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
            self.delete(self.find_all())

        center=(400, 300)

        #This should go before resizing!
        self.pieces=PieceCollection(images, dims)

        images, ratio=resize(images, dims, self.usage)

        centers=self.find_plot_locations(dims, (images[0].shape[0], images[0].shape[1]), center)

        self.locations=np.reshape(centers, (dims[0], dims[1], 2))

        self.tkimages=[ImageTk.PhotoImage(Image.fromarray(image)) for image in images]

        ids=[]
        for image, piece_center in zip(self.tkimages, centers):
            ids.append(self.create_image(piece_center[0], piece_center[1], image=image))
        self.ids=ids
        return ids


    def find_plot_locations(self, dims, piece_shape, center=(400,300), reference="center"):
        if reference=="center":
            full_size=np.array((piece_shape[1]*dims[1], piece_shape[0]*dims[0]))
            centers=np.array([(x*piece_shape[1], y*piece_shape[0]) for y in range(dims[0]) for x in range(dims[1])])
            centers+=center-full_size//2+(piece_shape[1]//2,piece_shape[0]//2)
            return centers

        else: raise NotImplementedError()


    def _diff_move(self, id, x0, y0, dx, dy, step, dt, end):
        rx, ry = self.coords(id)

        px, py = ( round(x0 + dx - rx), round(y0 + dy - ry) )

        if step<end:
            if abs(px)>0 or abs(py)>0:
                self.move(id, px, py)
            self.after(dt, lambda : self._diff_move(id, x0+dx, y0+dy, dx, dy, step+dt, dt, end))
        else:
            pass


    def move_image(self, id, delx, dely, time=None, r_rate=10):
        if time==None:
            self.move(id, dx, dy)
            return

        elif isinstance(time, int):
            dx, dy = ( delx / time * r_rate, dely / time * r_rate )
            x0, y0 = self.coords(id)
            self._diff_move(id, x0, y0, dx, dy, 0, r_rate, time)
        else:
            raise TypeError("time must be a positive integer")


    def move_piece(self, id, target_location):
        target_coords = self.locations[target_location[0], target_location[1]]
        current_coords = self.coords( id )
        delx, dely = ( target_coords[0] - current_coords[0], target_coords[1] - current_coords[1] )
        self.move_image( id, delx, dely, time=500)


    def move_pieces(self, iterator, start):

        self.pieces=PieceCollection([], self.pieces.dims)
        try:
            image=next(iterator)
            self.move_piece(image.id, image.location)
            #print((datetime.now()-start).seconds*1000+(datetime.now()-start).microseconds/1000)
            self.pieces.add(image.array)
            self.after(0, self.move_pieces(iterator, start))
        except StopIteration:
            pass

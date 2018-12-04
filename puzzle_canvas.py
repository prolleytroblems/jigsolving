from tkinter import Canvas
from image_obj import Piece, PieceCollection
import numpy as np
from math import floor
from PIL import ImageTk, Image
from utils import resize
import cv2

class PuzzleCanvas(Canvas):

    def __init__(self, master, size, usage=0.8):
        super().__init__(master)
        self.configure( width=size[0], height=size[1] )
        self.usage=(floor(size[0]*usage), floor(size[1]*usage))
        self.size=size
        self.collection=None


    def configure(self, **params):
        super().configure(**params)
        self.center=[None,None]
        if "height" in params:
            self.center[1]=params["height"]//2
            self.height=params["height"]
        if "width" in params:
            self.center[0]=params["width"]//2
            self.width=params["width"]

    def plot_pieces(self, collection, **params):
        "plots Piece objects from a PieceCollection based on their order (location values are ignored)"
        if not("clear" in params):
            params["clear"]=True

        assert isinstance(collection, PieceCollection)

        if params["clear"]==True:
            self.delete(self.find_all())

        self.resize_collection(collection)
        centers = self.find_plot_locations(collection, (self.size[0]//2, self.size[1]//2))
        tkimages=[self.array_to_image(array) for array in collection.mass_get("plotted")]

        ids=[]

        #plot the pieces
        for piece, piece_center, tkimage in zip(collection.get(), centers, tkimages):
            ids.append(self.create_image(piece_center[0], piece_center[1], image=tkimage))

        slots=[(a,b) for a in range(collection.dims[0]) for b in range(collection.dims[1])]

        collection.mass_set("slot", slots)
        collection.mass_set("location", list(centers))
        collection.mass_set("tkimage", tkimages)
        collection.mass_set("id", ids)

        self.collection=collection


    def array_to_image(self, array):
        return ImageTk.PhotoImage(Image.fromarray(array))


    def find_plot_locations(self, collection, center=(400,300), reference="center"):
        dims=collection.dims
        image=collection.mass_get("plotted")[0]
        piece_shape=(image.shape[0], image.shape[1])

        if reference=="center":
            full_size=np.array((piece_shape[1]*dims[1], piece_shape[0]*dims[0]))
            centers=np.array([(x*piece_shape[1], y*piece_shape[0]) for y in range(dims[0]) for x in range(dims[1])])
            centers+=center-full_size//2+(piece_shape[1]//2,piece_shape[0]//2)
            return centers

        else: raise NotImplementedError()


    def resize_collection(self, collection):
        pieces=collection.get()
        orig_images=[piece.array for piece in pieces]
        new_images, ratio=resize(orig_images, collection.dims, self.usage)

        collection.mass_set("plotted", new_images)

        return collection


    def _diff_move(self, id, x0, y0, dx, dy, step, dt, end):
        rx, ry = self.coords(id)
        px, py = ( round(x0 + dx - rx), round(y0 + dy - ry) )

        if step<end:
            if abs(px)>0 or abs(py)>0:
                self.move(id, px, py)
            self.after(dt, lambda : self._diff_move(id, x0+dx, y0+dy, dx, dy, step+dt, dt, end))
        else:
            pass


    def _move_piece(self, id, delx, dely, time=None, r_rate=10):
        if time==None:
            self.move(id, delx, dely)
            return
        elif isinstance(time, int):
            dx, dy = ( delx / time * r_rate, dely / time * r_rate )
            x0, y0 = self.coords(id)
            self._diff_move(id, x0, y0, dx, dy, 0, r_rate, time)
        else:
            raise TypeError("time must be a positive integer")


    def _move_piece_to_target(self, id, target_coords):
        current_coords = self.coords( id )
        delx, dely = ( target_coords[0] - current_coords[0], target_coords[1] - current_coords[1] )
        self._move_piece( id, delx, dely, time=500)
        self.collection.get(id=id).location=target_coords


    def update(self, id_slots):
        slots=[pair[1] for pair in id_slots]
        locations=[self.collection.get(slot=slot).location for slot in slots]
        for i in range(len(id_slots)):
            self._move_piece_to_target(id_slots[i][0], locations[i])

        self.collection.mass_set("slot", slots)

    def move_pieces(self, iterator, start):
        raise NotImplementedError()
        self.pieces=PieceCollection([], self.pieces.dims)
        try:
            image=next(iterator)
            self.move_piece(image.id, image.location)
            #print((datetime.now()-start).seconds*1000+(datetime.now()-start).microseconds/1000)
            self.pieces.add(image.array)
            self.after(0, self.move_pieces(iterator, start))
        except StopIteration:
            pass

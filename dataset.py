import json
import numpy as np
import random
import cv2
from os import makedirs
from pathlib import Path
from shutil import rmtree

class LocationGenerator(object):
    def __init__(self, dimensions, ratio=0.7):
        self.dimensions=dimensions
        self.ratio=ratio
        self.S=None
        self.references=None

    def get_grid_locations(self, count):
        for i in range(10):
            if count<=i**2:
                root_div=i
                break
            if i==12:
                raise Exception("Too many pieces!")
        self.S=root_div
        zones, m_sizes=self.gen_references()
        locations=random.sample([(i,j) for i in range(root_div)
                                       for j in range(root_div)], count)
        return ([zones[location] for location in locations], m_sizes)

    def get_random_positions(self, count, max_fill=0.8):
        grid_slots, m_sizes=self.get_grid_locations(count)
        pix_locations=[]
        for slot in grid_slots:
            x=int(random.random()*(slot[3]-max_fill*m_sizes[1])+slot[1])
            y=int(random.random()*(slot[2]-max_fill*m_sizes[0])+slot[0])
            pix_locations.append((x,y))
        return (pix_locations, (int(max_fill*m_sizes[0]), int(max_fill*m_sizes[1])))

    def random_length_split(self, length):
        assert self.ratio<1 and self.ratio>0
        m_length=int(length*self.ratio/(self.S-(1-self.ratio)*(self.S//2)))

        sub_lengths=[]
        for _ in range(self.S//2):
            sub_lengths.append(m_length)
        remaining=length-m_length*self.S//2

        l_length=remaining//(self.S-self.S//2)
        for _ in range(self.S-self.S//2-1):
            sub_lengths.append(l_length)
        assert length-sum(sub_lengths)>0

        sub_lengths.append(length-sum(sub_lengths))
        random.shuffle(sub_lengths)
        return (sub_lengths, m_length)

    def _ref_slices(self, value, dim, slice_dim, coord):
        """sets all items of a slice of said dimension at said coordinate(not across) to said value"""
        if dim==0:
            self.references[slice_dim, :, coord] = value
        elif dim==1:
            self.references[:, slice_dim, coord] = value
        else:
            raise Exception()
        return self.references

    def gen_references(self):
        self.references=np.zeros((self.S, self.S, 4), dtype=np.int32)
        lengths_y=self.random_length_split(self.dimensions[0])
        lengths_x=self.random_length_split(self.dimensions[1])
        lengths=(lengths_y, lengths_x)
        for dim in (0,1):
            cumulative=0
            for i, l in enumerate(lengths[dim][0]):
                self._ref_slices(cumulative, dim, i, dim)
                self._ref_slices(l, dim, i, 2+dim)
                cumulative+=l

        m_size=(lengths[0][1], lengths[1][1])
        return (self.references, m_size)


class ImageSplitter(object):
    def __init__(self, dimensions=(1024,768), txtpath="./truthvalues.json"):
        self.json_writer=TruthWriter(txtpath)
        self.pix_dimensions=dimensions
        self.location_generator=LocationGenerator((dimensions[1], dimensions[0]))

    def rescale_list(self, images, piece_size):
        ratio=min(piece_size[0]/images[0].shape[0], piece_size[1]/images[0].shape[1])
        return [cv2.resize(image, (0,0), fx=ratio, fy=ratio) for image in images]

    def place(self, image, subimage, pix_location):
        image[pix_location[1]:pix_location[1]+subimage.shape[0], pix_location[0]:pix_location[0]+subimage.shape[1]] = subimage
        return image

    def flat_split(self, image, slices):
        splits=np.array_split(image, slices[1], 1)
        out=[]
        for split in splits:
            out+=np.array_split(split, slices[0], 0)
        return out

    def place_pieces(self, piece_list, max_fill=0.9):
        assert max_fill<=1 and max_fill>0

        pix_locations, piece_size=self.location_generator.get_random_positions(len(piece_list), max_fill)
        pieces=self.rescale_list(piece_list, piece_size)
        #out=np.ones((self.pix_dimensions[1], self.pix_dimensions[0], 3), dtype=np.int8)*255
        out=np.ones((self.pix_dimensions[1], self.pix_dimensions[0], 3), dtype=np.int8)*255

        truth_boxes={}

        for i, pix_location, piece in zip(range(len(pieces)), pix_locations, pieces):
            out=self.place(out, piece, pix_location)
            truth_boxes[i]=[pix_location[0], pix_location[1], piece.shape[1], piece.shape[0]]

        return (out, truth_boxes)

    def gen(self, in_path, out_location, dims=(4,4), min=-1):
        suffix=in_path.suffix.lower()
        assert suffix==".jpg" or suffix==".png"

        if not(out_location.exists()):
            print()
            print("Output directory does not exist")
            create = input("Create directory?")
            if create == "y" or create =="Y":
                makedirs(out_location)

        image=cv2.imread(in_path.as_posix(), flags=1)

        pieces=self.flat_split(image, dims)
        random.shuffle(pieces)

        if min==-1:
            min=len(pieces)

        pieces=pieces[:random.randint(min, len(pieces))]
        image, truths=self.place_pieces(pieces)

        new_path=out_location/in_path.name
        print(str(new_path)+" - "+str(len(pieces))+" pieces")
        cv2.imwrite(new_path.as_posix(), image)
        self.json_writer.add_image(new_path.as_posix(), truths)

    def find_images(self, path=Path("./"), extension=".jpg"):
        image_iterator=path.glob("*"+extension)
        return image_iterator

    def gen_all(self, path, extension=".jpg", **kwargs):
        path=Path(path)
        assert path.is_dir()
        spath=path / "samples"
        try:
            makedirs(spath)
        except FileExistsError as E:
            delete=input("Delete samples folder?")
            if delete=="Y" or delete=="y":
                rmtree(spath)
                makedirs(spath)
            else:
                raise E

        for image_path in self.find_images(path, extension):
            self.gen(in_path=image_path, out_location=spath, **kwargs)

    def __enter__(self):
        pass

    def __exit__(self, *args):

        print(args)
        self.close()

    def close(self):
        self.json_writer.close()

class TruthWriter(object):

    def __init__(self, filename):
        self.filename=filename
        self.dict={}

    def add_image(self, image_name, truth_values):
        """Writes as (y,x) image coordinates"""
        self.dict[image_name] = truth_values

    def close(self):
        file=open(self.filename, "w")
        json.dump(self.dict, file)
        file.close()
        self.dict={}

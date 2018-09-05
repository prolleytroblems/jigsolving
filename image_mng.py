

class PieceCollection:

    def __init__(self, images, dims):
        self._images=images
        self.dims=dims

    def get(self):
        return self._images

    def add(self, image):
        self._images.insert(0, image)



class PieceFinder(object):
    def __init__(self, **kwargs):
        init selective searcher
        init BBoxFilter

    def find_boxes(self, image):
        feed image to selective search
        pass boxes through boxfilter
        try to get dimensions

    def get_boxes(self, path, check_dims=False, iter=1 **kwargs):
        if iter>n:
            raise Exception("Could not find good boxes with given restrictions.")

        array = cv2.imread(path.tostr)

        box_list = self.find_boxes(array)

        if check_dims:
            full_shape=kwargs["full_shape"]
            similarity = self.check_dims(box_list, full_shape)
            if similarity > something:
                return box_list
            else:
                return self.get_boxes(path, check_dims=True, full_shape=full_shape, iter=iter+1)
        else:
            return box_list


    def check_dims(self, box_list, full_shape)


class BBoxFilter(object):

    def __init__(self, image, edgewidth=1):
        pass

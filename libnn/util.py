class ForwardBackwardStorage:
    def __init__(self):
        self.__saved_tensors = None

    @property
    def saved_tensors(self):
        saved_tensors, self.__saved_tensors = self.__saved_tensors, None
        return saved_tensors

    @saved_tensors.setter
    def saved_tensors(self, tensors):
        if getattr(self, 'training', True):
            self.__saved_tensors = tensors

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

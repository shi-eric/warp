
import ctypes
import numpy as np

#----------------------
# built-in types


class vec3:
    def __init__(self):
        x = 0.0
        y = 0.0
        z = 0.0

    @staticmethod
    def length():
        return 3

    @staticmethod
    def size():
        return 12

class vec4:
    def __init__(self):
        x = 0.0
        y = 0.0
        z = 0.0
        w = 0.0

    @staticmethod
    def length():
        return 4


    @staticmethod
    def size():
        return 16

class quat:
    def __init__(self):
        x = 0.0
        y = 0.0
        z = 0.0
        w = 1.0

    @staticmethod
    def length():
        return 4

    @staticmethod
    def size():
        return 16


class mat22:
    def __init__(self):
        pass

    @staticmethod
    def length():
        return 4

    @staticmethod
    def size():
        return 16

class mat33:
    def __init__(self):
        pass

    @staticmethod
    def length():
        return 9

    @staticmethod
    def size():
        return 36

class spatial_vector:
    def __init__(self):
        pass

    @staticmethod
    def length():
        return 6

    @staticmethod
    def size():
        return 36

class spatial_matrix:
    def __init__(self):
        pass

    @staticmethod
    def length():
        return 36

    @staticmethod
    def size():
        return 144

class spatial_transform:
    def __init__(self):
        pass

    @staticmethod
    def length():
        return 7
    
    @staticmethod
    def size():
        return 28

class void:
    def __init__(self):
        pass

class float32:

    @staticmethod
    def length():
        return 1
    
    @staticmethod
    def size():
        return 4

class float64:

    @staticmethod
    def length():
        return 1
    
    @staticmethod
    def size():
        return 8

class int32:

    @staticmethod
    def length():
        return 1
    
    @staticmethod
    def size():
        return 4

class int64:

    @staticmethod
    def length():
        return 1
    
    @staticmethod
    def size():
        return 8


def type_length(dtype):
    if (dtype == float or dtype == int):
        return 1
    else:
        return dtype.length()

def type_size_in_bytes(dtype):
    if (dtype == float or dtype == int):
        return 4
    else:
        return dtype.size()

class array:

    def __init__(self, dtype, length=0, capacity=0, data=None, device=None, context=None, owner=True):
        self.length = length
        self.capacity = capacity
        self.dtype = dtype
        self.data = data
        self.device = device
        self.context = context
        self.owner = owner

        self.__name__ = "array<" + type.__name__ + ">"

    def __del__(self):
        
        if (self.owner and self.context):

            if (self.device == "cpu"):
                self.context.free_host(self.data)
            else:
                self.context.free_device(self.data)
                

    def numpy(self):

        if (self.device == "cpu"):

            # todo: make each og type return it's corresponding ctype
            ptr_type = ctypes.POINTER(ctypes.c_float)
            ptr = ctypes.cast(self.data, ptr_type)

            view = np.ctypeslib.as_array(ptr, shape=(self.length, type_length(self.dtype)))
            return view
        
        else:
            raise RuntimeError("Cannot convert CUDA array to numpy, copy to a host array first")


    def to(self, device):

        if (self.device == device):
            return self
        else:
            from oglang.context import empty, copy, synchronize

            dest = empty(n=self.capacity, dtype=self.dtype, device=device)
            copy(dest, self)
            synchronize()

            return dest



class Mesh:

    def __init__(self, points, indices, device):
        
        self.points = points
        self.indices = indices

        self.id = runtime.mesh_create_host()

    def __del__(self):

        runtime.mesh_destroy_host(self.id)

    def refit(points, indices):
        
        self.points = points
        self.indices = indices

        runtime.mesh_update_host()

    def bounds():
        pass



class Volume:

    def __init__(self, vdb):
        pass




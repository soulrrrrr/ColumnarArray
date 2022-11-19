import _col_array

class ColArray():

    def __init__(self, data):
        self.arrays = []
        for i in range(len(data[0])):
            try:
                self.arrays.append(_col_array.IntArray([arr[i] for arr in data]))
            except:
                self.arrays.append(_col_array.FloatArray([arr[i] for arr in data]))


    def show(self):
        for arr in self.arrays:
            for i in range(arr.len):
                print(arr[i],  ' ')

    def size(self):
        return self.arrays[0].len, len(self.arrays)



    

    
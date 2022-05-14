from numpy import (float32 as np_float32,
                   float64 as np_float64,
                   int64 as np_int64,
                   int32 as np_int32,
                   savetxt as np_savetxt)


class File_Handler:
    file_name = None
    log_to_file = False

    def __init__(self, log = False):
        self.log_to_file = log

    def open_file(self, name_of_file, precision):
        if self.log_to_file:
            if isinstance(precision(0), (np_float32)):
                tp = "32bit"
            elif isinstance(precision(0), (np_float64)):
                tp = "64bit"
            elif isinstance(precision(0), (np_int64)):
                tp = "int64"
            elif isinstance(precision(0), (np_int32)):
                tp = "int32"
            self.file_name = open(f"{name_of_file}_{tp}.dat", "w")

    def close_file(self):
        if self.log_to_file:
            self.file_name.close()

    def print_to_file(self, matrix_to_compare, test_matrix, test_result):
        if self.log_to_file:
                try:
                    self.file_name.write("OPENCL RESULT: \n")
                    np_savetxt(self.file_name, matrix_to_compare)
                    self.file_name.write("\n \n")
                    self.file_name.write("NUMPY/MATH RESULT: \n")
                    np_savetxt(self.file_name, test_matrix)
                    self.file_name.write("\n \n")
                    self.file_name.write("ABSOLUTE ERROR: \n")
                    np_savetxt(self.file_name, test_result)
                except ValueError:
                    self.file_name.write(str(matrix_to_compare))
                    self.file_name.write("\n \n")
                    self.file_name.write("NUMPY/MATH RESULT: \n")
                    self.file_name.write(str(test_matrix))
                    self.file_name.write("\n \n")
                    self.file_name.write("ABSOLUTE ERROR: \n")
                    self.file_name.write(str(test_result))
                    self.file_name.write("\n \n")

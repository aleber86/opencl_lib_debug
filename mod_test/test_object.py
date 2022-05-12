import numpy as np

class Test_Class:
    NO_COLOR = "\033[0m"
    GREEN = "\033[48;5;82m"
    RED = "\033[48;5;160m"
    YELLOW = "\033[48;5;190m"
    Compare_Value = None
    function_test_result_matrix = []
    reduction_result = 0
    test_dictionary = {"np_test_matrix" : ["Sum", "Substract",
                                           "Product", "Division",
                                           "ACOS", "ACOSH", "ACOS/PI",
                                           "ASIN", "ASINH", "ASIN/PI",
                                           "ATAN", "ATAN(X;Y)/PI", "ATAN",
                                           "ATAN2", "ATAN(X;Y)/PI","CBRT",
                                           "CEIL", "Copysign", "COS",
                                           "COSH", "COS(PI*X)", "ERF",
                                           "ERFC", "EXP", "EXP2", "EXP10",
                                           "EXP(X) - 1", "ABS", "FDIM",
                                           "FMA", "MAX", "MIN", "MOD",
                                           "FRAC", "HYPOT", "LDEXP"],
                       "reduction": ["Reduction on Global Mem"]}

    HEADERS = 80*"*"
    TITLE = f"TEST NAME{20*' '} STATUS\t GOOD\t\t WARN\t FAILED (>Abs. err.)"

    def fdim(self, arg1, arg2, _wp = np.float64):
        matrix_out = np.zeros(arg1.shape, dtype = _wp)
        for index,value in enumerate(arg1):
            if value > arg2:
                matrix_out[index] =_wp(value - arg2)
            else:
                matrix_out[index] = _wp(+0)
        return matrix_out

    def erf(self, arg1, _wp = np.float64):
        matrix_out = np.zeros(arg1.shape, dtype = _wp)
        for index,value in enumerate(arg1):
            matrix_out[index] = np.math.erf(value)
        return matrix_out

    def erfc(self, arg1, _wp = np.float64):
       return _wp(1.0) - self.erf(arg1, _wp)

    def numpy_test(self, matrix_argument_in : "np.array", _wp : "type(precision)"= np.float64):
        matrix_argument_out = np.zeros(matrix_argument_in.shape, dtype = _wp)

        matrix_argument_out[0] += matrix_argument_in[0]
        matrix_argument_out[1] -= matrix_argument_in[1]
        matrix_argument_out[2] = matrix_argument_in[2] * _wp(1.1)
        matrix_argument_out[3] = matrix_argument_in[3] / _wp(1.1)
        matrix_argument_out[4] = np.arccos(matrix_argument_in[4])
        matrix_argument_out[5] = np.arccosh(matrix_argument_in[5])
        matrix_argument_out[6] = np.arccos(matrix_argument_in[6]) / np.pi
        matrix_argument_out[7] = np.arcsin(matrix_argument_in[7])
        matrix_argument_out[8] = np.arcsinh(matrix_argument_in[8])
        matrix_argument_out[9] = np.arcsin(matrix_argument_in[9]) / np.pi
        matrix_argument_out[10] = np.arctan(matrix_argument_in[10])
        matrix_argument_out[11] = np.arctan2(np.float64(2.0), matrix_argument_in[11])
        matrix_argument_out[12] = np.arctanh(matrix_argument_in[12])
        matrix_argument_out[13] = np.arctan(matrix_argument_in[13]) / np.pi
        matrix_argument_out[14] = np.arctan2(np.float64(2.0), matrix_argument_in[14]) / np.pi
        matrix_argument_out[15] = np.cbrt(matrix_argument_in[15])
        matrix_argument_out[16] = np.ceil(matrix_argument_in[16])
        matrix_argument_out[17] = np.copysign(matrix_argument_in[17], _wp(-1.0))
        matrix_argument_out[18] = np.cos(matrix_argument_in[18])
        matrix_argument_out[19] = np.cosh(matrix_argument_in[19])
        matrix_argument_out[20] = np.cos(np.pi * matrix_argument_in[20])
        matrix_argument_out[21] = self.erf(matrix_argument_in[21])
        matrix_argument_out[22] = self.erfc(matrix_argument_in[22])
        matrix_argument_out[23] = np.exp(matrix_argument_in[23])
        matrix_argument_out[24] = np.exp2(matrix_argument_in[24])
        matrix_argument_out[25] = np.power(_wp(10.0), matrix_argument_in[25])
        matrix_argument_out[26] = np.expm1(matrix_argument_in[26])
        matrix_argument_out[27] = np.fabs(matrix_argument_in[27])
        matrix_argument_out[28] = self.fdim(matrix_argument_in[28], _wp(2.0)) #fdim ---> generar funcion
        matrix_argument_out[29] = _wp(2.0*1.1) + matrix_argument_in[29]
        matrix_argument_out[30] = np.fmax(np.float64(2.0),matrix_argument_in[30])
        matrix_argument_out[31] = np.fmin(np.float64(2.0),matrix_argument_in[31])
        matrix_argument_out[32] = np.fmod(matrix_argument_in[32], _wp(2.1))
        #matrix_argument_out[33] = frac
        matrix_argument_out[34] = np.hypot(matrix_argument_in[34], _wp(2.1))
        matrix_argument_out[35] = np.ldexp(matrix_argument_in[35], np.int32(4))

        self.function_test_result_matrix = matrix_argument_out

    def function_test_result(self, matrix_to_compare, limit = 1.e-15, verbose = True, warn_test = [21,22]):
        print("OpenCL Functions:\n")
        test = np.abs(self.function_test_result_matrix - matrix_to_compare)
        self._operation(test, self.test_dictionary["np_test_matrix"], limit, verbose, warn_test)

    def reduction_(self, matrix_in, matrix_to_compare, verbose = True, warn_test = [], limit = 1.e-15):

        print("Reduction:\n")
        self.reduction_result = np.sum(matrix_in)
        test = [[np.abs(self.reduction_result - matrix_to_compare)]]
        self._operation(test, self.test_dictionary["reduction"], limit, verbose, warn_test)

    def _color_output(self, value_to_test, condition_result_in, precision = 1.e-15, expected = False):

        if value_to_test <= precision:
            test =  True
        elif (value_to_test > precision and not expected):
            test = False
        elif (value_to_test > precision and expected):
            test = None
        condition_result_in.append(test)

    def _output_value(self,name_op, condition_result, print_value, tabs):
        DIGIT = 9
        True_number = condition_result.count(True)
        False_number = condition_result.count(False)
        None_number = condition_result.count(None)
        length = len(condition_result)
        HEAD = f"{name_op}{tabs}"
        TRUE = f"{True_number}/{length}"
        WARN = f"{None_number}/{length}"
        FAIL = f"{False_number}/{length}"
        PAD_TRUE, PAD_WARN, PAD_FAIL = DIGIT-len(TRUE), DIGIT-len(WARN), DIGIT-len(FAIL)

        if True_number==length:
            PRINT_OUT = f"{HEAD}{self.GREEN}  OK  {self.NO_COLOR}"
        elif (True_number<length and None_number>0 and False_number==0):
            PRINT_OUT = f"{HEAD}{self.YELLOW} WARN {self.NO_COLOR}"
        elif True_number<length and False_number>0:
            PRINT_OUT = f"{HEAD}{self.RED} FAIL {self.NO_COLOR}"
        PRINT_OUT_ALL = f"{PRINT_OUT}\t{TRUE}{PAD_TRUE*' '}\t{WARN}{PAD_WARN*' '}"\
            f"{FAIL}{PAD_FAIL*' '}\t {print_value}"
        print(PRINT_OUT_ALL)


    def _operation(self, test_value, dictionary_item, limit, verbose, warn):
        self._print_headers()
        index = 0
        for value_row, name_op in zip(test_value, dictionary_item):
            if verbose:
                print_value = np.max(value_row)
            else:
                print_value = ""

            expected_value = False
            tabs = 30-len(name_op)
            tabs = tabs*" "
            condition_result = []
            for value in value_row:
                if index in warn:
                    expected_value = True
                self._color_output(value, condition_result, limit, expected = expected_value)

            self._output_value(name_op, condition_result, print_value, tabs)
            index+=1

    def _print_headers(self):
        print(self.HEADERS)
        print(self.TITLE)
        print(self.HEADERS)

import numpy as np

class Test_Class:
    NO_COLOR = "\033[0m"
    GREEN = "\033[48;5;82m"
    RED = "\033[48;5;160m"
    YELLOW = "\033[48;5;190m"
    Compare_Value = None
    function_test_result_matrix = []
    reduction_result = 0
    matrix_id_value = []
    matrix_product = []
    test = []
    test_dictionary = {"np_test_matrix" : ["Sum", "Substract",
                                           "Product", "Division",
                                           "ACOS", "ACOSH", "ACOS/PI",
                                           "ASIN", "ASINH", "ASIN/PI",
                                           "ATAN", "ATAN(X;Y)/PI", "ATAN",
                                           "ATAN2", "ATAN(X;Y)/PI","CBRT",
                                           "CEIL", "Copysign", "COS",
                                           "COSH", "COS(PI*X)", "ERF",
                                           "ERFC", "EXP", "EXP2", "EXP10 Numpy power(10.0, n)",
                                           "EXP(X) - 1", "ABS", "FDIM",
                                           "FMA", "MAX", "MIN", "MOD",
                                           "FRAC", "HYPOT", "LDEXP",
                                           "ILOGB", "LGAMMA (math lib)", "LGAMMAR (math lib)",
                                           "LOG", "LOG2", "LOG10", "NEXTAFTER",
                                           "POW", "POWN", "POWR", "MAD", "MODF (mantisa)", "MODF (int)",
                                           "REMINDER", "REMQUO", "RINT", "ROOTN", "ROUND", "RSQRT",
                                           "SIN", "SINCOS (SIN)", "SINCOS (COS)", "SINH", "SINPI",
                                           "SQRT", "TAN", "TANH", "TANPI", "TGAMMA (math lib)", "TRUNC"],
                       "reduction": ["Reduction on Global Mem"],
                       "id_test" : ["Id test"], "matrix_product": ["Matrix Product nxn"]}

    HEADERS = 100*"*"
    TITLE = f"TEST NAME{20*' '} STATUS\t GOOD\t\t WARN\t FAILED \tAbs. err. (min, max)"


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

    def gamma(self, arg1, _wp = np.float64):
        matrix_out = np.zeros(arg1.shape, dtype = _wp)
        for index,value in enumerate(arg1):
            matrix_out[index] = np.math.gamma(value)
        return matrix_out

    def lgamma(self, arg1, _wp = np.float64):
        matrix_out = np.zeros(arg1.shape, dtype = _wp)
        for index,value in enumerate(arg1):
            matrix_out[index] = np.math.lgamma(value)
        return matrix_out

    def remquo(self, arg1, arg2, _wp = np.float64):
        matrix_out = np.zeros(arg2.shape, dtype=_wp)
        for index, value in enumerate(arg2):
            rem = np.remainder(arg1, value)
            matrix_out[index] = rem
        return matrix_out

    def numpy_test(self, matrix_argument_in : "np.array", _wp : "type(precision)"= np.float64):
        """Function test for every math function in OpenCL 1.0 version"""

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
        matrix_argument_out[21] = self.erf(matrix_argument_in[21], _wp)
        matrix_argument_out[22] = self.erfc(matrix_argument_in[22], _wp)
        matrix_argument_out[23] = np.exp(matrix_argument_in[23])
        matrix_argument_out[24] = np.exp2(matrix_argument_in[24])
        matrix_argument_out[25] = np.power(_wp(10.0), matrix_argument_in[25])
        matrix_argument_out[26] = np.expm1(matrix_argument_in[26])
        matrix_argument_out[27] = np.fabs(matrix_argument_in[27])
        matrix_argument_out[28] = self.fdim(matrix_argument_in[28], _wp(2.0), _wp) #fdim ---> generar funcion
        matrix_argument_out[29] = _wp(2.0*1.1) + matrix_argument_in[29]
        matrix_argument_out[30] = np.fmax(_wp(2.0),matrix_argument_in[30])
        matrix_argument_out[31] = np.fmin(_wp(2.0),matrix_argument_in[31])
        matrix_argument_out[32] = np.fmod(matrix_argument_in[32], _wp(2.1))
        matrix_argument_out[33] = matrix_argument_in[33] - np.floor(matrix_argument_in[33])
        matrix_argument_out[34] = np.hypot(matrix_argument_in[34], _wp(2.1))
        matrix_argument_out[35] = np.ldexp(matrix_argument_in[35], np.int32(4))
        matrix_argument_out[36] = np.floor(np.log2(matrix_argument_in[36]))
        matrix_argument_out[37] = self.lgamma(matrix_argument_in[37], _wp)
        matrix_argument_out[38] = self.lgamma(np.abs(matrix_argument_in[38]), _wp)
        matrix_argument_out[39] = np.log(matrix_argument_in[39])
        matrix_argument_out[40] = np.log2(matrix_argument_in[40])
        matrix_argument_out[41] = np.log10(matrix_argument_in[41])
        matrix_argument_out[42] = np.log1p(matrix_argument_in[42])
        matrix_argument_out[43] = np.nextafter(_wp(2.0),matrix_argument_in[43])
        matrix_argument_out[44] = np.power(_wp(2.0),matrix_argument_in[44])
        matrix_argument_out[45] = np.power(matrix_argument_in[45], np.int32(2))
        matrix_argument_out[46] = np.power(matrix_argument_in[46], np.int32(2))
        matrix_argument_out[47] = _wp(2.0)*_wp(1.1) + matrix_argument_in[47]
        matrix_argument_out[48], matrix_argument_out[49] = np.modf(matrix_argument_in[48])
        matrix_argument_out[50] = np.remainder(_wp(2.1), matrix_argument_in[50])
        matrix_argument_out[51] = self.remquo(_wp(15.5), matrix_argument_in[51], _wp)
        matrix_argument_out[52] = np.rint(matrix_argument_in[52])
        matrix_argument_out[53] = np.power(np.abs(matrix_argument_in[53]), 1/6)
        matrix_argument_out[54] = np.round(matrix_argument_in[54])
        matrix_argument_out[55] = _wp(1.0/np.sqrt(np.abs(matrix_argument_in[55])))
        matrix_argument_out[56] = np.sin(matrix_argument_in[56])
        matrix_argument_out[57] = np.sin(matrix_argument_in[57])
        matrix_argument_out[58] = np.cos(matrix_argument_in[57])
        matrix_argument_out[59] = np.sinh(matrix_argument_in[58])
        matrix_argument_out[60] = np.sin(np.pi*matrix_argument_in[59])
        matrix_argument_out[61] = np.sqrt(np.abs(matrix_argument_in[60]))
        matrix_argument_out[62] = np.tan(matrix_argument_in[61])
        matrix_argument_out[63] = np.tanh(matrix_argument_in[62])
        matrix_argument_out[64] = np.tan(np.pi*matrix_argument_in[63])
        matrix_argument_out[65] = self.gamma(matrix_argument_in[64])
        matrix_argument_out[66] = np.trunc(matrix_argument_in[65])

        self.function_test_result_matrix = matrix_argument_out

    def matrix_id(self, dimension):
        """Create matrix with static result expected for id_test_dimension"""

        dimension_max_value = np.max(dimension)
        matrix_for_test = np.zeros((dimension_max_value), dtype = np.int32)
        for index, num in enumerate(matrix_for_test):
            matrix_for_test[index] += index+dimension[1]+dimension[2]-2
        self.matrix_id_value = matrix_for_test


    def id_test_dimension(self, matrix_to_compare, dimension, limit = 0, warn_limit = -1, verbose = True, warn_test = []):
        """Test and compare results from OpenCL get_global_id and global = get_group_id*get_local_size + get_local_id
        using a problem shape of (max_dim, max_dim, max_dim) from device propierties. Local size
        is set by shape (4,4,4)"""

        print(f"Id test using absolute error limit {limit}\n")
        test = np.abs(self.matrix_id_value-matrix_to_compare)
        self.test = [test]
        self._operation(self.test, self.test_dictionary["id_test"], limit, warn_limit, verbose, warn_test)
        print(self.HEADERS,"\n")

    def function_test_result(self, matrix_to_compare, limit = 1.e-15, warn_limit = 1.e-14,
                             verbose = True, warn_test = [21,22]):
        """Test and compare results between math functions in math/numpy lib and OpenCL 1.0"""

        print(f"OpenCL Functions using absolute error limit {limit}\n")
        self.test = np.abs(self.function_test_result_matrix - matrix_to_compare)
        self._operation(self.test, self.test_dictionary["np_test_matrix"], limit, warn_limit, verbose, warn_test)
        print(self.HEADERS,"\n")


    def reduction_(self, matrix_in, matrix_to_compare, verbose = True, warn_test = [],
                   limit = 1.e-15, warn_limit = 1.e-14):
        """Test and compare results of reduction made in OpenCL"""

        print(f"Reduction using absolute error limit {limit}\n")
        self.reduction_result = np.sum(matrix_in)
        self.test = [[np.abs(self.reduction_result - matrix_to_compare)]]
        self._operation(self.test, self.test_dictionary["reduction"], limit, warn_limit, verbose, warn_test)
        print(self.HEADERS,"\n")


    def matrix_product_definition(self, matrix_1, matrix_2):
        self.matrix_product = np.matmul(matrix_1, matrix_2)

    def matrix_product_test(self, matrix_to_compare, limit = 1.e-15, warn_limit = 1.e-14,
                             verbose = True, warn_test = [21,22]):
        """Test and compare results between matrix product in math/numpy lib and OpenCL 1.0"""

        print(f"OpenCL Functions using absolute error limit {limit}\n")
        self.test = np.abs(self.matrix_product - matrix_to_compare)
        self._operation(self.test, self.test_dictionary["matrix_product"], limit, warn_limit, verbose, warn_test)
        print(self.HEADERS,"\n")


    def _color_output(self, value_to_test, condition_result_in, precision = 1.e-15,
                      warn_limit = 1.e-14, expected = False):
        """Create list with color index for print"""

        if value_to_test <= precision:
            test =  True
        elif ((value_to_test > precision and not expected) and
              (value_to_test > warn_limit)):
            test = False
        elif ((value_to_test > precision and expected) or
              (value_to_test > precision and value_to_test < warn_limit)):
            test = None
        condition_result_in.append(test)

    def _output_value(self,name_op, condition_result, print_value, tabs, array_2_d = False):
        """Print on screen results as good/warn/failed. 'print_value' is the
        absolute error of test values"""

        DIGIT = 9
        True_number = condition_result.count(True)
        False_number = condition_result.count(False)
        None_number = condition_result.count(None)

        if array_2_d:
            length = len(condition_result)*len(condition_result[0])
        else:
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


    def _operation(self, test_value, dictionary_item, limit, warn_limit, verbose, warn):
        """Execute the compare function. 'tabs' gives the pad space for _output_value function"""

        self._print_headers()
        index = 0
        for value_row, name_op in zip(test_value, dictionary_item):
            if verbose:
                print_value = (np.min(value_row), np.max(value_row))
            else:
                print_value = ""

            expected_value = False
            tabs = 30-len(name_op)
            tabs = tabs*" "
            condition_result = []
            for value in value_row:
                if index in warn:
                    expected_value = True
                self._color_output(value, condition_result, limit, warn_limit, expected = expected_value)

            self._output_value(name_op, condition_result, print_value, tabs)
            index+=1

    def _print_headers(self):
        """Print function for headers"""

        print(self.HEADERS)
        print(self.TITLE)
        print(self.HEADERS)



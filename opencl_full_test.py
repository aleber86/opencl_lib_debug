#!/usr/bin/env python3


import numpy as np
import pyopencl as cl
from mod_cmd.command_line_arguments import menu_cmd
from mod_opencl.opencl_class_device import OpenCL_Object, Platform_Device_OpenCL
from mod_test.test_object import Test_Class


def id_test_dimension(Instance_OpenCL_Object : "OpenCL_Object", Test_obj,
                      _wp : "type(..,16,32,64..)" = np.int32,
                      kernel_name = "kernel_full_debug",
                      Dev_prop = None, precision = 0 ):

    if Dev_prop is None:
        total_size_per_dim = 2**8
        problem_shapie = (total_size_per_dim,)
        Total_size = problem_shape
    else:
        problem_shape = tuple((dimension for dimension in Dev_prop[3]))
        Total_size = max(Dev_prop[3])

    matrix_id_global = np.zeros(Total_size, dtype = _wp)
    matrix_id_local = np.zeros_like(matrix_id_global, dtype = _wp)
    Instance_OpenCL_Object.program(kernel_name)
    Instance_OpenCL_Object.buffer_global(matrix_id_global, "matrix_in_gid")
    Instance_OpenCL_Object.buffer_global(matrix_id_local, "matrix_in_lid")

    Instance_OpenCL_Object.kernel.id_test_dimension(Instance_OpenCL_Object.queue,
                                                  problem_shape, (4,4,4),
                                                  Instance_OpenCL_Object.matrix_in_gid_device,
                                                  Instance_OpenCL_Object.matrix_in_lid_device)
    Instance_OpenCL_Object.queue.finish()

    cl.enqueue_copy( Instance_OpenCL_Object.queue,
                     Instance_OpenCL_Object.matrix_in_gid,
                     Instance_OpenCL_Object.matrix_in_gid_device)

    cl.enqueue_copy( Instance_OpenCL_Object.queue,
                     Instance_OpenCL_Object.matrix_in_lid,
                     Instance_OpenCL_Object.matrix_in_lid_device)
    Test_obj.matrix_id(Dev_prop[3])
    Test_obj.id_test_dimension(Instance_OpenCL_Object.matrix_in_gid, precision)
    Test_obj.id_test_dimension(Instance_OpenCL_Object.matrix_in_lid, precision)
    Instance_OpenCL_Object.free_buffer()
    Instance_OpenCL_Object.reset_attrib()

def reduccion(Instance_OpenCL_Object, Test_obj,
              _wp : "type(..,16,32,64..)" = np.int32,
              Dev_prop = None, kernel_name = "kernel_full_debug",
              precision = 1.e-6, w_limit = 1.e-5, verb = True):

    if Dev_prop is None:
        local_size = [16]
        element_quant = 2**12
    else:
        element_quant = int(Dev_prop[3][0])
        local_size = Dev_prop[4]
    problem_shape = element_quant
    #Test.
    matrix_in = np.array([np.random.randint(0,2) for i in range(element_quant)], dtype=_wp)
    matrix_out = np.zeros(problem_shape, dtype=_wp)
    Instance_OpenCL_Object.buffer_global(matrix_in, "matrix_in", False)
    Instance_OpenCL_Object.buffer_global(matrix_out, "matrix_out")
    Instance_OpenCL_Object.program(kernel_name) #<- comiled at first test

    for local_value in local_size:
        local_value_size = (int(local_value), )
        Instance_OpenCL_Object.kernel.reduccion(Instance_OpenCL_Object.queue,
                                                           (problem_shape,), local_value_size,
                                                           Instance_OpenCL_Object.matrix_in_device,
                                                           Instance_OpenCL_Object.matrix_out_device)
        Instance_OpenCL_Object.queue.finish()
        cl.enqueue_copy(Instance_OpenCL_Object.queue,
                        Instance_OpenCL_Object.matrix_out,
                        Instance_OpenCL_Object.matrix_out_device)
        suma =np.sum(Instance_OpenCL_Object.matrix_out)
        Test_obj.reduction_(matrix_in, suma, verbose = verb, warn_test = [], limit = precision, warn_limit = w_limit)
        #Reset buffer out
        matrix_out = np.zeros(problem_shape, dtype=_wp)
        Instance_OpenCL_Object.buffer_global(matrix_out, "matrix_out")
    #Removes every attribute of Instance_OpenCL_Object
    Instance_OpenCL_Object.free_buffer()
    Instance_OpenCL_Object.reset_attrib()


def functions_test(Instance_OpenCL_Object, Test_obj,
                   _wp : "type(..16,32,64,...)" = np.float64,
                   Dev_prop = None, dim = 512, kernel_name = "kernel_full_debug",
                   precision = 1.e-6, w_limit = 1.e-5, verb = True):


    if isinstance(_wp(0.0), (np.float64, np.float32)):
        matrix_in = np.array([[np.random.random() for _ in range(dim)] for _ in range(47)],dtype = _wp)
        matrix_in[5] = np.fabs(matrix_in[5]  ) + _wp(1.0)#arccosh >= 1
        matrix_in[33] = np.fabs(matrix_in[33]  ) + _wp(10.0)#arccosh >= 1
        matrix_in[37] = np.fabs(matrix_in[37]  ) + _wp(1.0)#lgamma > 0
        matrix_in[39] = np.fabs(matrix_in[39]  ) + _wp(1.0)#log > 0
        matrix_in[40] = np.fabs(matrix_in[40]  ) + _wp(1.0)#log2 > 0
        matrix_in[41] = np.fabs(matrix_in[41]  ) + _wp(1.0)#log10 > 0
        matrix_in[42] = np.fabs(matrix_in[42]  ) + _wp(1.0)#log1p > 0
        matrix_in[46] = np.fabs(matrix_in[46]  ) + _wp(1.0)#powr >= 0
    else:
        matrix_in = np.array([[np.random.randint(0,1) for _ in range(dim)] for _ in range(47)],dtype = _wp)
        matrix_in[5] = _wp(1)
        matrix_in[37] = np.fabs(matrix_in[37]  ) + _wp(1)#lgamma > 0
        matrix_in[39] = np.fabs(matrix_in[39]  ) + _wp(1)#log > 0
        matrix_in[40] = np.fabs(matrix_in[40]  ) + _wp(1)#log2 > 0
        matrix_in[41] = np.fabs(matrix_in[41]  ) + _wp(1)#log10 > 0
        matrix_in[42] = np.fabs(matrix_in[42]  ) + _wp(1)#log1p > 0
        matrix_in[46] = np.fabs(matrix_in[46]  ) + _wp(1)#powr >= 0
    matrix_out = np.zeros(matrix_in.shape, dtype = _wp)
    problem_shape = matrix_in.shape
    Test_obj.numpy_test(matrix_in)
    Instance_OpenCL_Object.buffer_global(matrix_in, "matrix_in", False)
    Instance_OpenCL_Object.buffer_global(matrix_out, "matrix_out")
    Instance_OpenCL_Object.program(kernel_name) #<- comiled at first test
    Instance_OpenCL_Object.kernel.math_functions(Instance_OpenCL_Object.queue,
                                                            (dim,), None,
                                                            Instance_OpenCL_Object.matrix_in_device,
                                                            Instance_OpenCL_Object.matrix_out_device)
    Instance_OpenCL_Object.queue.finish()
    cl.enqueue_copy(Instance_OpenCL_Object.queue,
                    Instance_OpenCL_Object.matrix_out,
                    Instance_OpenCL_Object.matrix_out_device)

    Test_obj.function_test_result(Instance_OpenCL_Object.matrix_out, limit = precision,
                                  warn_limit = w_limit, verbose = verb, warn_test = [] )
    Instance_OpenCL_Object.free_buffer()
    Instance_OpenCL_Object.reset_attrib()

def main():
    values_input_value = menu_cmd()
    #Warn limit as order of magnitud -> 1.eX * 10**(int)
    warn_32 = np.float32((10**values_input_value.warn_limit)*
                        np.float32(values_input_value.simple_precision))
    warn_64 = np.float64((10**values_input_value.warn_limit)*
                        np.float64(values_input_value.double_precision))
    simple_precision = np.float32(values_input_value.simple_precision)
    double_precision = np.float64(values_input_value.double_precision)
    verbose = values_input_value.verbose
    file_print = values_input_value.log
    #**************************************************************************

    First_OpenCL_Object = OpenCL_Object() #Instance of OpenCL test object
    Test_Object = Test_Class(file_print) #Instance of test object

    #Device_Propierties -> [[extensions], max_glob_mem, max_loc_mem, [w_it,w_it,w_it]]
    Device_Propierties = First_OpenCL_Object.return_attrib()


    kernel_script = {"float":["kernel_full_debug_32", np.float32,
                              simple_precision,
                              warn_32],
                     "cl_khr_fp64" : ["kernel_full_debug", np.float64,
                                      double_precision,
                                      warn_64] }

    for key, kernel_precision in kernel_script.items():
        #Test calls -> inside OpenCL calls & numpy tests
        id_test_dimension(First_OpenCL_Object, Test_Object ,Dev_prop=Device_Propierties )
        reduccion(First_OpenCL_Object, Test_Object,
                  Dev_prop=Device_Propierties, kernel_name = kernel_precision[0],
                  precision = kernel_precision[2], w_limit = kernel_precision[3], verb = verbose)
        functions_test(First_OpenCL_Object, Test_Object,
                       _wp = kernel_precision[1], dim=2**12, kernel_name= kernel_precision[0],
                       precision = kernel_precision[2], w_limit = kernel_precision[3], verb = verbose)

if __name__ == "__main__":
    main()

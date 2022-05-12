import pyopencl as cl
import numpy as np
from mod_opencl.opencl_class_device import OpenCL_Object, Platform_Device_OpenCL
from mod_test.test_object import Test_Class

def _limit_precision(_type):
    if isinstance(_type(0.0), (np.float32)):
        precision = np.float32(1.e-6) #-> low limit
    elif isinstance(_type(0.0),(np.float64)):
        precision = np.float64(1.e-15)
    else:
        precision = 0
    return precision


def id_test_dimension(Instance_OpenCL_Object : "OpenCL_Object",
                      _wp : "type(..,16,32,64..)" = np.int32,
                      kernel_name = "kernel_full_debug",
                      Dev_prop = None):

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
                                                  problem_shape, None,
                                                  Instance_OpenCL_Object.matrix_in_gid_device,
                                                  Instance_OpenCL_Object.matrix_in_lid_device)
    Instance_OpenCL_Object.queue.finish()
    cl.enqueue_copy( Instance_OpenCL_Object.queue,
                     Instance_OpenCL_Object.matrix_in_gid,
                     Instance_OpenCL_Object.matrix_in_gid_device)

    cl.enqueue_copy( Instance_OpenCL_Object.queue,
                     Instance_OpenCL_Object.matrix_in_lid,
                     Instance_OpenCL_Object.matrix_in_lid_device)
    print(Instance_OpenCL_Object.matrix_in_gid)
    print(Instance_OpenCL_Object.matrix_in_lid)

    Instance_OpenCL_Object.free_buffer()
    Instance_OpenCL_Object.reset_attrib()

def reduccion(Instance_OpenCL_Object, Test_obj,
              _wp : "type(..,16,32,64..)" = np.int32,
              Dev_prop = None, kernel_name = "kernel_full_debug"):

    precision = _limit_precision(_wp)
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
        Test_obj.reduction_(matrix_in, suma, verbose = True, warn_test = [], limit = precision)
        #Reset buffer out
        matrix_out = np.zeros(problem_shape, dtype=_wp)
        Instance_OpenCL_Object.buffer_global(matrix_out, "matrix_out")
    #Removes every attribute of Instance_OpenCL_Object
    Instance_OpenCL_Object.free_buffer()
    Instance_OpenCL_Object.reset_attrib()


def functions_test(Instance_OpenCL_Object, Test_obj,
                   _wp : "type(..16,32,64,...)" = np.float64,
                   Dev_prop = None, dim = 512, kernel_name = "kernel_full_debug"):

    precision = _limit_precision(_wp)

    if isinstance(_wp(0.0), (np.float64, np.float32)):
        matrix_in = np.array([[np.random.random() for _ in range(dim)] for _ in range(36)],dtype = _wp)
        matrix_in[5] = np.fabs(matrix_in[5]  ) + _wp(1.0)#arccosh >= 1
    else:
        matrix_in = np.array([[np.random.randint(0,1) for _ in range(dim)] for _ in range(36)],dtype = _wp)
        matrix_in[5] = _wp(1)
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

    Test_obj.function_test_result(Instance_OpenCL_Object.matrix_out, limit = precision, verbose = True, warn_test = [] )
    Instance_OpenCL_Object.free_buffer()
    Instance_OpenCL_Object.reset_attrib()

def main():
    #Device_Propierties -> [[extensions], max_glob_mem, max_loc_mem, [w_it,w_it,w_it]]
    First_OpenCL_Object = OpenCL_Object()
    Test_Object = Test_Class()
    Device_Propierties = First_OpenCL_Object.return_attrib()
    kernel_script = {"float":["kernel_full_debug_32", np.float32],
                     "cl_khr_fp64" : ["kernel_full_debug", np.float64] }
    for key, kernel_precision in kernel_script.items():
        #id_test_dimension(First_OpenCL_Object, Dev_prop=Device_Propierties)
        reduccion(First_OpenCL_Object, Test_Object,
                  Dev_prop=Device_Propierties, kernel_name = kernel_precision[0])
        functions_test(First_OpenCL_Object, Test_Object,
                       _wp = kernel_precision[1], dim=2**12, kernel_name= kernel_precision[0])

if __name__ == "__main__":
    main()

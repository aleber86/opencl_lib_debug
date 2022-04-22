import time
import numpy as np
import pyopencl as cl


def host_matrix(dim_row, dim_col, value_type = np.float64, debug = False):
    if debug:
        matrix = np.asarray(np.random.randint(0,2,size=(dim_row, dim_col))).astype(value_type)
    else:
        matrix = np.asarray(np.random.random(size = (dim_row,dim_col))).astype(value_type)
    return matrix

def opencl_buffer_global_mem(host_to_gpu : np.array,
                            mem_flag : cl.mem_flags,
                            ctx : cl.create_some_context,
                            write_only = True):

    #write_only = True -> Device writes on memory space
    #write_only = False -> Device only reads from memory space

    if write_only:
        WRITE = mem_flag.WRITE_ONLY
    else:
        WRITE = mem_flag.READ_ONLY

    object_host_to_device = cl.Buffer(ctx, WRITE | mem_flag.COPY_HOST_PTR, hostbuf = host_to_gpu)

    return object_host_to_device

def opencl_buffer_local_mem(size, byte_size):

    #Allocate memory space on device local memory
    #size = dim x dim x dim
    #byte_size = byte(float, int, ...)
    localbuffer = cl.LocalMemory(size*byte_size)
    return localbuffer

def kernel_compile(ctx : cl.create_some_context, file_name: str = 'kernel.cl' ):
    #Compile kernel with all programs in file

    #with open(file_name, 'r') as kernel_file:
    #    kernel_script = kernel_file.read()

    kernel_script="""

    void __kernel matrix_product3(__global double *A, __global double *B, 
        __global double *C, int d_row, int d_col){

    int gid_0 = get_global_id(0);
    int gid_1 = get_global_id(1);
    int grp_0 = get_group_id(0);
    int grp_1 = get_group_id(1);
    int gsz_0 = get_global_size(0);
    int gsz_1 = get_global_size(1);
    int lsz_0 = get_local_size(0);
    int lsz_1 = get_local_size(1);
    int lid_0 = get_local_id(0);
    int lid_1 = get_local_id(1);
    int ngp_0 = get_num_groups(0);
    int ngp_1 = get_num_groups(1);

    __local double l_a[8][8];
    __local double l_b[8][8];
    __private double p_c = 0.0;


    for(int group_1 = 0; group_1 < ngp_1; group_1++ ){


        l_a[lid_0][lid_1] = A[(lid_0 + grp_0*lsz_0)*gsz_1 + (lid_1 + group_1*lsz_1)];
        l_b[lid_0][lid_1] = B[(lid_0 + group_1*lsz_0)*gsz_1 + (lid_1 + grp_1*lsz_1)];
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i=0; i<lsz_0; i++){
            p_c += l_a[lid_0][i]*l_b[i][lid_1];
//            p_c = mad(l_a[lid_0][i],l_b[i][lid_1],p_c); 
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        }
    C[gid_0*gsz_1 + gid_1] = p_c;
}"""

    #opt = ["-cl-mad-enable", "-cl-unsafe-math-optimizations"]
    #opt=["-cl-strict-aliasing", "-cl-opt-disable"]
    opt=[]
    compiled_kernel = cl.Program(ctx, kernel_script).build(options=opt)

    return compiled_kernel

def assert_matrix_product(mat_a, mat_b, value_type = np.float64):
    mat_C = np.matmul(mat_a,mat_b)
    return mat_C


def main(row = 2**4, col = 2**4, local_size_1 = 2**3, local_size_2 = 2**3):
    d_row = np.int32(row)
    d_col = np.int32(col)

    l_size_1 = local_size_1
    l_size_2 = local_size_2
    matrix_A = host_matrix(d_row,d_col, debug = False)
    matrix_B = host_matrix(d_col,d_col, debug = False)
    matrix_C = np.zeros((d_row, d_col)).astype(np.float64)

    safe_value = np.size(matrix_A)+np.size(matrix_B)+np.size(matrix_C)
    safe_value = safe_value/1024**3
    if(safe_value>= 1.1):
        print(f"Error. Imposible asignar un buffer de {safe_value} GByte(s)", '\n')
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    time_start = time.perf_counter()
    gpu_matrix_A = opencl_buffer_global_mem(matrix_A, mf, ctx, False)
    gpu_matrix_B = opencl_buffer_global_mem(matrix_B, mf, ctx, False)
    gpu_matrix_C = opencl_buffer_global_mem(matrix_C, mf, ctx)
    kernel = kernel_compile(ctx)

 #   program = kernel.matrix_product1
 #   program = kernel.matrix_product2
    program = kernel.matrix_product3

    #   program(queue, matrix_A.shape,None, gpu_matrix_A,
    #        gpu_matrix_B, gpu_matrix_C)

    program(queue, (d_row,  d_col), (l_size_1, l_size_2), gpu_matrix_A,
            gpu_matrix_B, gpu_matrix_C, np.int32(d_row), np.int32(d_col))

    queue.finish()
    cl.enqueue_copy(queue, matrix_C, gpu_matrix_C)

    finish_time = time.perf_counter() - time_start

    gpu_matrix_A.release()
    gpu_matrix_B.release()
    gpu_matrix_C.release()


    print(matrix_A, '\n', 10*'*')
    print(matrix_B, '\n', 10*'*')
    print(matrix_C, '\n')
    print(f'GPU Time: {finish_time}', '\n')
    time_start = time.perf_counter()
    matrix_assert = assert_matrix_product(matrix_A, matrix_B)
    #matrix_assert = matrix_assert
    print(matrix_assert, '\n')
    finish_time = time.perf_counter() - time_start
    print(f"CPU Time: {finish_time}", '\n')
    result = matrix_C - matrix_assert
    print(result)
    max_pres = result.max()
    print(f"Precision: {max_pres}")


if __name__ == '__main__':
    main(2**7, 2**7, 8, 8)

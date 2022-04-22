#Test file for opencl reduction.

import numpy as np
import pyopencl as cl

#Kernel read

kernel_read="""
void __kernel reduccion(__global int *matrix, __global int *matrix_out){
  int gid_0 = get_global_id(0);
  int lid_0 = get_local_id(0);
  int gsz_0 = get_global_size(0);
  int lsz_0 = get_local_size(0);
  int sum = 0;

    for(int k=0; k<lsz_0; k++){
      sum += matrix[gid_0*lsz_0+k];
    }
barrier(CLK_GLOBAL_MEM_FENCE);

matrix_out[gid_0] = sum;

}
"""

def ocl_context():
  ctx = cl.create_some_context()
  queue = cl.CommandQueue(ctx)
  mem_flag = cl.mem_flags
  return ctx, queue, mem_flag

def buffer_global(ctx : cl.create_some_context, mf : cl.mem_flags, 
                  matrix : np.array, WRITE = True):
  
  if WRITE:
    WRITE_FLAG = mf.WRITE_ONLY
  else:
    WRITE_FLAG = mf.READ_ONLY
  
  matrix_gpu = cl.Buffer(ctx, WRITE_FLAG | mf.COPY_HOST_PTR, hostbuf = matrix)

  return matrix_gpu

def buffer_local(size : int, byte_size : int):
  local_buffer = cl.LocalMemory(size*byte_size)
  return local_buffer


def program(ctx : cl.create_some_context, kernel_script : str):
  prog = cl.Program(ctx, kernel_script).build()
  return prog

    
def reduccion(prog : cl.Program, queue : cl.CommandQueue, 
                 ctx : cl.create_some_context, mem_flag : cl.mem_flags):
  
  #Test. Reduction
  kernel = prog.reduccion
  matriz = np.array([np.random.randint(-100,100) for i in range(2**12)]).astype(np.int32)
  matriz_gpu = buffer_global(ctx, mem_flag, matriz)
  local_size = 256
  problem_shape = int(2**12/local_size)
  matriz_out = np.zeros((problem_shape)).astype(np.int32)
  matriz_out_gpu = buffer_global(ctx, mem_flag, matriz_out)
  print(np.sum(matriz))
  kernel(queue, (problem_shape,), None, matriz_gpu, matriz_out_gpu)
  queue.finish()
  cl.enqueue_copy(queue, matriz, matriz_gpu)
  suma = 0
  for index,j in enumerate(matriz):
      suma += j
  print(suma)




def main():

  ctx, queue, mem_flag = ocl_context()
  prog = program(ctx, kernel_read)
  reduccion(prog, queue, ctx, mem_flag)


if __name__ == '__main__':
  main()

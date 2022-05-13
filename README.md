# OpenCL Library for Debug. <br />


Inspired by last update of mesa-opencl-icd>20.0 for Ubuntu. <br />
Some old/legacy GPUs (HD7XXX) stopped working or give bad results with "Clover". <br />
Use this library for debug or testing. <br />



*Dependencies* <br />

-- Numpy -- <br />
-- Pyopencl -- <br />
-- OpenCL ICD -- <br />


### Excecute with `opencl_full_debug.py` for default options <br />

`-v True` or `--verbose True` print on screen absolute error (min, max). <br />
`-l True` or `--log True` print all results to files with suffix: **32bit**, **64bit**, **int**. <br />
`-sP (float)` or `--simple-precision (float)` absolute error threshold for simple precision test. Default 1.e-6. <br />
`-dP (float)` or `--double-precision (float)` absolute error threshold for double precision test. Default 1.e-15. <br />
`-w (int)` or `--warn-limit` warning limit for results expressed by order of magnitude. Default 1. Set 0 for no threshold. <br />

### Tests: <br />
*Global id test by using get_global_id() and global_id = get_group_id() get_local_size() + get_local_id()* <br />
*Reduction in global memory and last sum on host* <br />
*Test of math functions* <br />
~~Reduction in local memory and last sum on device~~ (Not implemented) <br />









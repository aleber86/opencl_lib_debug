# OpenCL Library for Debug.

Inspired by last update of mesa-opencl-icd>20.0 for Ubuntu.

Some old/legacy GPUs (HD7XXX) stopped working or give bad results with "Clover".
Use this library for debug or testing.


*Dependencies*
--Numpy--
--Pyopencl--
--OpenCL ICD--


###Excecute with `opencl_full_debug.py` for default options

`-v True` or `--verbose True` print on screen absolute error (min, max).
`-l True` or `--log True` print all results to files with suffix: **32bit**, **64bit**, **int**.
`-sP (float)` or `--simple-precision (float)` absolute error threshold for simple precision test. Default 1.e-6.
`-dP (float)` or `--double-precision (float)` absolute error threshold for double precision test. Default 1.e-15.
`-w (int)` or `--warn-limit` warning limit for results expressed by order of magnitude. Default 1. Set 0 for no threshold.

###Tests:
**Global id test by using get_global_id() and global_id = get_group_id()*get_local_size() + get_local_id()**
**Reduction in global memory and last sum on host**
**Test of math functions**
~~Reduction in local memory and last sum on device~~ (Not implemented)









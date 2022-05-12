
#pragma OPENCL EXTENSION cl_khr_fp64 : enable    // enable Float64

void __kernel reduccion(__global int *matrix, __global int *matrix_out){
    int gid_0 = get_global_id(0);
    int lsz_0 = get_local_size(0);
    int sum;

    if (gid_0%lsz_0 == 0){
        sum = 0;
        for(int k=0; k<lsz_0; k++){
            sum += matrix[gid_0+k];
    }

    matrix_out[gid_0] = sum;
    }
}

void __kernel id_test_dimension(__global int *matrix_gid, __global int *matrix_lid){
    int gid_0 = get_global_id(0);
    int gid_1 = get_global_id(1);
    int gid_2 = get_global_id(2);
    int lid_0 = get_local_id(0);
    int lid_1 = get_local_id(1);
    int lid_2 = get_local_id(2);
    int gsz_0 = get_global_size(0);
    int gsz_1 = get_global_size(1);
    int gsz_2 = get_global_size(2);
    int lsz_0 = get_local_size(0);
    int lsz_1 = get_local_size(1);
    int lsz_2 = get_local_size(2);
    int gup_0 = get_group_id(0);
    int gup_1 = get_group_id(1);
    int gup_2 = get_group_id(2);
    
    int index_0 = gup_0*lsz_0 + lid_0;
    int index_1 = gup_1*lsz_1 + lid_1;
    int index_2 = gup_2*lsz_2 + lid_2;

    matrix_gid[gid_0] = gid_0 + gid_1 + gid_2;
    matrix_lid[index_0] = index_0 + index_1 + index_2;
}


void __kernel math_functions(__global float *matrix_in, __global float *matrix_out){
    int gid_0 = get_global_id(0);
    int gsz_0 = get_global_size(0);

    matrix_out[0*gsz_0 + gid_0] += matrix_in[gsz_0*0 + gid_0];
    matrix_out[1*gsz_0 + gid_0] -= matrix_in[gsz_0*1 + gid_0];
    matrix_out[2*gsz_0 + gid_0] = matrix_in[gsz_0*2 + gid_0] * (float)(1.1);
    matrix_out[3*gsz_0 + gid_0] = matrix_in[gsz_0*3 + gid_0] / (float)(1.1);
    matrix_out[4*gsz_0 + gid_0] = acos(matrix_in[gsz_0*4 + gid_0]);
    matrix_out[5*gsz_0 + gid_0] = acosh(matrix_in[gsz_0*5 + gid_0]);
    matrix_out[6*gsz_0 + gid_0] = acospi(matrix_in[gsz_0*6 + gid_0]);
    matrix_out[7*gsz_0 + gid_0] = asin(matrix_in[gsz_0*7 + gid_0]);
    matrix_out[8*gsz_0 + gid_0] = asinh(matrix_in[gsz_0*8 + gid_0]);
    matrix_out[9*gsz_0 + gid_0] = asinpi(matrix_in[gsz_0*9 + gid_0]);
    matrix_out[10*gsz_0 + gid_0] = atan(matrix_in[gsz_0*10 + gid_0]);
    matrix_out[11*gsz_0 + gid_0] = atan2((float)(2.0), matrix_in[gsz_0*11 + gid_0]);
    matrix_out[12*gsz_0 + gid_0] = atanh(matrix_in[gsz_0*12 + gid_0]);
    matrix_out[13*gsz_0 + gid_0] = atanpi(matrix_in[gsz_0*13 + gid_0]);
    matrix_out[14*gsz_0 + gid_0] = atan2pi((float)(2.0), matrix_in[gsz_0*14 + gid_0]);
    matrix_out[15*gsz_0 + gid_0] = cbrt(matrix_in[gsz_0*15 + gid_0]);
    matrix_out[16*gsz_0 + gid_0] = ceil(matrix_in[gsz_0*16 + gid_0]);
    matrix_out[17*gsz_0 + gid_0] = copysign(matrix_in[gsz_0*17 + gid_0], (float)(-1.0));
    matrix_out[18*gsz_0 + gid_0] = cos(matrix_in[gsz_0*18 + gid_0]);
    matrix_out[19*gsz_0 + gid_0] = cosh(matrix_in[gsz_0*19 + gid_0]);
    matrix_out[20*gsz_0 + gid_0] = cospi(matrix_in[gsz_0*20 + gid_0]);
    matrix_out[21*gsz_0 + gid_0] = erfc(matrix_in[gsz_0*21 + gid_0]);
    matrix_out[22*gsz_0 + gid_0] = erf(matrix_in[gsz_0*22 + gid_0]);
    matrix_out[23*gsz_0 + gid_0] = exp(matrix_in[gsz_0*23 + gid_0]);
    matrix_out[24*gsz_0 + gid_0] = exp2(matrix_in[gsz_0*24 + gid_0]);
    matrix_out[25*gsz_0 + gid_0] = exp10(matrix_in[gsz_0*25 + gid_0]);
    matrix_out[26*gsz_0 + gid_0] = expm1(matrix_in[gsz_0*26 + gid_0]);
    matrix_out[27*gsz_0 + gid_0] = fabs(matrix_in[gsz_0*27 + gid_0]);
    matrix_out[28*gsz_0 + gid_0] = fdim(matrix_in[gsz_0*28 + gid_0], (float)(2.0));
    matrix_out[29*gsz_0 + gid_0] = fma((float)(2.0),(float)(1.1),matrix_in[gsz_0*29 + gid_0]);
    matrix_out[30*gsz_0 + gid_0] = fmax((float)(2.0),matrix_in[gsz_0*30 + gid_0]);
    matrix_out[31*gsz_0 + gid_0] = fmin((float)(2.0),matrix_in[gsz_0*31 + gid_0]);
    matrix_out[32*gsz_0 + gid_0] = fmod(matrix_in[gsz_0*32 + gid_0], (float)(2.1));
//    matrix_out[33*gsz_0 + gid_0] = fract(matrix_in[gsz_0*33 + gid_0], remanent);
    matrix_out[34*gsz_0 + gid_0] = hypot(matrix_in[gsz_0*34 + gid_0], (float)(2.1));
    matrix_out[35*gsz_0 + gid_0] = ldexp(matrix_in[gsz_0*35 + gid_0], (int)(4));

}

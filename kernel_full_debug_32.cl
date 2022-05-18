

void __kernel id_test_dimension(__global int *matrix_gid, __global int *matrix_lid){
    /*Test: Global index output*/

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
    barrier(CLK_GLOBAL_MEM_FENCE);
    matrix_lid[index_0] = index_0 + index_1 + index_2;
}

void __kernel reduccion(__global int *matrix, __global int *matrix_out){
    /*Test: Reduction using private and global mem*/

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

void __kernel reduccion_local(__global int *matrix_in,
                         __global int *matrix_sal,
                         __global int *suma_total,
                         __global int *number_sum,
                         __local int *acumm,
                         int cantidad,
                         int g_div){

    /*Test: Reduction using local mem*/
    int gid_0, lsz_0, gsz_0, gup_0, lid_0;
    int gnum_0;
    gsz_0 = get_global_size(0);
    gid_0 = get_global_id(0);
    lid_0 = get_local_id(0);
    lsz_0 = get_local_size(0);
    gup_0 = get_group_id(0);
    gnum_0 = get_num_groups(0);
    int stride, address;
    int counter, counter2;

    counter = 2;
    stride = lsz_0/counter;
    address = 1;
    if(number_sum[0]==0){
    acumm[lid_0] = matrix_in[gup_0*lsz_0 + lid_0];
    }
    else{
    acumm[lid_0] = matrix_sal[gup_0*lsz_0 + lid_0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    while(stride >= 1){
        if(lid_0 < stride){
            acumm[counter*lid_0] += acumm[counter*lid_0 + address];
            acumm[counter*lid_0 + address] = 0;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        counter = counter *2;
        stride = lsz_0/(counter);
        address = address*2;
    }
    if(lid_0 == 0){
    matrix_sal[lid_0+gup_0] = acumm[lid_0];
    }
    if(gid_0 == 0 && cantidad == 1){
        suma_total[0] = matrix_sal[0];
    }
    if(gid_0 == 0){
        number_sum[0] = number_sum[0] + 1;
    }



}

void __kernel math_functions(__global float *matrix_in, __global float *matrix_out,
        __global int *matrix_out_int){
    /*Test: Math function test*/

    int gid_0 = get_global_id(0);
    int gsz_0 = get_global_size(0);
    float ptr[1];
    int ptr2[1];

    matrix_out[0*gsz_0 + gid_0] += matrix_in[gsz_0*0 + gid_0];
    matrix_out[1*gsz_0 + gid_0] -= matrix_in[gsz_0*1 + gid_0];
    matrix_out[2*gsz_0 + gid_0] = matrix_in[gsz_0*2 + gid_0] * (1.1f);
    matrix_out[3*gsz_0 + gid_0] = matrix_in[gsz_0*3 + gid_0] / (1.1f);
    matrix_out[4*gsz_0 + gid_0] = acos(matrix_in[gsz_0*4 + gid_0]);
    matrix_out[5*gsz_0 + gid_0] = acosh(matrix_in[gsz_0*5 + gid_0]);
    matrix_out[6*gsz_0 + gid_0] = acospi(matrix_in[gsz_0*6 + gid_0]);
    matrix_out[7*gsz_0 + gid_0] = asin(matrix_in[gsz_0*7 + gid_0]);
    matrix_out[8*gsz_0 + gid_0] = asinh(matrix_in[gsz_0*8 + gid_0]);
    matrix_out[9*gsz_0 + gid_0] = asinpi(matrix_in[gsz_0*9 + gid_0]);
    matrix_out[10*gsz_0 + gid_0] = atan(matrix_in[gsz_0*10 + gid_0]);
    matrix_out[11*gsz_0 + gid_0] = atan2((2.0f), matrix_in[gsz_0*11 + gid_0]);
    matrix_out[12*gsz_0 + gid_0] = atanh(matrix_in[gsz_0*12 + gid_0]);
    matrix_out[13*gsz_0 + gid_0] = atanpi(matrix_in[gsz_0*13 + gid_0]);
    matrix_out[14*gsz_0 + gid_0] = atan2pi((2.0f), matrix_in[gsz_0*14 + gid_0]);
    matrix_out[15*gsz_0 + gid_0] = cbrt(matrix_in[gsz_0*15 + gid_0]);
    matrix_out[16*gsz_0 + gid_0] = ceil(matrix_in[gsz_0*16 + gid_0]);
    matrix_out[17*gsz_0 + gid_0] = copysign(matrix_in[gsz_0*17 + gid_0], (-1.0f));
    matrix_out[18*gsz_0 + gid_0] = cos(matrix_in[gsz_0*18 + gid_0]);
    matrix_out[19*gsz_0 + gid_0] = cosh(matrix_in[gsz_0*19 + gid_0]);
    matrix_out[20*gsz_0 + gid_0] = cospi(matrix_in[gsz_0*20 + gid_0]);
    matrix_out[21*gsz_0 + gid_0] = erf(matrix_in[gsz_0*21 + gid_0]);
    matrix_out[22*gsz_0 + gid_0] = erfc(matrix_in[gsz_0*22 + gid_0]);
    matrix_out[23*gsz_0 + gid_0] = exp(matrix_in[gsz_0*23 + gid_0]);
    matrix_out[24*gsz_0 + gid_0] = exp2(matrix_in[gsz_0*24 + gid_0]);
    matrix_out[25*gsz_0 + gid_0] = exp10(matrix_in[gsz_0*25 + gid_0]);
    matrix_out[26*gsz_0 + gid_0] = expm1(matrix_in[gsz_0*26 + gid_0]);
    matrix_out[27*gsz_0 + gid_0] = fabs(matrix_in[gsz_0*27 + gid_0]);
    matrix_out[28*gsz_0 + gid_0] = fdim(matrix_in[gsz_0*28 + gid_0], (2.0f));
    matrix_out[29*gsz_0 + gid_0] = fma((2.0f),(1.1f),matrix_in[gsz_0*29 + gid_0]);
    matrix_out[30*gsz_0 + gid_0] = fmax((2.0f),matrix_in[gsz_0*30 + gid_0]);
    matrix_out[31*gsz_0 + gid_0] = fmin((2.0f),matrix_in[gsz_0*31 + gid_0]);
    matrix_out[32*gsz_0 + gid_0] = fmod(matrix_in[gsz_0*32 + gid_0], (2.1f));
    matrix_out[33*gsz_0 + gid_0] = fract(matrix_in[gsz_0*33 + gid_0], ptr);
    matrix_out[34*gsz_0 + gid_0] = hypot(matrix_in[gsz_0*34 + gid_0], (2.1f));
    matrix_out[35*gsz_0 + gid_0] = ldexp(matrix_in[gsz_0*35 + gid_0], (int)(4));
//    matrix_out[36*gsz_0 + gid_0] = (float)(ilogb(matrix_in[gsz_0*36 + gid_0]));
    matrix_out[37*gsz_0 + gid_0] = lgamma(matrix_in[gsz_0*37 + gid_0]);
    matrix_out[38*gsz_0 + gid_0] = lgamma_r(matrix_in[gsz_0*38 + gid_0], ptr2);
    matrix_out[39*gsz_0 + gid_0] = log(matrix_in[gsz_0*39 + gid_0]);
    matrix_out[40*gsz_0 + gid_0] = log2(matrix_in[gsz_0*40 + gid_0]);
    matrix_out[41*gsz_0 + gid_0] = log10(matrix_in[gsz_0*41 + gid_0]);
    matrix_out[42*gsz_0 + gid_0] = log1p(matrix_in[gsz_0*42 + gid_0]);
    matrix_out[43*gsz_0 + gid_0] = nextafter((2.0f), matrix_in[gsz_0*43 + gid_0]);
    matrix_out[44*gsz_0 + gid_0] = pow((2.0f),matrix_in[gsz_0*44 + gid_0]);
    matrix_out[45*gsz_0 + gid_0] = pown(matrix_in[gsz_0*45 + gid_0], 2);
    matrix_out[46*gsz_0 + gid_0] = powr(matrix_in[gsz_0*46 + gid_0], 2);
    matrix_out[47*gsz_0 + gid_0] = mad((2.0f),(1.1f),matrix_in[gsz_0*47 + gid_0]);
    matrix_out[48*gsz_0 + gid_0] = modf(matrix_in[gsz_0*48 + gid_0], matrix_out + (gsz_0*49) + gid_0);
    matrix_out[50*gsz_0 + gid_0] = remainder(2.1f, matrix_in[gsz_0*50 + gid_0]);
    matrix_out[51*gsz_0 + gid_0] = remquo(15.5f, matrix_in[gsz_0*51 + gid_0], matrix_out_int+(1*gsz_0)+gid_0);
    matrix_out[52*gsz_0 + gid_0] = rint(matrix_in[gsz_0*52+gid_0]);
    matrix_out[53*gsz_0 + gid_0] = rootn(fabs(matrix_in[gsz_0*53+gid_0]), 6);
    matrix_out[54*gsz_0 + gid_0] = round(matrix_in[gsz_0*54+gid_0]);
    matrix_out[55*gsz_0 + gid_0] = rsqrt(fabs(matrix_in[gsz_0*55+gid_0]));
    matrix_out[56*gsz_0 + gid_0] = sin(matrix_in[gsz_0*56+gid_0]);
    matrix_out[57*gsz_0 + gid_0] = sincos(matrix_in[gsz_0*57+gid_0], matrix_out + (gsz_0*58) + gid_0);
    matrix_out[59*gsz_0 + gid_0] = sinh(matrix_in[gsz_0*58+gid_0]);
    matrix_out[60*gsz_0 + gid_0] = sinpi(matrix_in[gsz_0*59+gid_0]);
    matrix_out[61*gsz_0 + gid_0] = sqrt(fabs(matrix_in[gsz_0*60+gid_0]));
    matrix_out[62*gsz_0 + gid_0] = tan(matrix_in[gsz_0*61+gid_0]);
    matrix_out[63*gsz_0 + gid_0] = tanh(matrix_in[gsz_0*62+gid_0]);
    matrix_out[64*gsz_0 + gid_0] = tanpi(matrix_in[gsz_0*63+gid_0]);
    matrix_out[65*gsz_0 + gid_0] = tgamma(matrix_in[gsz_0*64+gid_0]);
    matrix_out[66*gsz_0 + gid_0] = trunc(matrix_in[gsz_0*65+gid_0]);

    matrix_out_int[0*gsz_0 + gid_0] = ilogb(matrix_in[gsz_0*36 + gid_0]);
}

void __kernel matrix_product(__global float *A, __global float *B, 
        __global float *C){
    /*Test: Square matrix product of using local mem and private mem. Not intended for 
     * real use. This algorithm is very unstable and prone to error propagation.
     */
    
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

    __local float l_a[8][8];
    __local float l_b[8][8];
    __private float p_c = 0.0f;


    for(int group_1 = 0; group_1 < ngp_1; group_1++ ){
        l_a[lid_0][lid_1] = A[(lid_0 + grp_0*lsz_0)*gsz_1 + (lid_1 + group_1*lsz_1)];
        l_b[lid_0][lid_1] = B[(lid_0 + group_1*lsz_0)*gsz_1 + (lid_1 + grp_1*lsz_1)];
        mem_fence(CLK_LOCAL_MEM_FENCE);
        for(int i=0; i<lsz_0; i++){
            p_c = fma(l_a[lid_0][i], l_b[i][lid_1], p_c);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        }
    C[gid_0*gsz_1 + gid_1] = p_c;
}



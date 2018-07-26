#ifndef _CLEAP_KERNEL_VORONOI_TRANSFORMATION_H
#define _CLEAP_KERNEL_VORONOI_TRANSFORMATION_H

template<unsigned int block_size>
__global__ void cleap_kernel_circumcenter_calculus( float4* mesh_data, GLuint* triangles){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i<edge_count ){
        // use volatile variables, this forces register use. Sometimes manual optimization achieves better performance.
        volatile int2 n = edges_n[i];
        volatile int2 a = edges_a[i];
        volatile int2 b = edges_b[i];
        volatile int2 op = edges_op[i];
        // if the t_a pair of indexes are broken
        if( (n.x != triangles[a.x] || n.y != triangles[a.y]) ){
            // then repair them.
            int t_index = trirel[ a.x/3 ];
            if( triangles[3*t_index+0] == n.x ){
                a.x = 3*t_index+0;
                triangles[3*t_index+1] == n.y ? (a.y = 3*t_index+1, op.x = 3*t_index+2) : (a.y = 3*t_index+2, op.x = 3*t_index+1);
            }
            else if( triangles[3*t_index+1] == n.x ){
                a.x = 3*t_index+1;
                triangles[3*t_index+0] == n.y ? (a.y = 3*t_index+0, op.x = 3*t_index+2) : (a.y = 3*t_index+2, op.x = 3*t_index+0);
            }
            else if( triangles[3*t_index+2] == n.x ){
                a.x = 3*t_index+2;
                triangles[3*t_index+0] == n.y ? (a.y = 3*t_index+0, op.x = 3*t_index+1) : (a.y = 3*t_index+1, op.x = 3*t_index+0);
            }
        }
        if( b.x != -1 ){
            if( (n.x != triangles[b.x] || n.y != triangles[b.y]) ){
                int t_index = trirel[ b.x/3 ];
                if( triangles[3*t_index+0] == n.x ){
                    b.x = 3*t_index+0;
                    triangles[3*t_index+1] == n.y ? (b.y = 3*t_index+1, op.y = 3*t_index+2) : (b.y = 3*t_index+2, op.y = 3*t_index+1);
                }
                else if( triangles[3*t_index+1] == n.x ){
                    b.x = 3*t_index+1;
                    triangles[3*t_index+0] == n.y ? (b.y = 3*t_index+0, op.y = 3*t_index+2) : (b.y = 3*t_index+2, op.y = 3*t_index+0);
                }
                else if( triangles[3*t_index+2] == n.x ){
                    b.x = 3*t_index+2;
                    triangles[3*t_index+0] == n.y ? (b.y = 3*t_index+0, op.y = 3*t_index+1) : (b.y = 3*t_index+1, op.y = 3*t_index+0);
                }
            }
        }
        edges_a[i] = make_int2(a.x, a.y);
        edges_b[i] = make_int2(b.x, b.y);
        edges_op[i] = make_int2(op.x, op.y);
    }
}
/*/
_device__ float cleap_d_dot_product( float3 u, float3 v ){
    return u.x*v.x + u.y*v.y + u.z*v.z;
}
__device__ float3 cleap_d_cross_product( float3 u, float3 v){
return make_float3( u.y*v.z - v.y*u.z, u.x*v.z - v.x*u.z, u.x*v.y - v.x*u.y );
}
__device__ float cleap_d_magnitude( float3 v ){
return __fsqrt_rn( powf(v.x, 2) + powf(v.y, 2) + powf(v.z,2));

/*/


#endif
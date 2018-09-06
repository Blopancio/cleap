#ifndef _CLEAP_KERNEL_VORONOI_TRANSFORMATION_H
#define _CLEAP_KERNEL_VORONOI_TRANSFORMATION_H

template<unsigned int block_size>
__global__ void cleap_kernel_circumcenter_calculus( float4* vertex_data, GLuint* triangles, float4* circumcenters, int face_count){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i<face_count ){
        float4 a, b, c;
        float3 A, B, C, u, v, r1, r2;
        float factor;


        a = vertex_data[triangles[i*3]];
        b = vertex_data[triangles[i*3+1]];
        c = vertex_data[triangles[i*3+2]];

        A = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
        B = make_float3(c.x - b.x, c.y - b.y, c.z - b.z);
        C = make_float3(a.x - c.x, a.y - c.y, a.z - c.z);

        u = cleap_d_cross_product(A,B);
        u = make_float3(u.x, -u.y, u.z);

        v = cleap_d_cross_product(C,u);
        v = make_float3(v.x, -v.y, v.z);

        r1 = make_float3(c.x + a.x, c.y + a.y, c.z + a.z);
        r1 = make_float3(r1.x/2.0, r1.y/2.0, r1.z/2.0);

        u = make_float3(u.x/2.0, u.y/2.0, u.z/2.0);

        factor = cleap_d_magnitude(u);
        factor = cleap_d_dot_product(A, B) / (8* factor*factor);

        r2 = make_float3(factor*v.x, factor*v.y, factor*v.z);

        circumcenters[i] = make_float4(r1.x + r2.x, r1.y + r2.y, r1.z + r2.z, 1.0);
    }
}

template<unsigned int block_size>
__global__ void cleap_kernel_voronoi_edges( float4* vertex_data, int2 *voronoi_edges, int2 *edges_n, int2 *edges_a, int2 *edges_b, float4* circumcenters, int edges_count, int* count){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i<edges_count ){
        if(edges_b[i].x == -1){
            voronoi_edges[i] = make_int2(i, i);
            atomicAdd(count, 1);
            //printf("External edge (tr A, tr B): %i\n", edges_a[i].x / 3);
        }
        else{
            int t_index_a = edges_a[i].x/3;
            int t_index_b = edges_b[i].x/3;
            voronoi_edges[i] = make_int2(t_index_a, t_index_b);

        }
    }
}

__global__ void cleap_kernel_external_edges( float4* vertex_data, float4* external_edges_data, int2 *external_edges, int2 *edges_n, int2 *edges_a, int2 *edges_b, float4* circumcenters, int edges_count){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i<edges_count ) {
        if (edges_b[i].x == -1) {
            printf("External edge (tr A, tr B): %i, void\n", edges_a[i].x / 3);
            float4 mid_point = make_float4((vertex_data[edges_n[i].x].x + vertex_data[edges_n[i].y].x) / 2.0,
                                           (vertex_data[edges_n[i].x].y + vertex_data[edges_n[i].y].y) / 2.0,
                                           (vertex_data[edges_n[i].x].z + vertex_data[edges_n[i].y].z) / 2.0, 1.0);

        }
    }
}


/*/TODO: Unir hacia adentro con otro color!!!!   Agregar este punto a la lista a dibujar.

/*/

#endif
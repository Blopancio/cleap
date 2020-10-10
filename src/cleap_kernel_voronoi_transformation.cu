#ifndef _CLEAP_KERNEL_VORONOI_TRANSFORMATION_H
#define _CLEAP_KERNEL_VORONOI_TRANSFORMATION_H


template<unsigned int block_size>
__global__ void cleap_kernel_circumcenter_calculus( float4* vertex_data, GLuint* triangles, float4* circumcenters, int3* vertex_edges_index, int face_count){

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
        vertex_edges_index[i].x = -1;
        vertex_edges_index[i].y = -1;
        vertex_edges_index[i].z = -1;
//        printf("Circumcenter %i: %f, %f, %f\n", i, circumcenters[i].x, circumcenters[i].y, circumcenters[i].z );
    }
}

template<unsigned int block_size>
__global__ void cleap_kernel_voronoi_edges( float4* vertex_data, float4* external_edges_data, int2 *voronoi_edges, int2 *external_edges, int2 *edges_n, int2 *edges_a, int2 *edges_b, float4* circumcenters, int *edges_reserve, int *ext_count, int edges_count, int face_count){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i<edges_count ){
        edges_reserve[i] = -1;
        edges_reserve[i + edges_count] = -1;
        external_edges_data[edges_a[i].x/3] = make_float4(0, 0, 0, 1.0);
        external_edges_data[i + face_count] = make_float4(0, 0, 0, 1.0);
        external_edges[i] = make_int2(edges_a[i].x/3 , edges_a[i].x/3);
        if(edges_b[i].x == -1){
            atomicAdd(ext_count,1);
            voronoi_edges[i] = make_int2(edges_a[i].x/3, edges_a[i].x/3);
            float4 mid_point = make_float4((vertex_data[edges_n[i].x].x + vertex_data[edges_n[i].y].x) / 2.0,
                                           (vertex_data[edges_n[i].x].y + vertex_data[edges_n[i].y].y) / 2.0,
                                           (vertex_data[edges_n[i].x].z + vertex_data[edges_n[i].y].z) / 2.0, 1.0);
            external_edges_data[edges_a[i].x/3 ] = make_float4(circumcenters[edges_a[i].x/3].x, circumcenters[edges_a[i].x/3].y, circumcenters[edges_a[i].x/3].z, 1.0);
            external_edges_data[i  + face_count] = make_float4(mid_point.x, mid_point.y, mid_point.z, mid_point.w);
            external_edges[i] = make_int2(edges_a[i].x/3 , i  + face_count);
//            printf("%i o %i External edge (tr A, tr B): %i, void\n", i, i  + face_count, edges_a[i].x / 3);
//            printf("mid_point %i: %f, %f, %f\n", i, external_edges_data[i  + face_count].x, external_edges_data[i  + face_count].y, external_edges_data[i  + face_count].z );
        }
        else{
            int t_index_a = edges_a[i].x/3;
            int t_index_b = edges_b[i].x/3;
            if (t_index_a < t_index_b) voronoi_edges[i] = make_int2(t_index_a, t_index_b);
            else voronoi_edges[i] = make_int2(t_index_b, t_index_a);
//            printf("%i Internal edge (tr A, tr B): %i, %i\n", i, voronoi_edges[i].x, voronoi_edges[i].y);

        }
//        printf("ext_data: %i --> %f %f %f,  %i --> %f %f %f\n", edges_a[i].x/3, external_edges_data[edges_a[i].x/3].x, external_edges_data[edges_a[i].x/3].y, external_edges_data[edges_a[i].x/3].z, i  + face_count ,external_edges_data[i+face_count].x, external_edges_data[i+face_count].y, external_edges_data[i+face_count].z );
//        printf("%i external %i,%i\n", i, external_edges[i].x, external_edges[i].y, external_edges_data[i+face_count].x, external_edges_data[i+face_count].y, external_edges_data[i+face_count].z);
    }
}

template<unsigned int block_size>
__global__ void cleap_kernel_voronoi_edges_index( int3 *edges_index, int2 *edges_n, int2 *circumcenters_edges_n, int *external_edges_count, int *edges_reserve, int edges_count){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i<edges_count ) {

            int temp = atomicExch(&edges_reserve[i], circumcenters_edges_n[i].x);
            int temp2 = i;
            int val;

            val = atomicExch(&((edges_index[circumcenters_edges_n[i].x]).x), temp2 );
            if(val != -1){
                val = atomicExch(&((edges_index[circumcenters_edges_n[i].x])).y, val );
                if(val != -1){
                    val = atomicExch(&((edges_index[circumcenters_edges_n[i].x])).z, val );
                }
            }

            temp = atomicExch(&edges_reserve[i], circumcenters_edges_n[i].y);
            temp2 = i;

        if(circumcenters_edges_n[i].x != circumcenters_edges_n[i].y) {
            val = atomicExch(&((edges_index[circumcenters_edges_n[i].y])).x, temp2);
            if (val != -1) {
                val = atomicExch(&((edges_index[circumcenters_edges_n[i].y]).y), val);
                if (val != -1) {
                    val = atomicExch(&((edges_index[circumcenters_edges_n[i].y]).z), val);
                }
            }
        }
//        printf("%i Edges index: %i : %i, %i, %i ; %i: %i, %i, %i\n", i,
//          circumcenters_edges_n[i].x, edges_index[circumcenters_edges_n[i].x].x, edges_index[circumcenters_edges_n[i].x].y, edges_index[circumcenters_edges_n[i].x].z,
//          circumcenters_edges_n[i].y, edges_index[circumcenters_edges_n[i].y].x, edges_index[circumcenters_edges_n[i].y].y, edges_index[circumcenters_edges_n[i].y].z);
    }

}

template<unsigned int block_size>
__global__ void cleap_kernel_voronoi_edges_next_prev( int3 *edges_index, GLuint* triangles, float4 *vertex, float4 *circumcenters, float4 *external_mid_points, int2 *edges_n, int2 *edges_b, int2 *edges_op, int2 *voronoi_edges, int2 *external_edges_index, int2 *half_edges, int *vertreservs, int2 *polygons, int edges_count, int circumcenters_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i< edges_count * 2 ) {
        int vert_delaunay = -1, external_edge = edges_b[i%edges_count].x, pivot_index, isCircumcenterInside = 0, reverse_flag = 0;
        float3 circumcenter_diff, vertex_circumcenter_diff;

        if(i < edges_count) vert_delaunay =  (int) fminf ((float) edges_n[i%edges_count].x, (float) edges_n[i%edges_count].y);
        else vert_delaunay =  (int) fmaxf((float) edges_n[i%edges_count].x, (float) edges_n[i%edges_count].y);

        float4 vertex_delaunay = vertex[vert_delaunay], circumcenter_a = circumcenters[voronoi_edges[i%edges_count].x], circumcenter_b;

//        printf("%i va de vertdel %i a %i \n", i, edges_n[i%edges_count].x, edges_n[i%edges_count].y);
        if(external_edge != -1) circumcenter_b = circumcenters[voronoi_edges[i%edges_count].y];
        else  {
            circumcenter_b = external_mid_points[external_edges_index[i%edges_count].y];

            float3 vert_a, vert_b, vert_c, diff_AC, diff_CB, diff_V1CC, diff_CCV2;


            vert_a = make_float3(vertex[edges_n[i%edges_count].x].x, vertex[edges_n[i%edges_count].x].y, vertex[edges_n[i%edges_count].x].z);
            vert_b = make_float3(vertex[edges_n[i%edges_count].y].x, vertex[edges_n[i%edges_count].y].y, vertex[edges_n[i%edges_count].y].z);
            vert_c = make_float3(vertex[triangles[edges_op[i%edges_count].x]].x, vertex[triangles[edges_op[i%edges_count].x]].y, vertex[triangles[edges_op[i%edges_count].x]].z);
            diff_AC = make_float3(vert_a.x- vert_c.x, vert_a.y- vert_c.y, vert_a.z- vert_c.z);
            diff_CB = make_float3(vert_c.x- vert_b.x, vert_c.y- vert_b.y, vert_c.z- vert_b.z);
            if(cleap_d_cross_product(diff_AC, diff_CB).z >0){
                diff_V1CC = make_float3(vert_a.x-circumcenter_a.x, vert_a.y-circumcenter_a.y, vert_a.z-circumcenter_a.z);
                diff_CCV2 = make_float3(circumcenter_a.x-vert_b.x, circumcenter_a.y-vert_b.y, circumcenter_b.z-vert_b.z);
            }
            else{
                diff_V1CC = make_float3(vert_b.x-circumcenter_a.x, vert_b.y-circumcenter_a.y, vert_b.z-circumcenter_a.z);
                diff_CCV2 = make_float3(circumcenter_a.x-vert_a.x, circumcenter_a.y-vert_a.y, circumcenter_a.z-vert_a.z);
            }
            isCircumcenterInside = cleap_d_cross_product(diff_V1CC, diff_CCV2 ).z > 0? 1: 0;
            if (!isCircumcenterInside) reverse_flag = 1;
        }

        int cc_b_index = external_edge != -1? voronoi_edges[i%edges_count].y : external_edges_index[i%edges_count].y;

        if (reverse_flag) circumcenter_diff = make_float3( circumcenter_a.x - circumcenter_b.x, circumcenter_a.y - circumcenter_b.y, circumcenter_a.z - circumcenter_b.z); // A-B
        else circumcenter_diff = make_float3( circumcenter_b.x - circumcenter_a.x, circumcenter_b.y - circumcenter_a.y, circumcenter_b.z - circumcenter_a.z);
        vertex_circumcenter_diff = make_float3( circumcenter_b.x - vertex_delaunay.x, circumcenter_b.y - vertex_delaunay.y, circumcenter_b.z - vertex_delaunay.z );

        atomicAdd(&(polygons[vert_delaunay].x),1);
        float cross = cleap_d_cross_product(circumcenter_diff, vertex_circumcenter_diff).z;
//        printf("%i cc_diff %f , %f , %f\n", i, circumcenter_diff.x, circumcenter_diff.y, circumcenter_diff.z);
//        printf("%i vertex_circumcenter_diff %f , %f , %f\n", i, vertex_circumcenter_diff.x, vertex_circumcenter_diff.y, vertex_circumcenter_diff.z);
//        printf("%i cross %f\n", i, cross);
//        printf("%i vert_delaunay %i\n", i, vert_delaunay);
        if (cross > 0) {
            pivot_index = voronoi_edges[i%edges_count].x;

            if(external_edge == -1){
                atomicExch(&(vertreservs[vert_delaunay]), i);
                atomicExch(&(polygons[vert_delaunay].y), i);

            }
        }
        else if(cross == 0) {
            pivot_index = i >= edges_count? voronoi_edges[i%edges_count].x : cc_b_index;
        }
        else pivot_index = cc_b_index;

        if(atomicExch(&(vertreservs[vert_delaunay]), i) == -1){
            atomicExch(&(polygons[vert_delaunay].y), i);
        }
        half_edges[i].x = vert_delaunay;
        int3 possible_edges = edges_index[pivot_index];
        printf("%i Aristas posibles para %i son: %i (%i,%i); %i (%i,%i); %i (%i,%i)\n", i,  pivot_index, possible_edges.x, edges_n[possible_edges.x%edges_count].x, edges_n[possible_edges.x%edges_count].y, possible_edges.y, edges_n[possible_edges.y%edges_count].x, edges_n[possible_edges.y%edges_count].y, possible_edges.z, edges_n[possible_edges.z%edges_count].x, edges_n[possible_edges.z%edges_count].y);

        if(cross < 0 && external_edge == -1) {
            half_edges[i].y = -1;
            atomicAdd(&(polygons[vert_delaunay].x),1);
        }
        else {
            if (possible_edges.x != i % edges_count &&
                (edges_n[possible_edges.x].x == vert_delaunay || edges_n[possible_edges.x].y == vert_delaunay)) {
                half_edges[i].y = vert_delaunay ==
                                (int) fmaxf((float) edges_n[possible_edges.x].x, (float) edges_n[possible_edges.x].y)  ?
                                possible_edges.x + edges_count : possible_edges.x;
            } else if (possible_edges.y != i % edges_count &&
                       (edges_n[possible_edges.y].x == vert_delaunay || edges_n[possible_edges.y].y == vert_delaunay)) {
                half_edges[i].y = vert_delaunay ==
                                (int) fmaxf((float) edges_n[possible_edges.y].x, (float) edges_n[possible_edges.y].y) ?
                                possible_edges.y + edges_count : possible_edges.y;
            } else if (possible_edges.z != i % edges_count &&
                       (edges_n[possible_edges.z].x == vert_delaunay || edges_n[possible_edges.z].y == vert_delaunay)) {
                half_edges[i].y = vert_delaunay ==
                                (int) fmaxf((float) edges_n[possible_edges.z].x, (float) edges_n[possible_edges.z].y) ?
                                possible_edges.z + edges_count : possible_edges.z;
            }
        }

//        printf("%i i/edges_count %i possible_edges.x %i edges_n[possible_edges.x].x %i edges_n[possible_edges.x].y %i vert_delaunay %i\n", i,  i % edges_count ,possible_edges.x, edges_n[possible_edges.x].x, edges_n[possible_edges.x].y, vert_delaunay);
//        printf("%i i/edges_count %i possible_edges.y %i edges_n[possible_edges.y].x %i edges_n[possible_edges.y].y %i vert_delaunay %i\n", i,  i % edges_count ,possible_edges.y, edges_n[possible_edges.y].x, edges_n[possible_edges.y].y, vert_delaunay);
//        printf("%i i/edges_count %i possible_edges.z %i edges_n[possible_edges.z].x %i edges_n[possible_edges.z].y %i vert_delaunay %i\n", i,  i % edges_count ,possible_edges.z, edges_n[possible_edges.z].x, edges_n[possible_edges.z].y, vert_delaunay);
//        printf("%i half_edges[i].x: %i ; half_edges[i].y: %i\n",  i, half_edges[i].x, half_edges[i].y);
//        printf("Edge %i: %i, %i || CircDiff: V%i - V%i || vertexDf: V%i - D%i || pivot: %i || delaunay_vertex: %i || OP: %f %f %f\n", i, edges_n[i%edges_count].x, edges_n[i%edges_count].y, voronoi_edges[i%edges_count].x, vert_delaunay, pivot_index, vert_delaunay, vertex[triangles[edges_op[i%edges_count].x]].x, vertex[triangles[edges_op[i%edges_count].x]].y, vertex[triangles[edges_op[i%edges_count].x]].z);
    }
}



#endif

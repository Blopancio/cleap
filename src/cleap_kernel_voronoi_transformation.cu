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
        /*printf("Circumcenter %i: %f, %f, %f\n", i, circumcenters[i].x, circumcenters[i].y, circumcenters[i].z );*/
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
            //printf("%i External edge (tr A, tr B): %i, void\n", i, edges_a[i].x / 3);
        }
        else{
            int t_index_a = edges_a[i].x/3;
            int t_index_b = edges_b[i].x/3;
            if (t_index_a < t_index_b) voronoi_edges[i] = make_int2(t_index_a, t_index_b);
            else voronoi_edges[i] = make_int2(t_index_b, t_index_a);
            //printf("%i Internal edge (tr A, tr B): %i, %i\n", i, voronoi_edges[i].x, voronoi_edges[i].y);

        }
    }
}

template<unsigned int block_size>
__global__ void cleap_kernel_voronoi_edges_index( int3 *edges_index, int2 *edges_n, int2 *circumcenters_edges_n, int *external_edges_count, int *edges_reserve, int edges_count){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i<edges_count ) {

        if(circumcenters_edges_n[i].x != circumcenters_edges_n[i].y) {

            int temp = atomicExch(&edges_reserve[i], circumcenters_edges_n[i].x);
            int temp2 = i;
            int val;

            //printf("%i; edges_n[i].x %i; edges_n[i].y %i\n",i, edges_n[i].x, edges_n[i].y);

            //printf("%i cc1 %i, cc2 %i\n", i, circumcenters_edges_n[i].x, circumcenters_edges_n[i].y);


            if(!(temp == -1 || temp == circumcenters_edges_n[i].x))
                temp2 = temp2 + edges_count;

            val = atomicExch(&((edges_index[circumcenters_edges_n[i].x]).x), temp2 );
            if(val != -1){
                val = atomicExch(&((edges_index[circumcenters_edges_n[i].x])).y, val );
                if(val != -1){
                    val = atomicExch(&((edges_index[circumcenters_edges_n[i].x])).z, val );
                }
            }

            temp = atomicExch(&edges_reserve[i], circumcenters_edges_n[i].y);
            temp2 = i;

            if(!(temp == -1 || temp == circumcenters_edges_n[i].y))
                temp2 = temp2 + edges_count;

            val = atomicExch(&((edges_index[circumcenters_edges_n[i].y])).x, temp2);
            if (val != -1) {
                val = atomicExch(&((edges_index[circumcenters_edges_n[i].y]).y), val);
                if (val != -1) {
                    val = atomicExch(&((edges_index[circumcenters_edges_n[i].y]).z), val);
                }
            }
        }
        else{
            //printf("%i; edges_n[i].x %i; edges_n[i].y %i\n",i, edges_n[i].x, edges_n[i].y);

            //printf("%i cc1 %i, cc2 %i\n", i, circumcenters_edges_n[i].x, circumcenters_edges_n[i].y);

            int temp = atomicExch(&edges_reserve[i], circumcenters_edges_n[i].x);
            int temp2 = 2*(edges_count-1) + 1 + i;
            int val;

            if(!(temp == -1 || temp == circumcenters_edges_n[i].x))
                temp2 = temp2 + *external_edges_count;

            val = atomicExch(&((edges_index[circumcenters_edges_n[i].x]).x), temp2 );
            if(val != -1){
                val = atomicExch(&((edges_index[circumcenters_edges_n[i].x])).y, val );
                if(val != -1){
                    val = atomicExch(&((edges_index[circumcenters_edges_n[i].x])).z, val );
                }
            }
        }
        /*printf("%i Edges index: %i : %i, %i, %i ; %i: %i, %i, %i\n", i,
         * circumcenters_edges_n[i].x, edges_index[circumcenters_edges_n[i].x].x, edges_index[circumcenters_edges_n[i].x].y, edges_index[circumcenters_edges_n[i].x].z,
         * circumcenters_edges_n[i].y, edges_index[circumcenters_edges_n[i].y].x, edges_index[circumcenters_edges_n[i].y].y, edges_index[circumcenters_edges_n[i].y].z);*/
    }

}

template<unsigned int block_size>
__global__ void cleap_kernel_voronoi_edges_next_prev( int3 *edges_index, float4 *vertex, float4 *circumcenters, float4 *external_mid_points, int2 *edges_n, int2 *edges_b, int2 *voronoi_edges, int2 *external_edges_index, int *half_edges, int2 *polygons, int edges_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i< edges_count * 2 ) {
        int vertex_pos = -1 , vert_delaunay = -1, external_edge = edges_b[i%edges_count].x;
        float3 circumcenter_diff, vertex_circumcenter_diff, pivot;


        if(i < edges_count) vert_delaunay = (int) fminf ((float) edges_n[i%edges_count].x, (float) edges_n[i%edges_count].y);
        else vert_delaunay =  (int) fmaxf((float) edges_n[i%edges_count].x, (float) edges_n[i%edges_count].y);

        float4 vertex_delaunay = vertex[vert_delaunay], circumcenter_a = circumcenters[voronoi_edges[i%edges_count].x], circumcenter_b;

        if(external_edge != -1) circumcenter_b = circumcenters[voronoi_edges[i%edges_count].y];
        else  circumcenter_b = external_mid_points[external_edges_index[i%edges_count].y];

        circumcenter_diff = make_float3( circumcenter_a.x - circumcenter_b.x, circumcenter_a.y - circumcenter_b.y, circumcenter_a.z - circumcenter_b.z); // A-B
        vertex_circumcenter_diff = make_float3( circumcenter_b.x - vertex_delaunay.x, circumcenter_b.y - vertex_delaunay.y, circumcenter_b.z - vertex_delaunay.z );
    /*
        printf("%i extMiX X: %f Y: %f Z: %f\n", i, external_mid_points[voronoi_edges[i%edges_count].x].x, external_mid_points[voronoi_edges[i%edges_count].x].y, external_mid_points[voronoi_edges[i%edges_count].x].z);
        printf("%i extMiY X: %f Y: %f Z: %f\n", i, external_mid_points[voronoi_edges[i%edges_count].y].x, external_mid_points[voronoi_edges[i%edges_count].y].y, external_mid_points[voronoi_edges[i%edges_count].y].z);
        printf("%i CCA    X: %f Y: %f Z: %f\n", i, circumcenter_a.x, circumcenter_a.y, circumcenter_a.z);
        printf("%i CCB    X: %f Y: %f Z: %f\n", i, circumcenter_b.x, circumcenter_b.y, circumcenter_b.z);
        printf("%i CCDIFF X: %f Y: %f Z: %f\n", i, circumcenter_diff.x, circumcenter_diff.y, circumcenter_diff.z);
        printf("%i VERTCC X: %f Y: %f Z: %f\n", i, vertex_circumcenter_diff.x, vertex_circumcenter_diff.y, vertex_circumcenter_diff.z);
        */
        if (cleap_d_cross_product(circumcenter_diff, vertex_circumcenter_diff).z > 0) {
            if(external_edge == -1){
                atomicExch(&(polygons[i].y), i);
                printf("%i Este debería ser next->null\n", i);
            }
            vertex_pos = edges_n[i % edges_count].y;
            pivot = make_float3(circumcenter_b.x, circumcenter_b.y, circumcenter_b.z);
        }
        else{
            if(external_edge == -1){
                atomicExch(&(polygons[i].y), i);
                printf("%i Este debería tener un next y ser inicial\n", i);
            }
            vertex_pos = edges_n[i % edges_count].x;
            pivot = make_float3(circumcenter_a.x, circumcenter_a.y, circumcenter_a.z);
            circumcenter_diff = make_float3(-circumcenter_diff.x, -circumcenter_diff.y, -circumcenter_diff.z);
        }
        atomicAdd(&(polygons[vertex_pos].x),1);
        printf("%i Vertice %i,Caras: %i, inicial: %i\n", i, vertex_pos,polygons[i].x, polygons[i].y);
        //vertex_delaunay =
        //vertex_circumcenter_diff = make_float3( pivot.x - vertex_delaunay.x, pivot.y - vertex_delaunay.y, pivot - vertex_delaunay.z );
        //printf("%i aristas: %i\n",i, voronoi_edges[i%edges_count].x);
        //printf("%i aristas: %i, %i\n", i, edges_n[i%edges_count].x, edges_n[i%edges_count].y);
        printf("Arista: %i, Vertice: %i, edges_b[i/.edges_count].x: %i, i/.edges_count %i\n", i, vert_delaunay, edges_b[i%edges_count].x, i%edges_count);
        //Tomar menor vértice para seleccionar nombre de half edge. Tomar menor índice de half edge para calcular prev
    }
}
#endif

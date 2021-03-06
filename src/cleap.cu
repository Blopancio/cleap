//////////////////////////////////////////////////////////////////////////////////
//                                                                           	//
//	cleap                                                                   //
//	A library for handling / processing / rendering 3D meshes.	        //
//                                                                           	//
//////////////////////////////////////////////////////////////////////////////////
//										//
//	Copyright © 2011 Cristobal A. Navarro.					//
//										//	
//	This file is part of cleap.						//
//	cleap is free software: you can redistribute it and/or modify		//
//	it under the terms of the GNU General Public License as published by	//
//	the Free Software Foundation, either version 3 of the License, or	//
//	(at your option) any later version.					//
//										//
//	cleap is distributed in the hope that it will be useful,		//
//	but WITHOUT ANY WARRANTY; without even the implied warranty of		//
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the	    	//
//	GNU General Public License for more details.				//
//										//
//	You should have received a copy of the GNU General Public License	//
//	along with cleap.  If not, see <http://www.gnu.org/licenses/>. 		//
//										//
//////////////////////////////////////////////////////////////////////////////////



// public headers
#include "cleap_private.h"
#include "cleap_config.h"

// important cuda runtime headers
#include <cuda.h>
#include <cuda_gl_interop.h>

// kernel headers
#include "cleap_kernel_utils.cu"
#include "cleap_kernel_normalize_normals.cu"
#include "cleap_kernel_delaunay_transformation.cu"
#include "cleap_kernel_voronoi_transformation.cu"
#include "cleap_kernel_paint_mesh.cu"

// context creation header for opengl
// linux
#include "cleap_glx_context.cu"

#include <math.h>
#include <iostream>
#include <string>
#include <sstream>
// default blocksize
int CLEAP_CUDA_BLOCKSIZE = 256;

// timer structures
struct timeval t_ini2, t_fin2, t_ini, t_fin;

// cleap author
char CLEAP_AUTHOR[] = "Cristobal A. Navarro";

// cuda textures
texture<GLuint, 1, cudaReadModeElementType> tex_triangles;
texture<int, 1, cudaReadModeElementType> tex_edges;

int cleap_mesh_is_wireframe(_cleap_mesh *m){
	return m->wireframe;
}
int cleap_mesh_is_solid(_cleap_mesh *m){
	return m->solid;
}
int cleap_mesh_is_voronoi(_cleap_mesh *m){
	if( m-> circumcenters && m->voronoi_edge ) return m->voronoi;
	return 0;
}
void cleap_mesh_set_wireframe(_cleap_mesh *m, int w){
	m->wireframe = w;
}
void cleap_mesh_set_solid(_cleap_mesh *m, int s){
	m->solid = s;
}
void cleap_mesh_set_voronoi(_cleap_mesh *m, int v){
    m->voronoi = v;
}

float cleap_get_bsphere_r(_cleap_mesh *m){

        float view_diamx = m->max_coords.x - m->min_coords.x;
        float view_diamy = m->max_coords.y - m->min_coords.y;
        float view_diamz = m->max_coords.z - m->min_coords.z;
	return 0.5f*sqrt(powf(view_diamx, 2) + powf(view_diamz, 2) + powf(view_diamy, 2));
}
float cleap_get_bsphere_x(_cleap_mesh *m){

	return	0.5f*(m->max_coords.x + m->min_coords.x);
}
float cleap_get_bsphere_y(_cleap_mesh *m){

	return	0.5f*(m->max_coords.y + m->min_coords.y);
}
float cleap_get_bsphere_z(_cleap_mesh *m){

	return	0.5f*(m->max_coords.z + m->min_coords.z);
}


CLEAP_RESULT cleap_init(){

	_cleap_print_splash();
	_cleap_init_cuda();

	return CLEAP_SUCCESS;

}

CLEAP_RESULT cleap_init_no_render(){

	//_cleap_print_splash();
	_cleap_create_glx_context();
	_cleap_init_glew();
	_cleap_init_cuda();

	return CLEAP_SUCCESS;
}

CLEAP_RESULT cleap_end(){
	_cleap_destroy_glx_context();
	return CLEAP_SUCCESS;
}

int cleap_get_vertex_count(_cleap_mesh *hm){
	return hm->vertex_count;
}

int cleap_get_edge_count(_cleap_mesh *hm){
	return hm->edge_count;
}

int cleap_get_face_count(_cleap_mesh *hm){
	return hm->face_count;
}

_cleap_mesh* cleap_load_mesh(const char* filename){

	_cleap_mesh *m = new _cleap_mesh();	// create mew mesh
	_cleap_host_load_mesh(m, filename);	// load host part
	_cleap_device_load_mesh(m);		// load device part

	return m;
}

CLEAP_RESULT cleap_paint_mesh(_cleap_mesh *m, GLfloat r, GLfloat g, GLfloat b, GLfloat a ){

	//printf("CLEAP::kernel::paint_mesh::");
	size_t bytes;
	float4 *dptr;
	int vcount = cleap_get_vertex_count(m);
	cleap_device_mesh *dm = m->dm;
	cudaGraphicsMapResources(1, &dm->vbo_c_cuda, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&dptr, &bytes, dm->vbo_c_cuda);

	dim3 dimBlock(CLEAP_CUDA_BLOCKSIZE);
	dim3 dimGrid((vcount+CLEAP_CUDA_BLOCKSIZE) / dimBlock.x);
	cudaThreadSynchronize();
	cleap_kernel_paint_mesh<<< dimGrid, dimBlock >>>(dptr, vcount, r, g, b, a);
	cudaThreadSynchronize();
	// unmap buffer object
	cudaGraphicsUnmapResources(1, &dm->vbo_c_cuda, 0);
	//printf("ok\n");

	return CLEAP_SUCCESS;
}

CLEAP_RESULT cleap_render_mesh(_cleap_mesh *m){

	if(m->status == CLEAP_SUCCESS && m->dm->status == CLEAP_SUCCESS){
	  	glEnable (GL_POLYGON_OFFSET_FILL); 	//Necesario para permitir dibujar 2 poligonos
    		glPolygonOffset (1.0, 1.0); 		//coplanares (Wireframe y poligono solido)
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m->dm->eab);

		//! position vectors
		glBindBuffer(GL_ARRAY_BUFFER, m->dm->vbo_v);
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(3,      GL_FLOAT, 4*sizeof(float), 0);
		//! normal vectors
		glBindBuffer(GL_ARRAY_BUFFER, m->dm->vbo_n);
		glEnableClientState(GL_NORMAL_ARRAY);
		glNormalPointer(        GL_FLOAT, 4*sizeof(float), 0);
		//! color vectors
		glBindBuffer(GL_ARRAY_BUFFER, m->dm->vbo_c);
		glEnableClientState(GL_COLOR_ARRAY);
		glColorPointer(4,       GL_FLOAT, 4*sizeof(float), 0);

		if (m->solid){
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glDrawElements(GL_TRIANGLES, cleap_get_face_count(m)*3, GL_UNSIGNED_INT, BUFFER_OFFSET(0));
		}
		if (m->wireframe){
			glDisableClientState(GL_COLOR_ARRAY);
			glColor3f(0.0f, 0.0f, 1.0f);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			glDrawElements(GL_TRIANGLES, cleap_get_face_count(m)*3, GL_UNSIGNED_INT, BUFFER_OFFSET(0));
		}
        if (m->voronoi) { //TESIS
            glBindBuffer(GL_ARRAY_BUFFER, m->dm->circumcenters);
            glVertexPointer(3, GL_FLOAT, 4 * sizeof(float), 0);
            glDisableClientState(GL_COLOR_ARRAY);
            glEnable(GL_PROGRAM_POINT_SIZE);
            if(m->circumcenters_draw) glPointSize(5);
            glColor3f(1.0f, 0.0f, 0.0f);
            glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
            //glDrawElements(GL_POINTS, cleap_get_face_count(m), GL_UNSIGNED_INT, BUFFER_OFFSET(0));
            glDrawArrays(GL_POINTS, 0, cleap_get_face_count(m));//Indicar numero de objetos
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindBuffer(GL_ARRAY_BUFFER, m->dm->circumcenters);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m->dm->voronoi_edge);
            glDisableClientState(GL_COLOR_ARRAY);
            glColor3f(1.0f, 0.0f, 1.0f);
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glDrawElements(GL_LINES, cleap_get_edge_count(m)*2, GL_UNSIGNED_INT, BUFFER_OFFSET(0));

            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

            glBindBuffer(GL_ARRAY_BUFFER, m->dm->external_edge_vertex);
            glVertexPointer(3, GL_FLOAT, 4 * sizeof(float), 0);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m->dm->external_edge_index);
            glDisableClientState(GL_COLOR_ARRAY);
            glColor3f(1.0f, 1.0f, 0.0f);
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glDrawElements(GL_LINES, cleap_get_edge_count(m)*2, GL_UNSIGNED_INT, BUFFER_OFFSET(0));
		}

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);
		glDisable(GL_POLYGON_OFFSET_FILL);
		glDisable(GL_BLEND);
	}
	return CLEAP_SUCCESS;
	
}

//TODO: Hay un problema de elementos en voronoi_edges cuda. Arreglar en el futuro
CLEAP_RESULT cleap_sync_mesh(_cleap_mesh *m){

	float4 *d_vbo_v, *d_vbo_n, *d_vbo_c, *d_circumcenters;
	GLuint *d_eab;
	int2 *d_voronoi, *d_external_index, *d_voronoi_polygons, *d_next_edges;
	int3 *d_vertex_edges_index;

	size_t num_bytes=0;
	int mem_size_vbo = cleap_get_vertex_count(m)*sizeof(float4);
	int mem_size_eab = 3*cleap_get_face_count(m)*sizeof(GLuint);
    int mem_size_circumcenters = cleap_get_face_count(m)*sizeof(float4);
	int mem_size_edges = sizeof(int2)*cleap_get_edge_count(m);
    int mem_size_polygons = sizeof(int2)*cleap_get_vertex_count(m);
    int mem_size_next_edges = sizeof(int2)*cleap_get_edge_count(m)*2;

	cudaGraphicsMapResources(1, &m->dm->vbo_v_cuda, 0);
	cudaGraphicsMapResources(1, &m->dm->vbo_n_cuda, 0);
	cudaGraphicsMapResources(1, &m->dm->vbo_c_cuda, 0);
	cudaGraphicsMapResources(1, &m->dm->eab_cuda, 0);
    cudaGraphicsMapResources(1, &m->dm->circumcenters_cuda, 0);
	cudaGraphicsMapResources(1, &m->dm->voronoi_edges_cuda, 0);
	cudaGraphicsMapResources(1, &m->dm->voronoi_edges_vertex_cuda, 0);
	cudaGraphicsMapResources(1, &m->dm->voronoi_polygons_cuda, 0);
	cudaGraphicsMapResources(1, &m->dm->next_edges_cuda, 0);

	cudaGraphicsResourceGetMappedPointer( (void**)&d_vbo_v, &num_bytes, m->dm->vbo_v_cuda);
	cudaGraphicsResourceGetMappedPointer( (void**)&d_vbo_n, &num_bytes, m->dm->vbo_v_cuda);
	cudaGraphicsResourceGetMappedPointer( (void**)&d_vbo_c, &num_bytes, m->dm->vbo_v_cuda);
	cudaGraphicsResourceGetMappedPointer( (void**)&d_eab, &num_bytes, m->dm->eab_cuda);
    cudaGraphicsResourceGetMappedPointer( (void**)&d_circumcenters, &num_bytes, m->dm->circumcenters_cuda);
	cudaGraphicsResourceGetMappedPointer( (void**)&d_voronoi, &num_bytes, m->dm->voronoi_edges_cuda);
	cudaGraphicsResourceGetMappedPointer( (void**)&d_vertex_edges_index, &num_bytes, m->dm->voronoi_edges_vertex_cuda);
	cudaGraphicsResourceGetMappedPointer( (void**)&d_voronoi_polygons, &num_bytes, m->dm->voronoi_polygons_cuda);
	cudaGraphicsResourceGetMappedPointer( (void**)&d_next_edges, &num_bytes, m->dm->next_edges_cuda);
    if(m->external_edge) cudaGraphicsResourceGetMappedPointer( (void**)&d_external_index, &num_bytes, m->dm->external_edges_index_cuda);

	cudaMemcpy( m->vnc_data.v, d_vbo_v, mem_size_vbo, cudaMemcpyDeviceToHost );
	cudaMemcpy( m->vnc_data.n, d_vbo_n, mem_size_vbo, cudaMemcpyDeviceToHost );
	cudaMemcpy( m->vnc_data.c, d_vbo_c, mem_size_vbo, cudaMemcpyDeviceToHost );
	cudaMemcpy( m->triangles, d_eab, mem_size_eab, cudaMemcpyDeviceToHost );
    cudaMemcpy( m->circumcenters_data, d_circumcenters, mem_size_circumcenters, cudaMemcpyDeviceToHost );
	cudaMemcpy( m->voronoi_edges_data, d_voronoi, mem_size_edges, cudaMemcpyDeviceToHost );
    cudaMemcpy( m->voronoi_edges_index_vertex, d_vertex_edges_index, mem_size_edges, cudaMemcpyDeviceToHost );
	cudaMemcpy( m->voronoi_polygons_data, d_voronoi_polygons, mem_size_polygons, cudaMemcpyDeviceToHost );
	cudaMemcpy( m->next_edges, d_next_edges, mem_size_next_edges, cudaMemcpyDeviceToHost );
    if(m->external_edge) cudaMemcpy( m->external_edges_index_data, d_external_index, mem_size_edges, cudaMemcpyDeviceToHost );

	cudaGraphicsUnmapResources(1, &m->dm->vbo_v_cuda, 0);
	cudaGraphicsUnmapResources(1, &m->dm->vbo_n_cuda, 0);
	cudaGraphicsUnmapResources(1, &m->dm->vbo_c_cuda, 0);
	cudaGraphicsUnmapResources(1, &m->dm->eab_cuda, 0);
    cudaGraphicsUnmapResources(1, &m->dm->circumcenters_cuda, 0);
	cudaGraphicsUnmapResources(1, &m->dm->voronoi_edges_cuda, 0);
	cudaGraphicsUnmapResources(1, &m->dm->voronoi_edges_vertex_cuda, 0);
	cudaGraphicsUnmapResources(1, &m->dm->voronoi_polygons_cuda, 0);
	cudaGraphicsUnmapResources(1, &m->dm->next_edges_cuda, 0);

	cudaMemcpy( m->edge_data.n, m->dm->d_edges_n, mem_size_edges, cudaMemcpyDeviceToHost );
	cudaMemcpy( m->edge_data.a, m->dm->d_edges_a, mem_size_edges, cudaMemcpyDeviceToHost );
	cudaMemcpy( m->edge_data.b, m->dm->d_edges_b, mem_size_edges, cudaMemcpyDeviceToHost );
	cudaMemcpy( m->edge_data.op, m->dm->d_edges_op, mem_size_edges, cudaMemcpyDeviceToHost );
    cudaMemcpy( m->voronoi_edges_data, m->dm->voronoi_edges, mem_size_edges, cudaMemcpyDeviceToHost );
    cudaMemcpy( m->voronoi_edges_index_vertex, m->dm->voronoi_edges_vertex_index, mem_size_edges, cudaMemcpyDeviceToHost );
	cudaMemcpy( m->voronoi_polygons_data, m->dm->voronoi_polygons, mem_size_polygons, cudaMemcpyDeviceToHost );
	cudaMemcpy( m->next_edges, m->dm->next_edges, mem_size_next_edges, cudaMemcpyDeviceToHost );
    if(m->external_edge)  cudaMemcpy( m->external_edges_index_data, m->dm->external_edges_index, mem_size_edges, cudaMemcpyDeviceToHost );

	return CLEAP_SUCCESS;

}


void cleap_voronoi_print_mesh( _cleap_mesh *m ){

	float4 *d_external_edges, *d_circumcenter;
    int2 *d_polygons, *d_external_edges_index, *d_half_edges, *d_voronoi_edges;

	float4 *h_external_edges, *h_circumcenter;
    int2 *h_polygons, *h_external_edges_index, *h_half_edges, *h_voronoi_edges;


	int *h_external_edges_count, *external_to_list_index;


    h_external_edges_count = (int*)malloc(sizeof(int));
    cudaMemcpy( h_external_edges_count, m->dm->d_extedgescount, sizeof(int), cudaMemcpyDeviceToHost );


	h_polygons = (int2*) malloc(cleap_get_vertex_count(m)*sizeof(int2));
	h_external_edges = (float4*) malloc (( cleap_get_face_count(m)+ cleap_get_edge_count(m))*sizeof(float4));
    h_external_edges_index = (int2*) malloc(cleap_get_edge_count(m)*sizeof(int2));
    h_voronoi_edges = (int2*) malloc(cleap_get_edge_count(m)*sizeof(int2));
	h_circumcenter = (float4*) malloc(cleap_get_face_count(m)*sizeof(float4));
    h_half_edges = (int2*) malloc(cleap_get_edge_count(m)*2*sizeof(int2));
    external_to_list_index = (int*) malloc(cleap_get_edge_count(m)*sizeof(int));

	size_t num_bytes=0;
    int mem_size_polygons = cleap_get_vertex_count(m)*sizeof(int2);
    int mem_size_ext_edges_v = (cleap_get_face_count(m)+ cleap_get_edge_count(m))*sizeof(float4);
    int mem_size_ext_edges_i = cleap_get_edge_count(m)*sizeof(int2);
	int mem_size_circumcenter = cleap_get_face_count(m)*sizeof(float4);
    int mem_size_half_edges = cleap_get_edge_count(m)*2*sizeof(int2);


    cudaGraphicsMapResources(1, &m->dm->voronoi_polygons_cuda, 0);
    cudaGraphicsMapResources(1, &m->dm->external_edges_vertex_cuda, 0);
    cudaGraphicsMapResources(1, &m->dm->external_edges_index_cuda, 0);
    cudaGraphicsMapResources(1, &m->dm->voronoi_edges_cuda, 0);
	cudaGraphicsMapResources(1, &m->dm->circumcenters_cuda, 0);
    cudaGraphicsMapResources(1, &m->dm->next_edges_cuda, 0);

    cudaGraphicsResourceGetMappedPointer( (void**)&d_polygons, &num_bytes, m->dm->voronoi_polygons_cuda);
    cudaGraphicsResourceGetMappedPointer( (void**)&d_external_edges, &num_bytes, m->dm->external_edges_vertex_cuda);
    cudaGraphicsResourceGetMappedPointer( (void**)&d_external_edges_index, &num_bytes, m->dm->external_edges_index_cuda);
    cudaGraphicsResourceGetMappedPointer( (void**)&d_voronoi_edges, &num_bytes, m->dm->voronoi_edges_cuda);
	cudaGraphicsResourceGetMappedPointer( (void**)&d_circumcenter, &num_bytes, m->dm->circumcenters_cuda);
    cudaGraphicsResourceGetMappedPointer( (void**)&d_half_edges, &num_bytes, m->dm->next_edges_cuda);

	cudaMemcpy( h_polygons, d_polygons, mem_size_polygons, cudaMemcpyDeviceToHost );
    cudaMemcpy( h_external_edges, d_external_edges, mem_size_ext_edges_v, cudaMemcpyDeviceToHost );
    cudaMemcpy( h_external_edges_index, d_external_edges_index, mem_size_ext_edges_i, cudaMemcpyDeviceToHost );
    cudaMemcpy( h_voronoi_edges, d_voronoi_edges, mem_size_ext_edges_i, cudaMemcpyDeviceToHost );
	cudaMemcpy( h_circumcenter, d_circumcenter, mem_size_circumcenter, cudaMemcpyDeviceToHost );
    cudaMemcpy( h_half_edges, d_half_edges, mem_size_half_edges, cudaMemcpyDeviceToHost );

    cudaGraphicsUnmapResources(1, &m->dm->voronoi_polygons_cuda, 0);
    cudaGraphicsUnmapResources(1, &m->dm->external_edges_vertex_cuda, 0);
    cudaGraphicsUnmapResources(1, &m->dm->external_edges_index_cuda, 0);
    cudaGraphicsUnmapResources(1, &m->dm->voronoi_edges_cuda, 0);
	cudaGraphicsUnmapResources(1, &m->dm->circumcenters_cuda, 0);
    cudaGraphicsUnmapResources(1, &m->dm->next_edges_cuda, 0);

	printf("HALF EDGES (%i)\n", cleap_get_edge_count(m)*2 );
	for(int i=0; i< 2*cleap_get_edge_count(m); i++){
	    printf("%i --> %i\n", i, h_half_edges[i].y);
	}
    printf("OFF\n%i %i %i\n", cleap_get_face_count(m) + h_external_edges_count[0], cleap_get_vertex_count(m), 0);

    for(int i=0; i<cleap_get_face_count(m); i++){
        printf("%f %f %f\n", h_circumcenter[i].x, h_circumcenter[i].y, h_circumcenter[i].z);
    }

    int offset = 0;
	for(int i=0; i<cleap_get_edge_count(m); i++){
		if(h_external_edges_index[i].x != h_external_edges_index[i].y){
		    external_to_list_index[i] = offset;
            offset++;
            printf("%f %f %f\n", 3*h_external_edges[i+cleap_get_face_count(m)].x, 3*h_external_edges[i+cleap_get_face_count(m)].y, 3*h_external_edges[i+cleap_get_face_count(m)].z);
		}
		else{
		    external_to_list_index[i] = -1;
        }
	}

    for( int i=0; i<cleap_get_vertex_count(m); i++ ) {
        int initial = h_polygons[i].y;
        printf("%i ", h_polygons[i].x);
        int initial_vertex = index_to_vertex(m, h_voronoi_edges, external_to_list_index, initial, 0, -1);
        int prev_vertex = initial_vertex;
        printf("%i ", initial_vertex);
        int second_vertex = index_to_vertex(m, h_voronoi_edges, external_to_list_index, initial, 1, prev_vertex);
        if( second_vertex != -1) {
            printf("%i ", second_vertex);
            prev_vertex = second_vertex;
        }
        int j = h_half_edges[initial].y;
        int next_vertex = index_to_vertex(m, h_voronoi_edges, external_to_list_index, j, 0, prev_vertex);
        printf("%i ", next_vertex);
        while(true ){
            j = h_half_edges[j].y;
            prev_vertex = next_vertex;
            next_vertex = index_to_vertex(m, h_voronoi_edges, external_to_list_index, j, 0, prev_vertex);
            if(j== initial || j == -1) break;
            printf("%i ", next_vertex);
        }
        printf("\n");
    }
}

int index_to_vertex(_cleap_mesh *m , int2 *voronoi_edges, int *offsets, int index, int is_initial, int prev_vertex){

    if(voronoi_edges[ index % cleap_get_edge_count(m) ].x ==voronoi_edges[ index % cleap_get_edge_count(m) ].y ){
        if( is_initial ){
            return voronoi_edges[index % cleap_get_edge_count(m)].x;
		}
        return cleap_get_face_count(m) + offsets[index % cleap_get_edge_count(m) ];
    }
    if( is_initial ) {
    	return -1;
    }
    return prev_vertex == voronoi_edges[index % cleap_get_edge_count(m)].x ? voronoi_edges[index % cleap_get_edge_count(m)].y : voronoi_edges[index % cleap_get_edge_count(m)].x;
}

void cleap_print_mesh( _cleap_mesh *m ){

	cleap_sync_mesh(m);
	float4 *d_vbo_v, *d_vbo_n, *d_vbo_c;
	GLuint *d_eab;
	float4 *h_vbo_v, *h_vbo_n, *h_vbo_c;
	GLuint *h_eab;

	h_vbo_v = (float4*)malloc(cleap_get_vertex_count(m)*sizeof(float4));
	h_vbo_n = (float4*)malloc(cleap_get_vertex_count(m)*sizeof(float4));
	h_vbo_c = (float4*)malloc(cleap_get_vertex_count(m)*sizeof(float4));
	h_eab = (GLuint*)malloc(3*cleap_get_face_count(m)*sizeof(GLuint));

	size_t num_bytes=0;
	int mem_size_vbo = cleap_get_vertex_count(m)*sizeof(float4);
	int mem_size_eab = 3*cleap_get_face_count(m)*sizeof(GLuint);

	cudaGraphicsMapResources(1, &m->dm->vbo_v_cuda, 0);
	cudaGraphicsMapResources(1, &m->dm->vbo_n_cuda, 0);
	cudaGraphicsMapResources(1, &m->dm->vbo_c_cuda, 0);
	cudaGraphicsMapResources(1, &m->dm->eab_cuda, 0);

	cudaGraphicsResourceGetMappedPointer( (void**)&d_vbo_v, &num_bytes, m->dm->vbo_v_cuda);
	cudaGraphicsResourceGetMappedPointer( (void**)&d_vbo_n, &num_bytes, m->dm->vbo_v_cuda);
	cudaGraphicsResourceGetMappedPointer( (void**)&d_vbo_c, &num_bytes, m->dm->vbo_v_cuda);

	cudaGraphicsResourceGetMappedPointer( (void**)&d_eab, &num_bytes, m->dm->eab_cuda);

	cudaMemcpy( h_vbo_v, d_vbo_v, mem_size_vbo, cudaMemcpyDeviceToHost );
	cudaMemcpy( h_vbo_n, d_vbo_n, mem_size_vbo, cudaMemcpyDeviceToHost );
	cudaMemcpy( h_vbo_c, d_vbo_c, mem_size_vbo, cudaMemcpyDeviceToHost );

	cudaMemcpy( h_eab, d_eab, mem_size_eab, cudaMemcpyDeviceToHost );

	cudaGraphicsUnmapResources(1, &m->dm->vbo_v_cuda, 0);
	cudaGraphicsUnmapResources(1, &m->dm->vbo_n_cuda, 0);
	cudaGraphicsUnmapResources(1, &m->dm->vbo_c_cuda, 0);
	cudaGraphicsUnmapResources(1, &m->dm->eab_cuda, 0);

	for(int i=0; i<cleap_get_vertex_count(m); i++){
		printf("mesh_data[%i] = (%f, %f, %f)  w=%f\n", i, h_vbo_v[i].x, h_vbo_v[i].y, h_vbo_v[i].z, h_vbo_v[i].w);
	}
	for(int i=0; i<cleap_get_face_count(m); i++){
		printf("T[%i] = (%i, %i, %i)\n", i, h_eab[3*i], h_eab[3*i+1], h_eab[3*i+2]);
	}

	for( int i=0; i<cleap_get_edge_count(m); i++ ){
	    printf("edge[%i]:\n", i);
	    printf("\tn = (%i, %i)\t", m->edge_data.n[i].x, m->edge_data.n[i].y);
	    printf("a = (%i, %i)\t", m->edge_data.a[i].x, m->edge_data.a[i].y);
	    printf("b = (%i, %i)\n", m->edge_data.b[i].x, m->edge_data.b[i].y);
	}
}

CLEAP_RESULT cleap_delaunay_transformation(_cleap_mesh *m, int mode){

	//printf("CLEAP::delaunay_transformation_%id::", mode);
	float4 *d_vbo_v, *d_circumcenters, *d_external_vertex;
	GLuint *d_eab;
	int2 *d_voronoi, *d_external_index, *d_voronoi_polygons, *d_next_edges;
	int3 *d_vertex_edges_index;
	size_t bytes=0;
	int *h_listo, it=0, *external_edges;
	// Map resources
    cudaGraphicsMapResources(1, &m->dm->circumcenters_cuda, 0);
    cudaGraphicsMapResources(1, &m->dm->external_edges_vertex_cuda, 0);
    cudaGraphicsMapResources(1, &m->dm->voronoi_edges_cuda, 0);
    cudaGraphicsMapResources(1, &m->dm->external_edges_index_cuda, 0);
	cudaGraphicsMapResources(1, &m->dm->voronoi_edges_vertex_cuda, 0);
	cudaGraphicsMapResources(1, &m->dm->voronoi_polygons_cuda, 0);
	cudaGraphicsMapResources(1, &m->dm->next_edges_cuda, 0);
	cudaGraphicsMapResources(1, &m->dm->vbo_v_cuda, 0);
	cudaGraphicsMapResources(1, &m->dm->eab_cuda, 0);
	cudaGraphicsResourceGetMappedPointer( (void**)&d_circumcenters, &bytes, m->dm->circumcenters_cuda);
    cudaGraphicsResourceGetMappedPointer( (void**)&d_external_vertex, &bytes, m->dm->external_edges_vertex_cuda);
    cudaGraphicsResourceGetMappedPointer( (void**)&d_vbo_v, &bytes, m->dm->vbo_v_cuda);
	cudaGraphicsResourceGetMappedPointer( (void**)&d_eab, &bytes, m->dm->eab_cuda);
    cudaGraphicsResourceGetMappedPointer( (void**)&d_voronoi, &bytes, m->dm->voronoi_edges_cuda);
    cudaGraphicsResourceGetMappedPointer( (void**)&d_external_index, &bytes, m->dm->external_edges_index_cuda);
	cudaGraphicsResourceGetMappedPointer( (void**)&d_vertex_edges_index, &bytes, m->dm->voronoi_edges_vertex_cuda);
	cudaGraphicsResourceGetMappedPointer( (void**)&d_voronoi_polygons, &bytes, m->dm->voronoi_polygons_cuda);
	cudaGraphicsResourceGetMappedPointer( (void**)&d_next_edges, &bytes, m->dm->next_edges_cuda);
	// TEXTURE
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<GLuint>();
	cudaBindTexture(0, tex_triangles, d_eab, channelDesc, cleap_get_face_count(m)*3*sizeof(GLuint));
	int block_size = CLEAP_CUDA_BLOCKSIZE;
	dim3 dimBlock(block_size);
	dim3 dimGrid((cleap_get_edge_count(m)+block_size-1) / dimBlock.x);
	dim3 dimBlockInit(block_size);
	dim3 dimGridInit((cleap_get_face_count(m)+block_size-1) / dimBlock.x);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	// if C.C is 1.2 or higher, then use zero-copy for the flag
	if( (deviceProp.major == 1 && deviceProp.minor >= 2) || (deviceProp.major >= 2) ){
		//printf("CLEAP::device::gpu::%s\n", deviceProp.name );
		//printf("CLEAP::device_property::canMapHostMemory = %i\n", deviceProp.canMapHostMemory);
		cudaHostAlloc((void **)&h_listo, sizeof(int), cudaHostAllocMapped);
        cudaHostAlloc((void **)&external_edges, sizeof(int), cudaHostAllocMapped);
		h_listo[0] = 0;
        external_edges[0] = 0;
		cudaHostGetDevicePointer((void **)&m->dm->d_listo, (void *)h_listo, 0);
        cudaHostGetDevicePointer((void **)&m->dm->d_extedgescount, (void *)external_edges, 0);
		_cleap_start_timer();
		while( !h_listo[0] ){
			h_listo[0] = 1;
			cudaThreadSynchronize();
			_cleap_init_device_dual_arrays_int(m->dm->d_trirel, m->dm->d_trireservs, cleap_get_face_count(m), -1, dimBlockInit, dimGridInit); //demora el orden de 10^-5 secs
			cudaThreadSynchronize();
			if( mode == CLEAP_MODE_2D )
				cleap_kernel_exclusion_processing_2d<256><<< dimGrid, dimBlock >>>(d_vbo_v, d_eab, m->dm->d_edges_n, m->dm->d_edges_a, m->dm->d_edges_b, m->dm->d_edges_op, cleap_get_edge_count(m), m->dm->d_listo, m->dm->d_trirel, m->dm->d_trireservs);
			else 
				cleap_kernel_exclusion_processing_3d<256><<< dimGrid, dimBlock >>>(d_vbo_v, d_eab, m->dm->d_edges_n, m->dm->d_edges_a, m->dm->d_edges_b, m->dm->d_edges_op, cleap_get_edge_count(m), m->dm->d_listo, m->dm->d_trirel, m->dm->d_trireservs);
			
			cudaThreadSynchronize();
			if( h_listo[0] ){break;}
			cleap_kernel_repair<<< dimGrid, dimBlock >>>(d_eab, m->dm->d_trirel, m->dm->d_edges_n, m->dm->d_edges_a, m->dm->d_edges_b, m->dm->d_edges_op, cleap_get_edge_count(m)); //update
			it++;
		}
	}
	// else use memcpy transfers
	else{
		//! ZERO COPY = OFF
		//printf("CLEAP::device::gpu::%s\n", deviceProp.name );
		h_listo = (int*)malloc(sizeof(int));
		external_edges = (int*)malloc(sizeof(int));
		h_listo[0] = 0;
        external_edges[0] = 0;
		cudaMalloc( (void**) &m->dm->d_listo , sizeof(int) );
        cudaMalloc( (void**) &m->dm->d_extedgescount , sizeof(int) );
		//listo es una variable que indica cuando el algoritmo ha finalizado. cuanto listo = 1 entonces todos los edges son delaunay.
		_cleap_start_timer();
		while( !h_listo[0] ){

			h_listo[0] = 1;
			cudaMemcpy( m->dm->d_listo, h_listo, sizeof(int), cudaMemcpyHostToDevice );
            cudaMemcpy( m->dm->d_extedgescount, external_edges, sizeof(int), cudaMemcpyHostToDevice );
			_cleap_init_device_dual_arrays_int(m->dm->d_trirel, m->dm->d_trireservs, cleap_get_face_count(m), -1, dimBlockInit, dimGridInit); //demora el orden de 10^-5 secs
			if( mode == CLEAP_MODE_2D )
				cleap_kernel_exclusion_processing_2d<256><<< dimGrid, dimBlock >>>(d_vbo_v, d_eab, m->dm->d_edges_n, m->dm->d_edges_a, m->dm->d_edges_b, m->dm->d_edges_op, cleap_get_edge_count(m), m->dm->d_listo, m->dm->d_trirel, m->dm->d_trireservs);
			else 
				cleap_kernel_exclusion_processing_3d<256><<< dimGrid, dimBlock >>>(d_vbo_v, d_eab, m->dm->d_edges_n, m->dm->d_edges_a, m->dm->d_edges_b, m->dm->d_edges_op, cleap_get_edge_count(m), m->dm->d_listo, m->dm->d_trirel, m->dm->d_trireservs);
			
			cudaThreadSynchronize();
			cudaMemcpy( h_listo, m->dm->d_listo, sizeof(int), cudaMemcpyDeviceToHost );
            cudaMemcpy( external_edges, m->dm->d_extedgescount, sizeof(int), cudaMemcpyDeviceToHost );
			if( h_listo[0] ){
				break;
			}
			cleap_kernel_repair<<< dimGrid, dimBlock >>>(d_eab, m->dm->d_trirel, m->dm->d_edges_n, m->dm->d_edges_a, m->dm->d_edges_b, m->dm->d_edges_op, cleap_get_edge_count(m)); //update
			it++;
		}
		cudaFree(m->dm->d_listo);
	}

	m->circumcenters = 1;
    m->voronoi_edge = 1;
    m->external_edge = 1;
    _cleap_start_timer2();

    cleap_kernel_circumcenter_calculus<256><<< dimGrid, dimBlock >>>(d_vbo_v, d_eab, d_circumcenters, d_vertex_edges_index, cleap_get_face_count(m));

    cleap_kernel_voronoi_edges<256><<< dimGrid, dimBlock >>>(d_vbo_v, d_external_vertex, d_voronoi, d_external_index, m->dm->d_edges_n, m->dm->d_edges_a, m->dm->d_edges_b, d_circumcenters, m->dm->d_trireservs, external_edges, cleap_get_edge_count(m), cleap_get_face_count(m));
    cudaThreadSynchronize();

    cleap_kernel_voronoi_edges_index<256><<< dimGrid, dimBlock >>>(d_vertex_edges_index, m->dm->d_edges_n, d_voronoi, external_edges, m->dm->d_trireservs, cleap_get_edge_count(m));
    cudaThreadSynchronize();

    _cleap_init_array_int(m->dm->d_vertreservs, cleap_get_vertex_count(m),-1);
	_cleap_init_device_dual_arrays_int2(d_voronoi_polygons, cleap_get_vertex_count(m), 0, dimBlockInit, dimGridInit);
	cleap_kernel_voronoi_edges_next_prev<256><<< dimGrid, dimBlock >>>(d_vertex_edges_index, d_eab, d_vbo_v, d_circumcenters, d_external_vertex, m->dm->d_edges_n, m->dm->d_edges_b, m->dm->d_edges_op, d_voronoi, d_external_index, d_next_edges, m->dm->d_vertreservs, d_voronoi_polygons, cleap_get_edge_count(m), cleap_get_face_count(m));

    cudaThreadSynchronize();
    printf("Voronoi: %.5g[s]\n", _cleap_stop_timer2() );
	printf("Delaunay + Voronoi: %.5g[s]\n", _cleap_stop_timer() );
    //cleap_voronoi_print_mesh(m);
    //printf("computed in %.5g[s] (%i iterations)\n", _cleap_stop_timer(), it );
	//printf("%.6f\n", _cleap_stop_timer());
	//!Unbind Texture
	cudaUnbindTexture(tex_triangles);
	// unmap buffer object
    cudaGraphicsUnmapResources(1, &m->dm->circumcenters_cuda, 0);
    cudaGraphicsUnmapResources(1, &m->dm->external_edges_index_cuda, 0);
    cudaGraphicsUnmapResources(1, &m->dm->voronoi_edges_cuda, 0);
    cudaGraphicsUnmapResources(1, &m->dm->external_edges_index_cuda, 0);
	cudaGraphicsUnmapResources(1, &m->dm->voronoi_edges_vertex_cuda, 0);
	cudaGraphicsUnmapResources(1, &m->dm->voronoi_polygons_cuda, 0);
	cudaGraphicsUnmapResources(1, &m->dm->next_edges_cuda, 0);
	cudaGraphicsUnmapResources(1, &m->dm->vbo_v_cuda, 0);
	cudaGraphicsUnmapResources(1, &m->dm->eab_cuda, 0);
	cudaFreeHost(h_listo);

	return CLEAP_SUCCESS;

}

int cleap_delaunay_transformation_interactive(_cleap_mesh *m, int mode){

    float4 *d_vbo_v, *d_circumcenters, *d_external_vertex;
	GLuint *d_eab;
	int2 *d_voronoi, *d_external_index, *d_voronoi_polygons, *d_next_edges;
	int3 *d_vertex_edges_index;
	size_t bytes=0;
	int *h_listo, it=0, *flips, *external_edges;

    cudaGraphicsMapResources(1, &m->dm->circumcenters_cuda, 0);
    cudaGraphicsMapResources(1, &m->dm->external_edges_vertex_cuda, 0);
    cudaGraphicsMapResources(1, &m->dm->voronoi_edges_cuda, 0);
    cudaGraphicsMapResources(1, &m->dm->external_edges_index_cuda, 0);
	cudaGraphicsMapResources(1, &m->dm->voronoi_edges_vertex_cuda, 0);
	cudaGraphicsMapResources(1, &m->dm->voronoi_polygons_cuda, 0);
	cudaGraphicsMapResources(1, &m->dm->next_edges_cuda, 0);
	cudaGraphicsMapResources(1, &m->dm->vbo_v_cuda, 0);
	cudaGraphicsMapResources(1, &m->dm->eab_cuda, 0);
    cudaGraphicsResourceGetMappedPointer( (void**)&d_circumcenters, &bytes, m->dm->circumcenters_cuda);
    cudaGraphicsResourceGetMappedPointer( (void**)&d_external_vertex, &bytes, m->dm->external_edges_vertex_cuda);
	cudaGraphicsResourceGetMappedPointer( (void**)&d_vbo_v, &bytes, m->dm->vbo_v_cuda);
    cudaGraphicsResourceGetMappedPointer( (void**)&d_eab, &bytes, m->dm->eab_cuda);
    cudaGraphicsResourceGetMappedPointer( (void**)&d_voronoi, &bytes, m->dm->voronoi_edges_cuda);
    cudaGraphicsResourceGetMappedPointer( (void**)&d_external_index, &bytes, m->dm->external_edges_index_cuda);
	cudaGraphicsResourceGetMappedPointer( (void**)&d_vertex_edges_index, &bytes, m->dm->voronoi_edges_vertex_cuda);
	cudaGraphicsResourceGetMappedPointer( (void**)&d_voronoi_polygons, &bytes, m->dm->voronoi_polygons_cuda);
	cudaGraphicsResourceGetMappedPointer( (void**)&d_next_edges, &bytes, m->dm->next_edges_cuda);

	// TEXTURE
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<GLuint>();
	cudaBindTexture(0, tex_triangles, d_eab, channelDesc, cleap_get_face_count(m)*3*sizeof(GLuint));
	int block_size = CLEAP_CUDA_BLOCKSIZE;
	dim3 dimBlock(block_size);
	dim3 dimGrid((cleap_get_edge_count(m)+block_size-1) / dimBlock.x);
	dim3 dimBlockInit(block_size);
	dim3 dimGridInit((cleap_get_face_count(m)+block_size-1) / dimBlock.x);
	cudaHostAlloc((void **)&h_listo, sizeof(int), cudaHostAllocMapped);
	cudaHostAlloc((void **)&flips, sizeof(int), cudaHostAllocMapped);
    cudaHostAlloc((void **)&external_edges, sizeof(int), cudaHostAllocMapped);
	h_listo[0] = 0;
	flips[0] = 0;
    external_edges[0] = 0;
	int *dflips;
	cudaHostGetDevicePointer((void **)&m->dm->d_listo, (void *)h_listo, 0);
	cudaHostGetDevicePointer((void **)&dflips, (void *)flips, 0);
    cudaHostGetDevicePointer((void **)&m->dm->d_extedgescount, (void *)external_edges, 0);
    _cleap_start_timer();
	// compute iteration
	h_listo[0] = 1;
	cudaThreadSynchronize();
	_cleap_init_device_dual_arrays_int(m->dm->d_trirel, m->dm->d_trireservs, cleap_get_face_count(m), -1, dimBlockInit, dimGridInit); //demora el orden de 10^-5 secs
	cudaThreadSynchronize();

    if( mode == CLEAP_MODE_2D )
		cleap_kernel_exclusion_processing_2d_debug<256><<< dimGrid, dimBlock >>>(d_vbo_v, d_eab, m->dm->d_edges_n, m->dm->d_edges_a, m->dm->d_edges_b, m->dm->d_edges_op, cleap_get_edge_count(m), m->dm->d_listo, m->dm->d_trirel, m->dm->d_trireservs, dflips);
	else 
		cleap_kernel_exclusion_processing_3d<256><<< dimGrid, dimBlock >>>(d_vbo_v, d_eab, m->dm->d_edges_n, m->dm->d_edges_a, m->dm->d_edges_b, m->dm->d_edges_op, cleap_get_edge_count(m), m->dm->d_listo, m->dm->d_trirel, m->dm->d_trireservs);

	cleap_kernel_repair<<< dimGrid, dimBlock >>>(d_eab, m->dm->d_trirel, m->dm->d_edges_n, m->dm->d_edges_a, m->dm->d_edges_b, m->dm->d_edges_op, cleap_get_edge_count(m));
	cudaThreadSynchronize();

    m->circumcenters = 1;
    m->voronoi_edge = 1;
    cleap_kernel_circumcenter_calculus<256><<< dimGrid, dimBlock >>>(d_vbo_v, d_eab, d_circumcenters, d_vertex_edges_index, cleap_get_face_count(m));

    cleap_kernel_voronoi_edges<256><<< dimGrid, dimBlock >>>(d_vbo_v, d_external_vertex, d_voronoi, d_external_index, m->dm->d_edges_n, m->dm->d_edges_a, m->dm->d_edges_b, d_circumcenters, m->dm->d_trireservs, external_edges, cleap_get_edge_count(m), cleap_get_face_count(m));
	cudaThreadSynchronize();

	cleap_kernel_voronoi_edges_index<256><<< dimGrid, dimBlock >>>(d_vertex_edges_index, m->dm->d_edges_n, d_voronoi, external_edges, m->dm->d_trireservs, cleap_get_edge_count(m));
    cudaThreadSynchronize();

	_cleap_init_array_int(m->dm->d_vertreservs, cleap_get_vertex_count(m),-1);
	_cleap_init_device_dual_arrays_int2(d_voronoi_polygons, cleap_get_vertex_count(m), 0, dimBlockInit, dimGridInit);
    cleap_kernel_voronoi_edges_next_prev<256><<< dimGrid, dimBlock >>>(d_vertex_edges_index, d_eab, d_vbo_v, d_circumcenters, d_external_vertex, m->dm->d_edges_n, m->dm->d_edges_b, m->dm->d_edges_op, d_voronoi, d_external_index, d_next_edges, m->dm->d_vertreservs, d_voronoi_polygons, cleap_get_edge_count(m), cleap_get_face_count(m));

    cudaThreadSynchronize();
    printf("computed in %.5g[s]\n", _cleap_stop_timer() );
    //cleap_voronoi_print_mesh(m);

    if( h_listo[0] ){
		cudaUnbindTexture(tex_triangles);
		// unmap buffer object
        cudaGraphicsUnmapResources(1, &m->dm->voronoi_edges_cuda, 0);
        cudaGraphicsUnmapResources(1, &m->dm->external_edges_index_cuda, 0);
		cudaGraphicsUnmapResources(1, &m->dm->voronoi_edges_vertex_cuda, 0);
		cudaGraphicsUnmapResources(1, &m->dm->voronoi_polygons_cuda, 0);
		cudaGraphicsUnmapResources(1, &m->dm->next_edges_cuda, 0);
		cudaGraphicsUnmapResources(1, &m->dm->vbo_v_cuda, 0);
		cudaGraphicsUnmapResources(1, &m->dm->eab_cuda, 0);
		cudaFreeHost(h_listo);
		return 0;
	}
	it++;
	//printf("CLEAP::delaunay_transformation_%id:: Iteration computed in %.5g[s]\n", mode, _cleap_stop_timer() );
	//!Unbind Texture
	cudaUnbindTexture(tex_triangles);
	// unmap buffer object
    cudaGraphicsUnmapResources(1, &m->dm->circumcenters_cuda, 0);
    cudaGraphicsUnmapResources(1, &m->dm->external_edges_vertex_cuda, 0);
    cudaGraphicsUnmapResources(1, &m->dm->voronoi_edges_cuda, 0);
    cudaGraphicsUnmapResources(1, &m->dm->external_edges_index_cuda, 0);
	cudaGraphicsUnmapResources(1, &m->dm->voronoi_edges_vertex_cuda, 0);
	cudaGraphicsUnmapResources(1, &m->dm->voronoi_polygons_cuda, 0);
	cudaGraphicsUnmapResources(1, &m->dm->next_edges_cuda, 0);
	cudaGraphicsUnmapResources(1, &m->dm->vbo_v_cuda, 0);
	cudaGraphicsUnmapResources(1, &m->dm->eab_cuda, 0);
	cudaFreeHost(h_listo);

    return *flips;

}
CLEAP_RESULT cleap_clear_mesh(_cleap_mesh *m){

	if(m->status){
		free(m->vnc_data.v);
		free(m->vnc_data.n);
		free(m->vnc_data.c);
		free(m->edge_data.n);
		free(m->edge_data.a);
		free(m->edge_data.b);
		free(m->edge_data.op);
		free(m->triangles);
		free(m->circumcenters_data);
        free(m->voronoi_edges_data);
        free(m->voronoi_polygons_data);
		free(m->next_edges);

        if(m->external_edge) {
            free(m->external_edges_vertex_data);
            free(m->external_edges_index_data);

        }

		if(m->dm->status){
			cudaFree(m->dm->d_edges_n);
			cudaFree(m->dm->d_edges_a);
			cudaFree(m->dm->d_edges_b);
			cudaFree(m->dm->d_edges_op);
			cudaFree(m->dm->voronoi_edges);
			cudaFree(m->dm->voronoi_polygons);
			cudaFree(m->dm->next_edges);

            if(m->external_edge)  {
                cudaFree(m->dm->external_edges_index);
            }

			cudaFree(m->dm->d_trirel);
			cudaFree(m->dm->d_trireservs);
            cudaFree(m->dm->d_edgesreservs);
            cudaFree(m->dm->d_vertreservs);
			cudaFree(m->dm->d_listo);
            cudaFree(m->dm->d_extedgescount);

			glDeleteBuffers(1, &m->dm->vbo_v );
			glDeleteBuffers(1, &m->dm->vbo_n );
			glDeleteBuffers(1, &m->dm->vbo_c );
			glDeleteBuffers(1, &m->dm->eab );

			// opengl method above should have deleted the arrays, this following lines are the equivalent on cuda
			float4 *d_vbo_v, *d_vbo_n, *d_vbo_c;
			GLuint *d_eab;
			size_t bytes=0;

			cudaGraphicsMapResources(1, &m->dm->vbo_v_cuda, 0);
			cudaGraphicsMapResources(1, &m->dm->vbo_n_cuda, 0);
			cudaGraphicsMapResources(1, &m->dm->vbo_c_cuda, 0);
			cudaGraphicsMapResources(1, &m->dm->eab_cuda, 0);
			cudaGraphicsResourceGetMappedPointer( (void**)&d_vbo_v, &bytes, m->dm->vbo_v_cuda);
			cudaGraphicsResourceGetMappedPointer( (void**)&d_vbo_n, &bytes, m->dm->vbo_n_cuda);
			cudaGraphicsResourceGetMappedPointer( (void**)&d_vbo_c, &bytes, m->dm->vbo_c_cuda);
			cudaGraphicsResourceGetMappedPointer( (void**)&d_eab, &bytes, m->dm->eab_cuda);

			cudaFree(d_vbo_v);
			cudaFree(d_vbo_n);
			cudaFree(d_vbo_c);
			cudaFree(d_eab);

		}
		delete m->dm;
		delete m;
	}
	//printf("CLEAP::clear_mesh::");
	//_cleap_print_gpu_mem();
	return CLEAP_SUCCESS;
}

CLEAP_RESULT cleap_save_mesh(_cleap_mesh *m, const char *filename){

	//before saving mesh, we have to sync the data from device and host
	cleap_sync_mesh(m);
	int vcount = cleap_get_vertex_count(m);
	int fcount = cleap_get_face_count(m);
	int ecount = cleap_get_edge_count(m);
	//following line is for computer with other languages.
	setlocale(LC_NUMERIC, "POSIX");
	FILE *file_descriptor = fopen(filename,"w");
	fprintf(file_descriptor,"OFF\n");
	fprintf(file_descriptor,"%d %d %d\n",vcount, fcount, ecount);
	for(int i=0; i<vcount; i++) {
		fprintf(file_descriptor,"%f %f %f\n",m->vnc_data.v[i].x,m->vnc_data.v[i].y,m->vnc_data.v[i].z);
	}
	for(int i=0; i<fcount; i++) {
		fprintf(file_descriptor,"%d %d %d %d\n", 3, m->triangles[i*3+0],m->triangles[i*3+1], m->triangles[i*3+2] );
	}
	fclose(file_descriptor);
	setlocale(LC_NUMERIC, "");
	return CLEAP_SUCCESS;
}

CLEAP_RESULT cleap_save_mesh_no_sync(_cleap_mesh *m, const char *filename){

	int vcount = cleap_get_vertex_count(m);
	int fcount = cleap_get_face_count(m);
	int ecount = cleap_get_edge_count(m);
	//following line is for computer with other languages.
	setlocale(LC_NUMERIC, "POSIX");
	FILE *file_descriptor = fopen(filename,"w");
	fprintf(file_descriptor,"OFF\n");
	fprintf(file_descriptor,"%d %d %d\n",vcount, fcount, ecount);
	for(int i=0; i<vcount; i++) {
		fprintf(file_descriptor,"%f %f %f\n",m->vnc_data.v[i].x,m->vnc_data.v[i].y,m->vnc_data.v[i].z);
	}
	for(int i=0; i<fcount; i++) {
		fprintf(file_descriptor,"%d %d %d %d\n", 3, m->triangles[i*3+0],m->triangles[i*3+1], m->triangles[i*3+2] );
	}
	fclose(file_descriptor);
	setlocale(LC_NUMERIC, "");
	return CLEAP_SUCCESS;
}

void _cleap_start_timer(){
    gettimeofday(&t_ini, NULL); //Tiempo de Inicio
}
void _cleap_start_timer2(){
    gettimeofday(&t_ini2, NULL); //Tiempo de Inicio
}
double _cleap_stop_timer(){
    gettimeofday(&t_fin, NULL); //Tiempo de Termino
    return (double)(t_fin.tv_sec + (double)t_fin.tv_usec/1000000) - (double)(t_ini.tv_sec + (double)t_ini.tv_usec/1000000);
}
double _cleap_stop_timer2(){
    gettimeofday(&t_fin2, NULL); //Tiempo de Termino
    return (double)(t_fin2.tv_sec + (double)t_fin2.tv_usec/1000000) - (double)(t_ini2.tv_sec + (double)t_ini2.tv_usec/1000000);
}

void _cleap_reset_minmax(_cleap_mesh* m){

	m->min_coords.x = FLT_MAX;
	m->min_coords.y = FLT_MAX;
	m->min_coords.z = FLT_MAX;
	m->max_coords.x = -1*FLT_MAX;
	m->max_coords.y = -1*FLT_MAX;
	m->max_coords.z = -1*FLT_MAX;
}

CLEAP_RESULT _cleap_normalize_normals(_cleap_mesh *m){

	//printf("CLEAP::kernel::normalize_normals::");
	size_t bytes;
	float4 *dptr;
	int vcount = cleap_get_vertex_count(m);
	cleap_device_mesh *dm = m->dm;
	cudaGraphicsMapResources(1, &dm->vbo_n_cuda, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&dptr, &bytes, dm->vbo_n_cuda );

	dim3 dimBlock(CLEAP_CUDA_BLOCKSIZE);
	dim3 dimGrid( (vcount+CLEAP_CUDA_BLOCKSIZE)/dimBlock.x);
	cudaThreadSynchronize();
	cleap_kernel_normalize_normals<<< dimGrid, dimBlock >>>(dptr, vcount);
	cudaThreadSynchronize();
	// unmap buffer object
	cudaGraphicsUnmapResources(1, &dm->vbo_n_cuda, 0);
	//printf("ok\n");

	return CLEAP_SUCCESS;
}


CLEAP_RESULT _cleap_device_load_mesh(_cleap_mesh* m){

    int ini = _cleap_gpu_mem();

	// CLEAP::DEVICE_LOAD:: create instance of device_mesh struct
	m->dm = new cleap_device_mesh();
	cleap_device_mesh *dmesh = m->dm;
	cudaError_t err;
	// CLEAP::DEVICE_LOAD:: get sizes of _cleap_mesh arrays, in bytes
	GLintptr size = cleap_get_vertex_count(m) *4* sizeof(float);
    GLintptr triangles_bytes_size = sizeof(GLuint)*cleap_get_face_count(m)*3;
    GLintptr next_edge_size = sizeof(int)*cleap_get_edge_count(m)*4;
    GLintptr circumcenter_size = sizeof(float)*cleap_get_face_count(m)*4;
    GLintptr voronoi_polygon_size = sizeof(int)*cleap_get_vertex_count(m)*2;
    GLintptr external_vertex_size = sizeof(float)*4*(cleap_get_face_count(m)+cleap_get_edge_count(m));

	// CLEAP::DEVICE_LOAD:: vbo vertex data
	glGenBuffers(1, &dmesh->vbo_v);
	glBindBuffer(GL_ARRAY_BUFFER, dmesh->vbo_v);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, size, m->vnc_data.v);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	err = cudaGraphicsGLRegisterBuffer(&dmesh->vbo_v_cuda, dmesh->vbo_v, cudaGraphicsMapFlagsNone);
	if( err != cudaSuccess )
		printf("CLEAP::device_load_mesh::cudaGraphicsRegisterBuffer::vbo_p:: %s\n", cudaGetErrorString(err));

	// CLEAP::DEVICE_LOAD:: vbo normal data
	glGenBuffers(1, &dmesh->vbo_n);
	glBindBuffer(GL_ARRAY_BUFFER, dmesh->vbo_n);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, size, m->vnc_data.n);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	err = cudaGraphicsGLRegisterBuffer(&dmesh->vbo_n_cuda, dmesh->vbo_n, cudaGraphicsMapFlagsNone);
	if( err != cudaSuccess )	
		printf("CLEAP::device_load_mesh::cudaGraphicsRegisterBuffer::vbo_n:: %s\n", cudaGetErrorString(err));

	// CLEAP::DEVICE_LOAD:: vbo color data
	glGenBuffers(1, &dmesh->vbo_c);
	glBindBuffer(GL_ARRAY_BUFFER, dmesh->vbo_c);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, size, m->vnc_data.c);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	err = cudaGraphicsGLRegisterBuffer(&dmesh->vbo_c_cuda, dmesh->vbo_c, cudaGraphicsMapFlagsNone);
	if( err != cudaSuccess )
		printf("CLEAP::device_load_mesh::cudaGraphicsRegisterBuffer::vbo_c:: %s\n", cudaGetErrorString(err));
	

	// CLEAP::DEVICE_LOAD:: eab data
	glGenBuffers(1, &dmesh->eab);                                                                                   // Generate buffer //index VBO
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, dmesh->eab);                                                             // Bind the element array buffer
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangles_bytes_size , 0, GL_STATIC_DRAW);
	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, triangles_bytes_size, m->triangles);                                 //llenar indices por OpenGL -- OPCION A
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	err = cudaGraphicsGLRegisterBuffer(&dmesh->eab_cuda, dmesh->eab, cudaGraphicsMapFlagsNone);
	if( err != cudaSuccess )
		printf("CLEAP::device_load_mesh::cudaGraphicsRegisterBuffer::eab:: %s\n", cudaGetErrorString(err));

    // CLEAP::DEVICE_LOAD:: circumcenters_data
    glGenBuffers(1, &dmesh->circumcenters);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,dmesh->circumcenters);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, circumcenter_size, 0, GL_STATIC_DRAW);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, circumcenter_size, m->circumcenters_data);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    err = cudaGraphicsGLRegisterBuffer(&dmesh->circumcenters_cuda, dmesh->circumcenters, cudaGraphicsMapFlagsNone);
    if( err != cudaSuccess )
        printf("CLEAP::circumcenter_calculus::cudaGraphicsRegisterBuffer::circumcenters:: %s\n", cudaGetErrorString(err));

    // CLEAP::DEVICE_LOAD:: voronoi_edges data
    glGenBuffers(1, &dmesh->voronoi_edge);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, dmesh->voronoi_edge);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangles_bytes_size , 0, GL_STATIC_DRAW);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, triangles_bytes_size, m->voronoi_edges_data);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    err = cudaGraphicsGLRegisterBuffer(&dmesh->voronoi_edges_cuda, dmesh->voronoi_edge, cudaGraphicsMapFlagsNone);
    if( err != cudaSuccess )
        printf("CLEAP::device_load_mesh::cudaGraphicsRegisterBuffer::voronoi edges:: %s\n", cudaGetErrorString(err));

    // CLEAP::DEVICE_LOAD:: external edges vertex data
    glGenBuffers(1, &dmesh->external_edge_vertex);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,dmesh->external_edge_vertex);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, external_vertex_size, 0, GL_STATIC_DRAW);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, external_vertex_size, m->external_edges_vertex_data);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    err = cudaGraphicsGLRegisterBuffer(&dmesh->external_edges_vertex_cuda, dmesh->external_edge_vertex, cudaGraphicsMapFlagsNone);
    if( err != cudaSuccess )
        printf("CLEAP::circumcenter_calculus::cudaGraphicsRegisterBuffer::external edges vertex:: %s\n", cudaGetErrorString(err));

    // CLEAP::DEVICE_LOAD:: external edges index data
    glGenBuffers(1, &dmesh->external_edge_index);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, dmesh->external_edge_index);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangles_bytes_size , 0, GL_STATIC_DRAW);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, triangles_bytes_size, m->external_edges_index_data);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    err = cudaGraphicsGLRegisterBuffer(&dmesh->external_edges_index_cuda, dmesh->external_edge_index, cudaGraphicsMapFlagsNone);
    if( err != cudaSuccess )
        printf("CLEAP::device_load_mesh::cudaGraphicsRegisterBuffer::external edges index:: %s\n", cudaGetErrorString(err));

    // CLEAP::DEVICE_LOAD:: voronoi_vertex edges data
	glGenBuffers(1, &dmesh->voronoi_edges_vertex);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, dmesh->voronoi_edges_vertex);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangles_bytes_size , 0, GL_STATIC_DRAW);
	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, triangles_bytes_size, m->voronoi_edges_index_vertex);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	err = cudaGraphicsGLRegisterBuffer(&dmesh->voronoi_edges_vertex_cuda, dmesh->voronoi_edges_vertex, cudaGraphicsMapFlagsNone);
	if( err != cudaSuccess )
		printf("CLEAP::device_load_mesh::cudaGraphicsRegisterBuffer::voronoi edges index:: %s\n", cudaGetErrorString(err));

    /*// CLEAP::DEVICE_LOAD:: voronoi_polygons initial edge data
    glGenBuffers(1, &dmesh->voronoi_polygons_gluint);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, dmesh->voronoi_polygons_gluint);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, voronoi_polygon_size , 0, GL_STATIC_DRAW);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, voronoi_polygon_size, m->voronoi_polygons_data);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    err = cudaGraphicsGLRegisterBuffer(&dmesh->voronoi_polygons_cuda, dmesh->voronoi_polygons_gluint, cudaGraphicsMapFlagsNone);
    if( err != cudaSuccess )
        printf("CLEAP::device_load_mesh::cudaGraphicsRegisterBuffer::voronoi polygons:: %s\n", cudaGetErrorString(err));

	// CLEAP::DEVICE_LOAD:: voronoi_next edge data
	glGenBuffers(1, &dmesh->next_edges_gluint);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, dmesh->next_edges_gluint);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, next_edge_size , 0, GL_STATIC_DRAW);
	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, next_edge_size, m->next_edges);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	err = cudaGraphicsGLRegisterBuffer(&dmesh->next_edges_cuda, dmesh->next_edges_gluint, cudaGraphicsMapFlagsNone);
	if( err != cudaSuccess )
		printf("CLEAP::device_load_mesh::cudaGraphicsRegisterBuffer::next edges:: %s\n", cudaGetErrorString(err));
*/

	// CLEAP::DEVICE_LOAD:: malloc mesh and aux arrays
	size_t edge_bytes_size  = sizeof(int2)* cleap_get_edge_count(m);
    size_t next_edge_bytes_size  = sizeof(int2)* cleap_get_edge_count(m) *2;
	size_t face_bytes_size = sizeof(int)*cleap_get_face_count(m);
    size_t vertex_edges_bytes_size = sizeof(int3)*cleap_get_face_count(m);
    size_t polygons_bytes_size = sizeof(int2)*cleap_get_vertex_count(m);

	cudaMalloc( (void**) &dmesh->d_edges_n , edge_bytes_size );
	cudaMalloc( (void**) &dmesh->d_edges_a , edge_bytes_size );
	cudaMalloc( (void**) &dmesh->d_edges_b , edge_bytes_size );
	cudaMalloc( (void**) &dmesh->d_edges_op , edge_bytes_size );
	cudaMalloc( (void**) &dmesh->voronoi_edges , edge_bytes_size );
    cudaMalloc( (void**) &dmesh->external_edges_index , edge_bytes_size );
    cudaMalloc( (void**) &dmesh->voronoi_edges_vertex_index , vertex_edges_bytes_size );
	cudaMalloc( (void**) &dmesh->d_trirel, face_bytes_size );
	cudaMalloc( (void**) &dmesh->d_trireservs, face_bytes_size );
	cudaMalloc( (void**) &dmesh->d_edgesreservs, edge_bytes_size );
    cudaMalloc( (void**) &dmesh->d_vertreservs, vertex_edges_bytes_size );
    //cudaMalloc( (void**) &dmesh->next_edges, next_edge_bytes_size );
    //cudaMalloc( (void**) &dmesh->voronoi_polygons, polygons_bytes_size );

	// CLEAP::DEVICE_LOAD:: memcpy mesh and aux arrays
	cudaMemcpy( dmesh->d_edges_n, m->edge_data.n , edge_bytes_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dmesh->d_edges_a, m->edge_data.a , edge_bytes_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dmesh->d_edges_b, m->edge_data.b , edge_bytes_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dmesh->d_edges_op, m->edge_data.op , edge_bytes_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dmesh->voronoi_edges, m->voronoi_edges_data , edge_bytes_size, cudaMemcpyHostToDevice );
    cudaMemcpy( dmesh->voronoi_edges_vertex_index, m->voronoi_edges_index_vertex , vertex_edges_bytes_size, cudaMemcpyHostToDevice );
    cudaMemcpy( dmesh->external_edges_index, m->external_edges_index_data , edge_bytes_size, cudaMemcpyHostToDevice );
	//cudaMemcpy( dmesh->voronoi_polygons, m->voronoi_polygons_data , polygons_bytes_size, cudaMemcpyHostToDevice );
	//cudaMemcpy( dmesh->next_edges, m->next_edges , next_edge_bytes_size, cudaMemcpyHostToDevice );

	// CLEAP::DEVICE_LOAD:: add new device mesh entry into the array of device meshes
	// CLEAP::DEVICE_LOAD:: link main mesh with device_mesh id;
	dmesh->status = CLEAP_SUCCESS;
	//printf("CLEAP::device_load_mesh::ok\n");
	//printf("\n");
	fflush(stdout);

    int fin = _cleap_gpu_mem();
    int res = fin - ini;
    printf("Memoria: %d\n", res);
	// CLEAP::DEVICE_LOAD:: paint mesh (green by default)
	cleap_paint_mesh(m, 0.0f, 1.0f, 0.0f, 1.0f );

	// CLEAP::DEVICE_LOAD:: normalize normals
	_cleap_normalize_normals(m);

	// CLEAP::DEVICE_LOAD:: print gpu memory
	//printf("CLEAP::");
	//_cleap_print_gpu_mem();


	return CLEAP_SUCCESS;
}


void _cleap_init_array_int(int* h_array, int size, int value){

	int *d_array;
	cudaMalloc( (void**) &d_array , size*sizeof(int));
	dim3 dimBlock(CLEAP_CUDA_BLOCKSIZE);
	dim3 dimGrid((size+CLEAP_CUDA_BLOCKSIZE) / dimBlock.x);
	cudaThreadSynchronize();
	cleap_kernel_init_array_int<<< dimGrid, dimBlock >>>(d_array, size, value);
	cudaThreadSynchronize();
	//copy results to host
	cudaMemcpy( h_array, d_array, size*sizeof(int), cudaMemcpyDeviceToHost );
	cudaFree(d_array);
}




void _cleap_init_device_array_int(int* d_array, int length, int value){

	dim3 dimBlock(CLEAP_CUDA_BLOCKSIZE);
	dim3 dimGrid((length+CLEAP_CUDA_BLOCKSIZE) / dimBlock.x);
	cudaThreadSynchronize();
	cleap_kernel_init_array_int<<< dimGrid, dimBlock >>>(d_array, length, value);
	cudaThreadSynchronize();
}

void _cleap_init_device_dual_arrays_int(int* d_array1, int* d_array2, int length, int value, dim3 &dimBlock, dim3 &dimGrid){
	cleap_kernel_init_device_arrays_dual<<< dimGrid, dimBlock >>>(d_array1, d_array2, length, value);
}

void _cleap_init_device_dual_arrays_int2(int2* d_array, int length, int value, dim3 &dimBlock, dim3 &dimGrid){
	cleap_kernel_init_device_arrays_dual_int2<<< dimGrid, dimBlock >>>(d_array, length, value);
}

void _cleap_print_gpu_mem(){
	size_t free=0, total=0;
	cudaMemGetInfo(&free, &total);
	printf("gpu_memory_used::%i B (%i%%)\n" , (int)((total - free)), (int)((float)(total - free)/((float)total)*100.0));
}

int _cleap_gpu_mem(){
    size_t free=0, total=0;
    cudaMemGetInfo(&free, &total);
    return (int)(total - free);
}

int _cleap_choose_best_gpu_id(){

	int num_devices, device, max_device = 0;
	cudaGetDeviceCount(&num_devices);
	if (num_devices > 1) {
	      int max_multiprocessors = 0;
	      for (device = 0; device < num_devices; device++) {
		      cudaDeviceProp properties;
		      cudaGetDeviceProperties(&properties, device);
		      if (max_multiprocessors < properties.multiProcessorCount) {
		              max_multiprocessors = properties.multiProcessorCount;
		              max_device = device;
		      }
	      }
	}
	return max_device;
}

void _cleap_print_splash(){

	printf("\n\n************************************************\n");
	printf("****************** cleap-%d.%d.%d *****************\n", CLEAP_VERSION_MAJOR, CLEAP_VERSION_MINOR, CLEAP_VERSION_PATCH);
	printf("************************************************\n");
	printf("			by %s\n\n\n", CLEAP_AUTHOR);
	fflush(stdout);
}

void _cleap_init_cuda(){

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	cudaError_t err = cudaGLSetGLDevice( _cleap_choose_best_gpu_id() );
	//printf("CLEAP::init::CudaGLSetGLDevice::%s\n", cudaGetErrorString(err));
	//printf("CLEAP::init::gpu::%s\n", deviceProp.name );
	//printf("CLEAP::init::"); _cleap_print_gpu_mem();
	//printf("\n");
}

CLEAP_RESULT _cleap_init_glew(){
	if( glewInit() != GLEW_OK ){
		printf( ">> CLEAP::Init::GLEW Cannot Init\n");
		return CLEAP_FAILURE;
	}
	return CLEAP_SUCCESS;
}


# distutils: language = c++

from .src._fcbk cimport Graph_XL, Graph_L, TermType
from libc.stdint cimport uint32_t, uint64_t, int32_t

import numpy as np
cimport numpy as np
cimport cython

# Initialize NumPy C-API (required when using np.ndarray in signatures)
np.import_array()


cdef public class GraphL[object PyObject_GraphL, type GraphL]:

    cdef Graph_L* c_graph

    def __cinit__(self, uint64_t num_tmp_vertices = 0):
        """Constructor. 
        The first argument specifies the number of vertices that the temporary data structure should start with (should be 0 if no temporary data structure is wanted).
        """
        self.c_graph = new Graph_L(num_tmp_vertices)

    def __dealloc__(self):
        """Destructor.
        """
        del self.c_graph

    def add_vertices(self, uint64_t add_num_vertices = 1):
        """Adding a given number (only argument, default is 1) of vertices to the temporary residual graph. These are given larger indices than the current vertices (added at the end of the list of vertices). Returns the number of vertices in the temporary data structure after vertices have been added.
        """
        return self.c_graph.add_vertices(add_num_vertices)

    def add_edge(self, uint32_t i, uint32_t j, int32_t cap, int32_t rev_cap):
        """Adding a pair of non-terminal residual edges between two vertices in the temporary graph. The four arguments are i, j, cap, rev_cap. The first two arguments are the indices of the vertices (using vertex index) and the latter two specify the residual capacity of the residual edges (v_i,v_j) and (v_j,v_i), respectively. This function merges residual edges if residual edges already exist between these vertices (parallel residual edges).
        """
        self.c_graph.add_edge(i, j, cap, rev_cap)

    def add_tweights(self, uint32_t i, int32_t cap_source, int32_t cap_sink):
        """Adding terminal residual edges to a given vertex in the temporary residual graph. The arguments are i: the index of the vertex (using vertex index), cap_source: the capacity of an edge from s, and cap_sink: the capacity of an edge to t. This function merges residual edges if existing residual edges are present or if both given capacities are positive (and updates the flow if this results in a path s -> v -> t$). The value (after merging) has to be representable using a signed 32-bit integer, where a positive value corresponds to an edge from s and a negative value corresponds to an edge to t.
        """
        self.c_graph.add_tweights(i, cap_source, cap_sink)

    def prepare_maxflow(self):
        """Once the temporary graph has the wanted vertices and residual edges, calling this function will pack the residual graph into the final structure. After running (and during the running of) this function, both the final and temporary graph structure is in memory. After the final data structure has been built, this function also runs 'clear_tmp_struct', removing the temporary data structure from memory. After this function has been called, you can run 'maxflow'. 
        """
        self.c_graph.init_maxflow()
        self.c_graph.clear_tmp_struct()

    def maxflow(self):
        """Computes the maximum flow of the graph. Requires that you either build a temporary data structure and finish with 'prepare_maxflow', or run 'init_final_struct' and directly build the final graph using 'init_final_vertices' and one of the 'init_final_edges'.
        """
        return self.c_graph.maxflow()

    def what_partition(self, uint32_t i, TermType default_segm = TermType.SOURCE):
        """Query the cut partition/class of a vertex v in V after running 'maxflow' in the final data structure. A class of 'false' corresponds to v in S and a class of 'true' corresponds to v in T. This function takes two arguments. These are i: the index of the vertex (should be the placement in the interleaved data structure) and default_segm (optional): the default class if v notin S or T (default is 'false').
        """
        return self.c_graph.what_partition(i, default_segm)

    def get_partition(self):
        """Corresponds to calling 'what_partition' on all vertices and getting the answer in a list of length 'n'. The placement of the answers in this list corresponds to the vertex index. It has a single optional argument specifying the default class if v notin S or T (default is 'false').
        """
        return self.c_graph.get_partition()

    def get_num_vertices(self):
        """Get the number of vertices in the residual graph (in the final data structure if it has been built, otherwise the temporary data structure).
        """
        return self.c_graph.get_num_vertices()

    def get_num_redges(self):
        """Get the number of residual edges in the residual graph (in the final data structure if it has been built, otherwise the temporary data structure).
        """
        return self.c_graph.get_num_redges()

    def get_total_size(self):
        """Get the size (in bytes) of the graph representation (in C++). This includes all major components currently used for the final and temporary data structures.
        """
        return self.c_graph.get_total_size()

    def clear_tmp_struct(self):
        """After calling 'init_maxflow', this function will delete the temporary graph (otherwise it will stay in memory).
        """
        return self.c_graph.clear_tmp_struct()

    def init_final_struct(self, np.ndarray[uint32_t, ndim=1] arr):
        """If you know the degree (number of outgoing residual edges) for all vertices in the residual graph, you can directly build the structure of the final residual graph (interleaved data structure). The only argument is a 'uint32_t' array containing the degrees of all vertices. If this array is not already a contiguous array (C-order), a copy will be made. The resulting indices of vertices in the interleaved list can be found using the degrees. This is because the first index is 0 and the difference from vertex i to i+1 is 4+2 deg[i]. Using this function avoids the need for the temporary data structure, but does require that no parallel residual edges are present (or have been merged prior to calling this function).
        """
        # Ensure contiguous (C-order); if not, make a contiguous copy
        cdef np.ndarray[uint32_t, ndim=1, mode="c"] carr = np.ascontiguousarray(arr, dtype=np.uint32)

        return self.c_graph.init_final_struct(<const uint32_t*> carr.data, carr.size)

    def init_final_redge(self, uint32_t i, uint32_t j, uint32_t head, uint32_t sister_sat, int32_t r_cap):
        """After running 'init_final_struct', this function initializes the information in a residual edge of the final residual graph. The five arguments are i: the index in the interleaved list, j: the index of this residual edge in the list of outgoing residual edges from v_i (a value of '4294967295' corresponds to placing this edge at the end of the currently added edges), head: the index of the target in the interleaved list, sister_sat: if the mirror is saturated (a value of '0') or not/has positive residual capacity (a value of '1'), and r_cap: the residual capacity of the residual edge.
        """
        return self.c_graph.init_final_redge(i, j, head, sister_sat, r_cap)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def init_final_redges(self, a, b, cap, r_cap):
        """Initialize a list of residual edges (and their corresponding mirror) into the final data structure. The four arguments are a: the list of sources, b: the list of targets, cap: the list of residual capacities a -> b, and r_cap: the list of residual capacities b -> a. This function broadcasts the shapes to match and automatically finds the dimension of the data.
        """
        
        # Broadcasted views, ensuring identical shape
        a_b, b_b, cap_b, r_cap_b = np.broadcast_arrays(a, b, cap, r_cap)
        a_b = a_b.reshape(-1)
        b_b = b_b.reshape(-1)
        cap_b = cap_b.reshape(-1)
        r_cap_b = r_cap_b.reshape(-1)

        cdef Py_ssize_t i
        for i in range(a_b.shape[0]):
            self.init_final_redge(a_b[i], 4294967295, b_b[i], r_cap_b[i]==0, cap_b[i])
            self.init_final_redge(b_b[i], 4294967295, a_b[i], cap_b[i]==0, r_cap_b[i])

    def init_final_vertex(self, uint32_t i, int32_t tr_cap):
        """After running 'init_final_struct', this function initializes the information in a vertex of the final residual graph. The two arguments are i: the index of the vertex (in the interleaved list) and tr_cap: the terminal residual capacity (a signed 32-bit integer, where a positive value corresponds to an edge from s and a negative value corresponds to an edge to t).
        """
        return self.c_graph.init_final_vertex(i, tr_cap)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def init_final_vertices(self, a, tr_cap):
        """Initialize a list of vertices into the final data structure. The two arguments are a: the list of indices of the vertices (in the interleaved list) and tr_cap: the corresponding list of terminal residual capacities (a signed 32-bit integer, where a positive value corresponds to an edge from s and a negative value corresponds to an edge to t). This function broadcasts the shapes to match and automatically finds the dimension of the data.
        """

        # Broadcasted views, ensuring identical shape
        a_b, tr_cap_b = np.broadcast_arrays(a, tr_cap)
        a_b = a_b.reshape(-1)
        tr_cap_b = tr_cap_b.reshape(-1)
        
        cdef Py_ssize_t i
        for i in range(a_b.shape[0]):
            self.init_final_vertex(a_b[i], tr_cap_b[i])

    def inspect_vertex(self, uint32_t i):
        """Get the current information of a vertex and its outgoing edges. The only argument is the index i of the vertex (the placement in the interleaved list). The function returns a array of size 5+3 deg(v_i) containing signed 64-bit integers. The first five values are (1) the degree of v_i, (2) the next active vertex from v_i or '4294967295' if the vertex is passive/free, (3) the terminal residual capacity related to v_i, (4) the internal value 'is_sink', and (5) the distance heuristic for v_i. Each outgoing residual edge (v_i, v_j) has three values corresponding to (1) the index j, (2) the internal value 'mirror_sat', and (3) the residual capacity. 
        """
        return self.c_graph.inspect_vertex(i)

    def find_vertex_with_parallel_redge(self):
        """If no temporary data structure is used, no checks are done to ensure that parallel residual edges are not present. If you want to verify that no parallel residual edges are present, this function goes through all vertices and tries to find parallel residual edges. The function returns the vertex index (not the placement in the interleaved list) of a vertex with parallel residual edges or '-1' if no such vertex is found (the graph does not have parallel edges).
        """
        return self.c_graph.find_vertex_with_parallel_redge()


cdef public class GraphXL[object PyObject_GraphXL, type GraphXL]:

    cdef Graph_XL* c_graph

    def __cinit__(self, uint64_t expected_nodes = 0):
        """Constructor. 
        The first argument specifies the number of vertices that the temporary data structure should start with (should be 0 if no temporary data structure is wanted).
        """
        self.c_graph = new Graph_XL(expected_nodes)

    def __dealloc__(self):
        """Destructor.
        """
        del self.c_graph

    def add_vertices(self, uint64_t add_num_vertices = 1):
        """Adding a given number (only argument, default is 1) of vertices to the temporary residual graph. These are given larger indices than the current vertices (added at the end of the list of vertices). Returns the number of vertices in the temporary data structure after vertices have been added.
        """
        return self.c_graph.add_vertices(add_num_vertices)

    def add_edge(self, uint64_t i, uint64_t j, int32_t cap, int32_t rev_cap):
        """Adding a pair of non-terminal residual edges between two vertices in the temporary graph. The four arguments are i, j, cap, rev_cap. The first two arguments are the indices of the vertices (using vertex index) and the latter two specify the residual capacity of the residual edges (v_i,v_j) and (v_j,v_i), respectively. This function merges residual edges if residual edges already exist between these vertices (parallel residual edges).
        """
        self.c_graph.add_edge(i, j, cap, rev_cap)

    def add_tweights(self, uint64_t i, int32_t cap_source, int32_t cap_sink):
        """Adding terminal residual edges to a given vertex in the temporary residual graph. The arguments are i: the index of the vertex (using vertex index), cap_source: the capacity of an edge from s, and cap_sink: the capacity of an edge to t. This function merges residual edges if existing residual edges are present or if both given capacities are positive (and updates the flow if this results in a path s -> v -> t$). The value (after merging) has to be representable using a signed 32-bit integer, where a positive value corresponds to an edge from s and a negative value corresponds to an edge to t.
        """
        self.c_graph.add_tweights(i, cap_source, cap_sink)

    def prepare_maxflow(self):
        """Once the temporary graph has the wanted vertices and residual edges, calling this function will pack the residual graph into the final structure. After running (and during the running of) this function, both the final and temporary graph structure is in memory. After the final data structure has been built, this function also runs 'clear_tmp_struct', removing the temporary data structure from memory. After this function has been called, you can run 'maxflow'. 
        """
        self.c_graph.init_maxflow()
        self.c_graph.clear_tmp_struct()

    def maxflow(self):
        """Computes the maximum flow of the graph. Requires that you either build a temporary data structure and finish with 'prepare_maxflow', or run 'init_final_struct' and directly build the final graph using 'init_final_vertices' and one of the 'init_final_edges'.
        """
        return self.c_graph.maxflow()

    def what_partition(self, uint64_t i, TermType default_segm = TermType.SOURCE):
        """Query the cut partition/class of a vertex v in V after running 'maxflow' in the final data structure. A class of 'false' corresponds to v in S and a class of 'true' corresponds to v in T. This function takes two arguments. These are i: the index of the vertex (should be the placement in the interleaved data structure) and default_segm (optional): the default class if v notin S or T (default is 'false').
        """
        return self.c_graph.what_partition(i, default_segm)

    def get_partition(self):
        """Corresponds to calling 'what_partition' on all vertices and getting the answer in a list of length 'n'. The placement of the answers in this list corresponds to the vertex index. It has a single optional argument specifying the default class if v notin S or T (default is 'false').
        """
        return self.c_graph.get_partition()

    def get_num_vertices(self):
        """Get the number of vertices in the residual graph (in the final data structure if it has been built, otherwise the temporary data structure).
        """
        return self.c_graph.get_num_vertices()

    def get_num_redges(self):
        """Get the number of residual edges in the residual graph (in the final data structure if it has been built, otherwise the temporary data structure).
        """
        return self.c_graph.get_num_redges()

    def get_total_size(self):
        """Get the size (in bytes) of the graph representation (in C++). This includes all major components currently used for the final and temporary data structures.
        """
        return self.c_graph.get_total_size()

    def clear_tmp_struct(self):
        """After calling 'init_maxflow', this function will delete the temporary graph (otherwise it will stay in memory).
        """
        return self.c_graph.clear_tmp_struct()

    def init_final_struct(self, np.ndarray[uint32_t, ndim=1] arr):
        """If you know the degree (number of outgoing residual edges) for all vertices in the residual graph, you can directly build the structure of the final residual graph (interleaved data structure). The only argument is a 'uint32_t' array containing the degrees of all vertices. If this array is not already a contiguous array (C-order), a copy will be made. The resulting indices of vertices in the interleaved list can be found using the degrees. This is because the first index is 0 and the difference from vertex i to i+1 is 4+2 deg[i]. Using this function avoids the need for the temporary data structure, but does require that no parallel residual edges are present (or have been merged prior to calling this function).
        """
        # Ensure contiguous (C-order); if not, make a contiguous copy
        cdef np.ndarray[uint32_t, ndim=1, mode="c"] carr = np.ascontiguousarray(arr, dtype=np.uint32)

        return self.c_graph.init_final_struct(<const uint32_t*> carr.data, carr.size)
    
    def init_final_redge(self, uint64_t i, uint32_t j, uint64_t head, uint32_t sister_sat, int32_t r_cap):
        """After running 'init_final_struct', this function initializes the information in a residual edge of the final residual graph. The five arguments are i: the index in the interleaved list, j: the index of this residual edge in the list of outgoing residual edges from v_i (a value of '4294967295' corresponds to placing this edge at the end of the currently added edges), head: the index of the target in the interleaved list, sister_sat: if the mirror is saturated (a value of '0') or not/has positive residual capacity (a value of '1'), and r_cap: the residual capacity of the residual edge.
        """
        return self.c_graph.init_final_redge(i, j, head, sister_sat, r_cap)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def init_final_redges(self, a, b, cap, r_cap):
        """Initialize a list of residual edges (and their corresponding mirror) into the final data structure. The four arguments are a: the list of sources, b: the list of targets, cap: the list of residual capacities a -> b, and r_cap: the list of residual capacities b -> a. This function broadcasts the shapes to match and automatically finds the dimension of the data.
        """
        
        # Broadcasted views, ensuring identical shape
        a_b, b_b, cap_b, r_cap_b = np.broadcast_arrays(a, b, cap, r_cap)
        a_b = a_b.reshape(-1)
        b_b = b_b.reshape(-1)
        cap_b = cap_b.reshape(-1)
        r_cap_b = r_cap_b.reshape(-1)

        cdef Py_ssize_t i
        for i in range(a_b.shape[0]):
            self.init_final_redge(a_b[i], 4294967295, b_b[i], r_cap_b[i]==0, cap_b[i])
            self.init_final_redge(b_b[i], 4294967295, a_b[i], cap_b[i]==0, r_cap_b[i])

    def init_final_vertex(self, uint64_t i, int32_t tr_cap):
        """After running 'init_final_struct', this function initializes the information in a vertex of the final residual graph. The two arguments are i: the index of the vertex (in the interleaved list) and tr_cap: the terminal residual capacity (a signed 32-bit integer, where a positive value corresponds to an edge from s and a negative value corresponds to an edge to t).
        """
        return self.c_graph.init_final_vertex(i, tr_cap)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def init_final_vertices(self, a, tr_cap):
        """Initialize a list of vertices into the final data structure. The two arguments are a: the list of indices of the vertices (in the interleaved list) and tr_cap: the corresponding list of terminal residual capacities (a signed 32-bit integer, where a positive value corresponds to an edge from s and a negative value corresponds to an edge to t). This function broadcasts the shapes to match and automatically finds the dimension of the data.
        """

        # Broadcasted views, ensuring identical shape
        a_b, tr_cap_b = np.broadcast_arrays(a, tr_cap)
        a_b = a_b.reshape(-1)
        tr_cap_b = tr_cap_b.reshape(-1)
        
        cdef uint64_t i
        for i in range(a_b.shape[0]):
            self.init_final_vertex(a_b[i], tr_cap_b[i])

    def inspect_vertex(self, uint64_t intblockidx):
        """Get the current information of a vertex and its outgoing edges. The only argument is the index i of the vertex (the placement in the interleaved list). The function returns a array of size 5+3 deg(v_i) containing signed 64-bit integers. The first five values are (1) the degree of v_i, (2) the next active vertex from v_i or '4294967295' if the vertex is passive/free, (3) the terminal residual capacity related to v_i, (4) the internal value 'is_sink', and (5) the distance heuristic for v_i. Each outgoing residual edge (v_i, v_j) has three values corresponding to (1) the index j, (2) the internal value 'mirror_sat', and (3) the residual capacity. 
        """
        return self.c_graph.inspect_vertex(intblockidx)

    def find_vertex_with_parallel_redge(self):
        """If no temporary data structure is used, no checks are done to ensure that parallel residual edges are not present. If you want to verify that no parallel residual edges are present, this function goes through all vertices and tries to find parallel residual edges. The function returns the vertex index (not the placement in the interleaved list) of a vertex with parallel residual edges or '-1' if no such vertex is found (the graph does not have parallel edges).
        """
        return self.c_graph.find_vertex_with_parallel_redge()
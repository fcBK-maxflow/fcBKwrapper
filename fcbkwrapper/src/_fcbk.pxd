# _fcbk.pxd
# distutils: language = c++

from libcpp cimport bool
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t


cdef extern from "fcbk/fcbk.hpp" namespace "fcbk":
    cdef enum TermType:
        SOURCE = 0
        SINK = 1

    # Graph class for large datasets using orphan path flags
    cdef cppclass Graph_L "fcbk::Graph<uint32_t, true>":
        Graph_L(uint64_t num_tmp_vertices) except +
        uint64_t add_vertices(uint64_t add_num_vertices)
        void add_edge(uint32_t i, uint32_t j, int32_t cap, int32_t rev_cap)
        void add_tweights(uint32_t i, int32_t cap_source, int32_t cap_sink)
        void init_maxflow()
        int64_t maxflow()
        TermType what_partition(uint32_t i, TermType default_segment)
        uint64_t get_num_vertices()
        uint64_t get_num_redges()
        vector[bool] get_partition()
        uint64_t get_total_size()
        void clear_tmp_struct()
        uint32_t init_final_struct(const uint32_t* degree_seq, uint32_t n)
        void init_final_vertex(const uint32_t i, const int32_t tr_cap)
        void init_final_redge(const uint32_t i, const uint32_t j, const uint32_t head, const uint32_t mirror_sat, const int32_t r_cap)
        vector[int64_t] inspect_vertex(const uint32_t i)
        int64_t find_vertex_with_parallel_redge()
        

    # Graph class for extra large datasets using orphan path flags
    cdef cppclass Graph_XL "fcbk::Graph<uint64_t, true>":
        Graph_XL(uint64_t num_tmp_vertices) except +
        uint64_t add_vertices(uint64_t add_num_vertices)
        void add_edge(uint64_t i, uint64_t j, int32_t cap, int32_t rev_cap)
        void add_tweights(uint64_t i, int32_t cap_source, int32_t cap_sink)
        void init_maxflow()
        int64_t maxflow()
        TermType what_partition(uint64_t i, TermType default_segment)
        uint64_t get_num_vertices()
        uint64_t get_num_redges()
        vector[bool] get_partition()
        uint64_t get_total_size()
        void clear_tmp_struct()
        uint32_t init_final_struct(const uint32_t* degree_seq, uint64_t n)
        void init_final_vertex(const uint64_t i, const int32_t tr_cap)
        void init_final_redge(const uint64_t i, const uint32_t j, const uint64_t head, const uint32_t mirror_sat, const int32_t r_cap)
        vector[int64_t] inspect_vertex(const uint64_t i)
        int64_t find_vertex_with_parallel_redge()

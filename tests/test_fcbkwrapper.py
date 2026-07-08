import numpy as np
import pytest

import fcbkwrapper


GRAPH_CLASSES = (fcbkwrapper.GraphL, fcbkwrapper.GraphXL)
CHAIN_LENGTH = 3
CHAIN_VERTICES = np.array([0, 6, 14])
CHAIN_DEGREES = np.array([1, 2, 1], dtype=np.uint32)
CHAIN_CAPACITIES = np.array([4, 1], dtype=np.int32)
CHAIN_REVERSE_CAPACITIES = np.array([0, 0], dtype=np.int32)


def index_dtype(graph_cls):
    return np.uint32 if graph_cls is fcbkwrapper.GraphL else np.uint64


def make_temp_graph(graph_cls):
    graph = graph_cls()
    graph.add_vertices(CHAIN_LENGTH)
    graph.add_edge(0, 1, 4, 0)
    graph.add_edge(1, 2, 1, 0)
    graph.add_tweights(0, 5, 0)
    graph.add_tweights(2, 0, 3)
    return graph


def make_final_graph(graph_cls):
    graph = graph_cls()
    graph.init_final_struct(CHAIN_DEGREES)
    graph.init_final_vertices(
        CHAIN_VERTICES.astype(index_dtype(graph_cls)),
        np.array([5, 0, -3], dtype=np.int32),
    )
    return graph


def add_final_chain(graph, graph_cls):
    graph.init_final_redges(
        CHAIN_VERTICES[:2].astype(index_dtype(graph_cls)),
        CHAIN_VERTICES[1:].astype(index_dtype(graph_cls)),
        CHAIN_CAPACITIES,
        CHAIN_REVERSE_CAPACITIES,
    )


def assert_packed_size(graph):
    expected = 16 * graph.get_num_vertices() + 8 * graph.get_num_redges()
    assert graph.get_total_size() == expected


def assert_temporary_size(graph):
    expected = 28 * graph.get_num_vertices() + 8 * graph.get_num_redges()
    assert graph.get_total_size() == expected


@pytest.mark.parametrize("graph_cls", GRAPH_CLASSES)
def test_package_exports(graph_cls):
    assert hasattr(fcbkwrapper, graph_cls.__name__)


@pytest.mark.parametrize("graph_cls", GRAPH_CLASSES)
def test_temporary_graph_workflow(graph_cls):
    graph = make_temp_graph(graph_cls)

    assert graph.get_num_vertices() == 3
    assert graph.get_num_redges() == 4
    assert graph.find_vertex_with_parallel_redge() == -1
    assert_temporary_size(graph)

    graph.prepare_maxflow()

    assert graph.maxflow() == 1
    assert graph.get_partition() == [False, False, True]
    assert graph.get_num_vertices() == 3
    assert graph.get_num_redges() == 4


@pytest.mark.parametrize("graph_cls", GRAPH_CLASSES)
def test_clear_tmp_struct_smoke(graph_cls):
    graph = make_temp_graph(graph_cls)

    graph.clear_tmp_struct()

    assert graph.get_num_vertices() == 3
    assert graph.get_num_redges() == 4
    assert graph.get_total_size() == 0


@pytest.mark.parametrize("graph_cls", GRAPH_CLASSES)
def test_temporary_and_final_graph_match(graph_cls):
    temporary = make_temp_graph(graph_cls)
    final = make_final_graph(graph_cls)

    add_final_chain(final, graph_cls)

    assert temporary.get_num_vertices() == final.get_num_vertices() == 3
    assert temporary.get_num_redges() == final.get_num_redges() == 4
    assert_temporary_size(temporary)
    assert_packed_size(final)

    assert list(final.inspect_vertex(0))[0] == 1
    assert list(final.inspect_vertex(6))[0] == 2
    assert list(final.inspect_vertex(14))[0] == 1
    assert final.find_vertex_with_parallel_redge() == -1

    temporary.prepare_maxflow()
    temp_flow = temporary.maxflow()
    temp_partition = temporary.get_partition()

    final_flow = final.maxflow()
    final_partition = final.get_partition()

    assert temp_flow == final_flow == 1
    assert temp_partition == final_partition == [False, False, True]
    assert final.what_partition(0) == 0
    assert final.what_partition(6) == 0
    assert final.what_partition(14) == 1


@pytest.mark.parametrize("graph_cls", GRAPH_CLASSES)
def test_final_graph_direct_api(graph_cls):
    graph = make_final_graph(graph_cls)
    add_final_chain(graph, graph_cls)

    assert list(graph.inspect_vertex(0))[0] == 1
    assert list(graph.inspect_vertex(6))[0] == 2
    assert list(graph.inspect_vertex(14))[0] == 1
    assert graph.find_vertex_with_parallel_redge() == -1
    assert graph.get_num_vertices() == 3
    assert graph.get_num_redges() == 4
    assert_packed_size(graph)

    assert graph.get_partition() == [False, False, True]
    assert graph.maxflow() == 1
    assert graph.what_partition(0) == 0
    assert graph.what_partition(6) == 0
    assert graph.what_partition(14) == 1


@pytest.mark.parametrize(
    "graph_cls, a, b, cap, r_cap",
    [
        (
            graph_cls,
            np.array([0, 6], dtype=index_dtype(graph_cls)),
            np.array([6, 14], dtype=index_dtype(graph_cls)),
            np.array([4, 1], dtype=np.int32),
            np.array([0, 0], dtype=np.int32),
        )
        for graph_cls in GRAPH_CLASSES
    ]
    + [
        (
            graph_cls,
            np.array([[0, 6]], dtype=index_dtype(graph_cls)),
            np.array([[6, 14]], dtype=index_dtype(graph_cls)),
            np.array([[4, 1]], dtype=np.int32),
            np.array([[0, 0]], dtype=np.int32),
        )
        for graph_cls in GRAPH_CLASSES
    ]
    + [
        (
            graph_cls,
            np.array([[[0, 6]]], dtype=index_dtype(graph_cls)),
            np.array([[[6, 14]]], dtype=index_dtype(graph_cls)),
            np.array([[[4, 1]]], dtype=np.int32),
            np.array([[[0, 0]]], dtype=np.int32),
        )
        for graph_cls in GRAPH_CLASSES
    ]
    + [
        (
            graph_cls,
            np.array([[[[0, 6]]]], dtype=index_dtype(graph_cls)),
            np.array([[[[6, 14]]]], dtype=index_dtype(graph_cls)),
            np.array([[[[4, 1]]]], dtype=np.int32),
            np.array([[[[0, 0]]]], dtype=np.int32),
        )
        for graph_cls in GRAPH_CLASSES
    ]
    + [
        (
            graph_cls,
            np.array([[0, 6]], dtype=index_dtype(graph_cls)),
            np.array([6, 14], dtype=index_dtype(graph_cls)),
            np.array([[[4, 1]]], dtype=np.int32),
            np.array(0, dtype=np.int32),
        )
        for graph_cls in GRAPH_CLASSES
    ],
)
def test_init_final_redges_broadcasting(graph_cls, a, b, cap, r_cap):
    graph = make_final_graph(graph_cls)

    graph.init_final_redges(a, b, cap, r_cap)

    assert graph.get_num_redges() == 4
    assert list(graph.inspect_vertex(0))[0] == 1
    assert list(graph.inspect_vertex(6))[0] == 2
    assert list(graph.inspect_vertex(14))[0] == 1
    assert_packed_size(graph)
    assert graph.maxflow() == 1
    assert graph.get_partition() == [False, False, True]


@pytest.mark.parametrize("graph_cls", GRAPH_CLASSES)
def test_parallel_edge_detection_finds_duplicates(graph_cls):
    graph = graph_cls()
    graph.init_final_struct(np.array([2, 0], dtype=np.uint32))
    graph.init_final_vertices(
        np.array([0, 8], dtype=index_dtype(graph_cls)),
        np.array([0, 0], dtype=np.int32),
    )
    graph.init_final_redge(0, 4294967295, 8, 1, 2)
    graph.init_final_redge(0, 4294967295, 8, 1, 1)

    assert list(graph.inspect_vertex(0))[0] == 2
    assert graph.find_vertex_with_parallel_redge() == 0

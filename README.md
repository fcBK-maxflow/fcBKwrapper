# fcBKwrapper
A thin Python wrapper around the code for the fast and compact Boykov-Kolmogorov (`fcBK`) algorithm. The code is released in companionship with the paper:

<ul><b>Fast and Compact Graph Cuts for the Boykov-Kolmogorov Algorithm</b>,<br>
    <a href="https://christian.mikkelstrup.info/">Christian M. Mikkelstrup</a>, <a href="https://orbit.dtu.dk/en/persons/anders-bjorholm-dahl/">Anders B. Dahl</a>, <a href="https://people.compute.dtu.dk/phbi/">Philip Bille</a>, <a href="https://people.compute.dtu.dk/vand/">Vedrana A. Dahl</a>, <a href="https://people.compute.dtu.dk/inge/">Inge Li Gørtz</a>, 2026, (under review).<br>
  [ <a href="https://fcbk-maxflow.github.io/">Project page</a> ] [ <a href="https://doi.org/10.48550/arXiv.2605.13402">arXiv</a> ]
</ul>

For more information on the project, see the <a href="https://fcbk-maxflow.github.io/">project page</a>.

## Installation
Install this package by cloning this repository (including the [submodule](https://github.com/fcBK-maxflow/fcBK) using the `--recurse-submodules` flag). Then build the package using `pip install .`

## Minimal Example
```python
import numpy as np
import fcbkwrapper

# Builds the following graph:
#  s --5--> v_1 --4--> v_2 --1--> v_3 --3--> t

def compute_vertex_indices(degrees):
    # Calculate the indices of each vertex
    vertex_indices = np.zeros(len(degrees), dtype=np.uint32)
    for i in range(1, len(degrees)):
        vertex_indices[i] = vertex_indices[i - 1] + 4 + 2 * degrees[i - 1]

    return vertex_indices

def build_graph(use_temporary = False):
    # Options are GraphL (abs. indices) or GraphXL (rel. indices).
    graph = fcbkwrapper.GraphL()

    if use_temporary:
        # Adding vertices to the temporary data structure
        graph.add_vertices(3)
        # Adding internal edges with arguments (i, j, cap, rev_cap)
        graph.add_edge(0, 1, 4, 0)
        graph.add_edge(1, 2, 1, 0)
        # Adding terminal edges with arguments (i, cap_source, cap_sink)
        graph.add_tweights(0, 5, 0)
        graph.add_tweights(2, 0, 3)
        # Prepare the final graph structure from the temporary
        graph.prepare_maxflow()
    else:
        # Get the degrees of all vertices (in the residual graph)
        degrees = np.array([1, 2, 1], dtype=np.uint32)
        # Going from vertex index to placement in interleaved list
        vertex_indices = compute_vertex_indices(degrees)
        # Directly create the final data structure
        graph.init_final_struct(degrees)
        # Load the terminal capacities into vertices (negative to t)
        graph.init_final_vertices(vertex_indices, np.array([5, 0, -3], dtype=np.int32))
        # Load the capacity and residual capacity into residual edges
        graph.init_final_redges(
            vertex_indices[:-1],
            vertex_indices[1:],
            np.array([4, 1], dtype=np.int32),
            np.array([0, 0], dtype=np.int32),
        )

    return graph

if __name__ == "__main__":
    graph = build_graph()
    flow = graph.maxflow()
    partition = graph.get_partition()
    print(f"The flow is {flow}")
    print(f"The vertex is in T (for each vertex): {partition}")
```

## More functionality
For full functionality, consult the `fcbkwrapper/src/_fcbk.pyx` file and [this](https://github.com/fcBK-maxflow/fcBK#functionality).

## License
As the implementation of fcBK is distributed under the MIT license, so is this package.
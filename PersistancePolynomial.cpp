#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <boost/pending/disjoint_sets.hpp>
#include <iostream>
#include <ranges>

namespace py = pybind11;

py::tuple increasing_components(int n, const py::args &args) {
  int rank[n], parent[n];
  boost::disjoint_sets<int *, int *> ds(rank, parent);
  for (int i = 0; i < n; ++i) {
    ds.make_set(i);
  }

  std::ranges::iota_view<int, int> range(0, n);

  /* No pointer is passed, so NumPy will allocate the buffer */
  auto comps = py::array_t<int>(args.size() + 2);
  auto cycles = py::array_t<int>(args.size() + 1);
  int prev_comps = n, cur_comps = n;
  for (py::size_t i = 0; i < args.size(); ++i) {
    auto edges = py::cast<py::array_t<int>>(args[i]);
    for (py::ssize_t j = 0; j < edges.shape(0); ++j) {
      ds.union_set(edges.at(j, 0), edges.at(j, 1));
    }
    cur_comps = ds.count_sets(range.begin(), range.end());
    comps.mutable_at(args.size() - i) = prev_comps - cur_comps;
    prev_comps = cur_comps;

    cycles.mutable_at(args.size() - i - 1) =
        edges.shape(0) - comps.at(args.size() - i);
  }
  comps.mutable_at(0) = cur_comps;
  comps.mutable_at(args.size() + 1) = 0;
  cycles.mutable_at(args.size()) = 0;
  return py::make_tuple(comps, cycles);
}

PYBIND11_MODULE(components, m) {
  m.def("increasing_components", &increasing_components, "");
}

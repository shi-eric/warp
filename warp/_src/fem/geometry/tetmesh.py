# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import warp as wp
from warp._src.fem.cache import (
    TemporaryStore,
    borrow_temporary,
    borrow_temporary_like,
    cached_arg_value,
)
from warp._src.fem.types import (
    OUTSIDE,
    Coords,
    ElementIndex,
    Sample,
)

from .closest_point import project_on_tet_at_origin, project_on_tri_at_origin
from .element import Tetrahedron, Triangle
from .geometry import Geometry


@wp.struct
class TetmeshCellArg:
    tet_vertex_indices: wp.array2d(dtype=int)
    positions: wp.array(dtype=wp.vec3)

    # for global cell lookup
    tet_bvh: wp.uint64


@wp.struct
class TetmeshSideArg:
    cell_arg: TetmeshCellArg
    face_vertex_indices: wp.array(dtype=wp.vec3i)
    face_tet_indices: wp.array(dtype=wp.vec2i)


class Tetmesh(Geometry):
    """Tetrahedral mesh geometry"""

    dimension = 3

    def __init__(
        self,
        tet_vertex_indices: wp.array,
        positions: wp.array,
        build_bvh: bool = False,
        temporary_store: Optional[TemporaryStore] = None,
    ):
        """
        Constructs a tetrahedral mesh.

        Args:
            tet_vertex_indices: warp array of shape (num_tets, 4) containing vertex indices for each tet
            positions: warp array of shape (num_vertices, 3) containing 3d position for each vertex
            build_bvh: Whether to also build the tet BVH, which is necessary for the global `fem.lookup` operator to function without initial guess
            temporary_store: shared pool from which to allocate temporary arrays
        """

        self.tet_vertex_indices = tet_vertex_indices
        self.positions = positions

        self._face_vertex_indices: wp.array = None
        self._face_tet_indices: wp.array = None
        self._vertex_tet_offsets: wp.array = None
        self._vertex_tet_indices: wp.array = None
        self._tet_edge_indices: wp.array = None
        self._edge_count = 0
        self._build_topology(temporary_store)

        self._make_default_dependent_implementations()
        self.cell_coordinates = self._make_cell_coordinates(assume_linear=True)
        self.side_coordinates = self._make_side_coordinates(assume_linear=True)

        self._tet_bvh: wp.Bvh = None
        if build_bvh:
            self.build_bvh(self.positions.device)

    def cell_count(self):
        return self.tet_vertex_indices.shape[0]

    def vertex_count(self):
        return self.positions.shape[0]

    def side_count(self):
        return self._face_vertex_indices.shape[0]

    def edge_count(self):
        if self._tet_edge_indices is None:
            self._compute_tet_edges()
        return self._edge_count

    def boundary_side_count(self):
        return self._boundary_face_indices.shape[0]

    def reference_cell(self) -> Tetrahedron:
        return Tetrahedron()

    def reference_side(self) -> Triangle:
        return Triangle()

    @property
    def tet_edge_indices(self) -> wp.array:
        if self._tet_edge_indices is None:
            self._compute_tet_edges()
        return self._tet_edge_indices

    @property
    def face_tet_indices(self) -> wp.array:
        return self._face_tet_indices

    @property
    def face_vertex_indices(self) -> wp.array:
        return self._face_vertex_indices

    CellArg = TetmeshCellArg
    SideArg = TetmeshSideArg

    @wp.struct
    class SideIndexArg:
        boundary_face_indices: wp.array(dtype=int)

    # Geometry device interface

    def cell_arg_value(self, device) -> CellArg:
        args = self.CellArg()
        self.fill_cell_arg(args, device)
        return args

    def fill_cell_arg(self, args: CellArg, device):
        args.tet_vertex_indices = self.tet_vertex_indices.to(device)
        args.positions = self.positions.to(device)
        args.tet_bvh = self.bvh_id(device)

    @wp.func
    def cell_position(args: CellArg, s: Sample):
        tet_idx = args.tet_vertex_indices[s.element_index]
        w0 = 1.0 - s.element_coords[0] - s.element_coords[1] - s.element_coords[2]
        return (
            w0 * args.positions[tet_idx[0]]
            + s.element_coords[0] * args.positions[tet_idx[1]]
            + s.element_coords[1] * args.positions[tet_idx[2]]
            + s.element_coords[2] * args.positions[tet_idx[3]]
        )

    @wp.func
    def cell_deformation_gradient(args: CellArg, s: Sample):
        p0 = args.positions[args.tet_vertex_indices[s.element_index, 0]]
        p1 = args.positions[args.tet_vertex_indices[s.element_index, 1]]
        p2 = args.positions[args.tet_vertex_indices[s.element_index, 2]]
        p3 = args.positions[args.tet_vertex_indices[s.element_index, 3]]
        return wp.matrix_from_cols(p1 - p0, p2 - p0, p3 - p0)

    @wp.func
    def cell_inverse_deformation_gradient(args: CellArg, s: Sample):
        return wp.inverse(Tetmesh.cell_deformation_gradient(args, s))

    @wp.func
    def cell_closest_point(args: CellArg, tet_index: int, pos: wp.vec3):
        vidx = args.tet_vertex_indices[tet_index]
        p0 = args.positions[vidx[0]]

        q = pos - p0
        e1 = args.positions[vidx[1]] - p0
        e2 = args.positions[vidx[2]] - p0
        e3 = args.positions[vidx[3]] - p0

        dist, coords = project_on_tet_at_origin(q, e1, e2, e3)
        return coords, dist

    @cached_arg_value
    def side_index_arg_value(self, device) -> SideIndexArg:
        args = self.SideIndexArg()
        self.fill_side_index_arg(args, device)
        return args

    def fill_side_index_arg(self, args: SideIndexArg, device):
        args.boundary_face_indices = self._boundary_face_indices.to(device)

    @wp.func
    def boundary_side_index(args: SideIndexArg, boundary_side_index: int):
        """Boundary side to side index"""

        return args.boundary_face_indices[boundary_side_index]

    def side_arg_value(self, device) -> CellArg:
        args = self.SideArg()
        self.fill_side_arg(args, device)
        return args

    def fill_side_arg(self, args: SideArg, device):
        self.fill_cell_arg(args.cell_arg, device)
        args.face_vertex_indices = self._face_vertex_indices.to(device)
        args.face_tet_indices = self._face_tet_indices.to(device)

    @wp.func
    def side_position(args: SideArg, s: Sample):
        face_idx = args.face_vertex_indices[s.element_index]
        return (
            s.element_coords[0] * args.cell_arg.positions[face_idx[0]]
            + s.element_coords[1] * args.cell_arg.positions[face_idx[1]]
            + s.element_coords[2] * args.cell_arg.positions[face_idx[2]]
        )

    @wp.func
    def _side_vecs(args: SideArg, side_index: ElementIndex):
        face_idx = args.face_vertex_indices[side_index]
        v0 = args.cell_arg.positions[face_idx[0]]
        v1 = args.cell_arg.positions[face_idx[1]]
        v2 = args.cell_arg.positions[face_idx[2]]

        return v1 - v0, v2 - v0

    @wp.func
    def side_closest_point(args: SideArg, tri_index: ElementIndex, pos: wp.vec3):
        vidx = args.topology.face_vertex_indices[tri_index]
        p0 = args.positions[vidx[0]]

        q = pos - p0
        e1 = args.positions[vidx[1]] - p0
        e2 = args.positions[vidx[2]] - p0

        dist, coords = project_on_tri_at_origin(q, e1, e2)
        return coords, dist

    @wp.func
    def side_deformation_gradient(args: SideArg, s: Sample):
        e1, e2 = Tetmesh._side_vecs(args, s.element_index)
        return wp.matrix_from_cols(e1, e2)

    @wp.func
    def side_inner_cell_index(arg: SideArg, side_index: ElementIndex):
        return arg.face_tet_indices[side_index][0]

    @wp.func
    def side_outer_cell_index(arg: SideArg, side_index: ElementIndex):
        return arg.face_tet_indices[side_index][1]

    @wp.func
    def face_to_tet_coords(args: SideArg, side_index: ElementIndex, tet_index: ElementIndex, side_coords: Coords):
        fvi = args.face_vertex_indices[side_index]

        tv1 = args.cell_arg.tet_vertex_indices[tet_index, 1]
        tv2 = args.cell_arg.tet_vertex_indices[tet_index, 2]
        tv3 = args.cell_arg.tet_vertex_indices[tet_index, 3]

        c1 = float(0.0)
        c2 = float(0.0)
        c3 = float(0.0)

        for k in range(3):
            if tv1 == fvi[k]:
                c1 = side_coords[k]
            elif tv2 == fvi[k]:
                c2 = side_coords[k]
            elif tv3 == fvi[k]:
                c3 = side_coords[k]

        return Coords(c1, c2, c3)

    @wp.func
    def side_inner_cell_coords(args: SideArg, side_index: ElementIndex, side_coords: Coords):
        inner_cell_index = Tetmesh.side_inner_cell_index(args, side_index)
        return Tetmesh.face_to_tet_coords(args, side_index, inner_cell_index, side_coords)

    @wp.func
    def side_outer_cell_coords(args: SideArg, side_index: ElementIndex, side_coords: Coords):
        outer_cell_index = Tetmesh.side_outer_cell_index(args, side_index)
        return Tetmesh.face_to_tet_coords(args, side_index, outer_cell_index, side_coords)

    @wp.func
    def side_from_cell_coords(args: SideArg, side_index: ElementIndex, tet_index: ElementIndex, tet_coords: Coords):
        fvi = args.face_vertex_indices[side_index]

        tv1 = args.cell_arg.tet_vertex_indices[tet_index, 1]
        tv2 = args.cell_arg.tet_vertex_indices[tet_index, 2]
        tv3 = args.cell_arg.tet_vertex_indices[tet_index, 3]

        if tv1 == fvi[0]:
            c0 = tet_coords[0]
        elif tv2 == fvi[0]:
            c0 = tet_coords[1]
        elif tv3 == fvi[0]:
            c0 = tet_coords[2]
        else:
            c0 = 1.0 - tet_coords[0] - tet_coords[1] - tet_coords[2]

        if tv1 == fvi[1]:
            c1 = tet_coords[0]
        elif tv2 == fvi[1]:
            c1 = tet_coords[1]
        elif tv3 == fvi[1]:
            c1 = tet_coords[2]
        else:
            c1 = 1.0 - tet_coords[0] - tet_coords[1] - tet_coords[2]

        if tv1 == fvi[2]:
            c2 = tet_coords[0]
        elif tv2 == fvi[2]:
            c2 = tet_coords[1]
        elif tv3 == fvi[2]:
            c2 = tet_coords[2]
        else:
            c2 = 1.0 - tet_coords[0] - tet_coords[1] - tet_coords[2]

        return wp.where(c0 + c1 + c2 > 0.999, Coords(c0, c1, c2), Coords(OUTSIDE))

    @wp.func
    def side_to_cell_arg(side_arg: SideArg):
        return side_arg.cell_arg

    def _build_topology(self, temporary_store: TemporaryStore):
        from warp._src.fem.utils import compress_node_indices, host_read_at_index, masked_indices
        from warp._src.utils import array_scan

        device = self.tet_vertex_indices.device

        vertex_tet_offsets, vertex_tet_indices = compress_node_indices(
            self.vertex_count(), self.tet_vertex_indices, temporary_store=temporary_store
        )
        self._vertex_tet_offsets = vertex_tet_offsets.detach()
        self._vertex_tet_indices = vertex_tet_indices.detach()

        vertex_start_face_count = borrow_temporary(temporary_store, dtype=int, device=device, shape=self.vertex_count())
        vertex_start_face_count.array.zero_()
        vertex_start_face_offsets = borrow_temporary_like(vertex_start_face_count, temporary_store=temporary_store)

        vertex_face_other_vs = borrow_temporary(
            temporary_store, dtype=wp.vec2i, device=device, shape=(4 * self.cell_count())
        )
        vertex_face_tets = borrow_temporary(temporary_store, dtype=int, device=device, shape=(4 * self.cell_count(), 2))

        # Count face edges starting at each vertex
        wp.launch(
            kernel=Tetmesh._count_starting_faces_kernel,
            device=device,
            dim=self.cell_count(),
            inputs=[self.tet_vertex_indices, vertex_start_face_count.array],
        )

        array_scan(in_array=vertex_start_face_count.array, out_array=vertex_start_face_offsets.array, inclusive=False)

        # Count number of unique edges (deduplicate across faces)
        vertex_unique_face_count = vertex_start_face_count
        wp.launch(
            kernel=Tetmesh._count_unique_starting_faces_kernel,
            device=device,
            dim=self.vertex_count(),
            inputs=[
                self._vertex_tet_offsets,
                self._vertex_tet_indices,
                self.tet_vertex_indices,
                vertex_start_face_offsets.array,
                vertex_unique_face_count.array,
                vertex_face_other_vs.array,
                vertex_face_tets.array,
            ],
        )

        vertex_unique_face_offsets = borrow_temporary_like(vertex_start_face_offsets, temporary_store=temporary_store)
        array_scan(in_array=vertex_start_face_count.array, out_array=vertex_unique_face_offsets.array, inclusive=False)

        # Get back edge count to host
        face_count = int(
            host_read_at_index(
                vertex_unique_face_offsets.array, self.vertex_count() - 1, temporary_store=temporary_store
            )
        )

        self._face_vertex_indices = wp.empty(shape=(face_count,), dtype=wp.vec3i, device=device)
        self._face_tet_indices = wp.empty(shape=(face_count,), dtype=wp.vec2i, device=device)

        boundary_mask = borrow_temporary(temporary_store, shape=(face_count,), dtype=int, device=device)

        # Compress edge data
        wp.launch(
            kernel=Tetmesh._compress_faces_kernel,
            device=device,
            dim=self.vertex_count(),
            inputs=[
                vertex_start_face_offsets.array,
                vertex_unique_face_offsets.array,
                vertex_unique_face_count.array,
                vertex_face_other_vs.array,
                vertex_face_tets.array,
                self._face_vertex_indices,
                self._face_tet_indices,
                boundary_mask.array,
            ],
        )

        vertex_start_face_offsets.release()
        vertex_unique_face_offsets.release()
        vertex_unique_face_count.release()
        vertex_face_other_vs.release()
        vertex_face_tets.release()

        # Flip normals if necessary
        wp.launch(
            kernel=Tetmesh._flip_face_normals,
            device=device,
            dim=self.side_count(),
            inputs=[self._face_vertex_indices, self._face_tet_indices, self.tet_vertex_indices, self.positions],
        )

        boundary_face_indices, _ = masked_indices(boundary_mask.array)
        self._boundary_face_indices = boundary_face_indices.detach()

    def _compute_tet_edges(self, temporary_store: Optional[TemporaryStore] = None):
        from warp._src.fem.utils import host_read_at_index
        from warp._src.utils import array_scan

        device = self.tet_vertex_indices.device

        vertex_start_edge_count = borrow_temporary(temporary_store, dtype=int, device=device, shape=self.vertex_count())
        vertex_start_edge_count.array.zero_()
        vertex_start_edge_offsets = borrow_temporary_like(vertex_start_edge_count, temporary_store=temporary_store)

        vertex_edge_ends = borrow_temporary(temporary_store, dtype=int, device=device, shape=(6 * self.cell_count()))

        # Count face edges starting at each vertex
        wp.launch(
            kernel=Tetmesh._count_starting_edges_kernel,
            device=device,
            dim=self.cell_count(),
            inputs=[self.tet_vertex_indices, vertex_start_edge_count.array],
        )

        array_scan(in_array=vertex_start_edge_count.array, out_array=vertex_start_edge_offsets.array, inclusive=False)

        # Count number of unique edges (deduplicate across faces)
        vertex_unique_edge_count = vertex_start_edge_count
        wp.launch(
            kernel=Tetmesh._count_unique_starting_edges_kernel,
            device=device,
            dim=self.vertex_count(),
            inputs=[
                self._vertex_tet_offsets,
                self._vertex_tet_indices,
                self.tet_vertex_indices,
                vertex_start_edge_offsets.array,
                vertex_unique_edge_count.array,
                vertex_edge_ends.array,
            ],
        )

        vertex_unique_edge_offsets = borrow_temporary_like(
            vertex_start_edge_offsets.array, temporary_store=temporary_store
        )
        array_scan(in_array=vertex_start_edge_count.array, out_array=vertex_unique_edge_offsets.array, inclusive=False)

        # Get back edge count to host
        self._edge_count = int(
            host_read_at_index(
                vertex_unique_edge_offsets.array, self.vertex_count() - 1, temporary_store=temporary_store
            )
        )

        self._tet_edge_indices = wp.empty(
            dtype=int, device=self.tet_vertex_indices.device, shape=(self.cell_count(), 6)
        )

        # Compress edge data
        wp.launch(
            kernel=Tetmesh._compress_edges_kernel,
            device=device,
            dim=self.vertex_count(),
            inputs=[
                self._vertex_tet_offsets,
                self._vertex_tet_indices,
                self.tet_vertex_indices,
                vertex_start_edge_offsets.array,
                vertex_unique_edge_offsets.array,
                vertex_unique_edge_count.array,
                vertex_edge_ends.array,
                self._tet_edge_indices,
            ],
        )

        vertex_start_edge_offsets.release()
        vertex_unique_edge_offsets.release()
        vertex_unique_edge_count.release()
        vertex_edge_ends.release()

    @wp.kernel
    def _count_starting_faces_kernel(
        tet_vertex_indices: wp.array2d(dtype=int), vertex_start_face_count: wp.array(dtype=int)
    ):
        t = wp.tid()
        for k in range(4):
            vi = wp.vec3i(
                tet_vertex_indices[t, k], tet_vertex_indices[t, (k + 1) % 4], tet_vertex_indices[t, (k + 2) % 4]
            )
            vm = wp.min(vi)

            for i in range(3):
                if vm == vi[i]:
                    wp.atomic_add(vertex_start_face_count, vm, 1)

    @wp.func
    def _find_face(
        needle: wp.vec2i,
        values: wp.array(dtype=wp.vec2i),
        beg: int,
        end: int,
    ):
        for i in range(beg, end):
            if values[i] == needle:
                return i

        return -1

    @wp.kernel
    def _count_unique_starting_faces_kernel(
        vertex_tet_offsets: wp.array(dtype=int),
        vertex_tet_indices: wp.array(dtype=int),
        tet_vertex_indices: wp.array2d(dtype=int),
        vertex_start_face_offsets: wp.array(dtype=int),
        vertex_start_face_count: wp.array(dtype=int),
        face_other_vs: wp.array(dtype=wp.vec2i),
        face_tets: wp.array2d(dtype=int),
    ):
        v = wp.tid()

        face_beg = vertex_start_face_offsets[v]

        tet_beg = vertex_tet_offsets[v]
        tet_end = vertex_tet_offsets[v + 1]

        face_cur = face_beg

        for tet in range(tet_beg, tet_end):
            t = vertex_tet_indices[tet]

            for k in range(4):
                vi = wp.vec3i(
                    tet_vertex_indices[t, k], tet_vertex_indices[t, (k + 1) % 4], tet_vertex_indices[t, (k + 2) % 4]
                )
                min_v = wp.min(vi)

                if v == min_v:
                    max_v = wp.max(vi)
                    mid_v = vi[0] + vi[1] + vi[2] - min_v - max_v
                    other_v = wp.vec2i(mid_v, max_v)

                    # Check if other_v has been seen
                    seen_idx = Tetmesh._find_face(other_v, face_other_vs, face_beg, face_cur)

                    if seen_idx == -1:
                        face_other_vs[face_cur] = other_v
                        face_tets[face_cur, 0] = t
                        face_tets[face_cur, 1] = t
                        face_cur += 1
                    else:
                        face_tets[seen_idx, 1] = t

        vertex_start_face_count[v] = face_cur - face_beg

    @wp.kernel
    def _compress_faces_kernel(
        vertex_start_face_offsets: wp.array(dtype=int),
        vertex_unique_face_offsets: wp.array(dtype=int),
        vertex_unique_face_count: wp.array(dtype=int),
        uncompressed_face_other_vs: wp.array(dtype=wp.vec2i),
        uncompressed_face_tets: wp.array2d(dtype=int),
        face_vertex_indices: wp.array(dtype=wp.vec3i),
        face_tet_indices: wp.array(dtype=wp.vec2i),
        boundary_mask: wp.array(dtype=int),
    ):
        v = wp.tid()

        start_beg = vertex_start_face_offsets[v]
        unique_beg = vertex_unique_face_offsets[v]
        unique_count = vertex_unique_face_count[v]

        for f in range(unique_count):
            src_index = start_beg + f
            face_index = unique_beg + f

            face_vertex_indices[face_index] = wp.vec3i(
                v,
                uncompressed_face_other_vs[src_index][0],
                uncompressed_face_other_vs[src_index][1],
            )

            t0 = uncompressed_face_tets[src_index, 0]
            t1 = uncompressed_face_tets[src_index, 1]
            face_tet_indices[face_index] = wp.vec2i(t0, t1)
            if t0 == t1:
                boundary_mask[face_index] = 1
            else:
                boundary_mask[face_index] = 0

    @wp.kernel
    def _flip_face_normals(
        face_vertex_indices: wp.array(dtype=wp.vec3i),
        face_tet_indices: wp.array(dtype=wp.vec2i),
        tet_vertex_indices: wp.array2d(dtype=int),
        positions: wp.array(dtype=wp.vec3),
    ):
        e = wp.tid()

        tet = face_tet_indices[e][0]

        tet_vidx = tet_vertex_indices[tet]
        face_vidx = face_vertex_indices[e]

        tet_centroid = (
            positions[tet_vidx[0]] + positions[tet_vidx[1]] + positions[tet_vidx[2]] + positions[tet_vidx[3]]
        ) / 4.0

        v0 = positions[face_vidx[0]]
        v1 = positions[face_vidx[1]]
        v2 = positions[face_vidx[2]]

        face_center = (v1 + v0 + v2) / 3.0
        face_normal = wp.cross(v1 - v0, v2 - v0)

        # if face normal points toward first tet centroid, flip indices
        if wp.dot(tet_centroid - face_center, face_normal) > 0.0:
            face_vertex_indices[e] = wp.vec3i(face_vidx[0], face_vidx[2], face_vidx[1])

    @wp.kernel
    def _count_starting_edges_kernel(
        tri_vertex_indices: wp.array2d(dtype=int), vertex_start_edge_count: wp.array(dtype=int)
    ):
        t = wp.tid()
        for k in range(3):
            v0 = tri_vertex_indices[t, k]
            v1 = tri_vertex_indices[t, (k + 1) % 3]

            if v0 < v1:
                wp.atomic_add(vertex_start_edge_count, v0, 1)
            else:
                wp.atomic_add(vertex_start_edge_count, v1, 1)

        for k in range(3):
            v0 = tri_vertex_indices[t, k]
            v1 = tri_vertex_indices[t, 3]

            if v0 < v1:
                wp.atomic_add(vertex_start_edge_count, v0, 1)
            else:
                wp.atomic_add(vertex_start_edge_count, v1, 1)

    @wp.func
    def _find_edge(
        needle: int,
        values: wp.array(dtype=int),
        beg: int,
        end: int,
    ):
        for i in range(beg, end):
            if values[i] == needle:
                return i

        return -1

    @wp.kernel
    def _count_unique_starting_edges_kernel(
        vertex_tet_offsets: wp.array(dtype=int),
        vertex_tet_indices: wp.array(dtype=int),
        tet_vertex_indices: wp.array2d(dtype=int),
        vertex_start_edge_offsets: wp.array(dtype=int),
        vertex_start_edge_count: wp.array(dtype=int),
        edge_ends: wp.array(dtype=int),
    ):
        v = wp.tid()

        edge_beg = vertex_start_edge_offsets[v]

        tet_beg = vertex_tet_offsets[v]
        tet_end = vertex_tet_offsets[v + 1]

        edge_cur = edge_beg

        for tet in range(tet_beg, tet_end):
            t = vertex_tet_indices[tet]

            for k in range(3):
                v0 = tet_vertex_indices[t, k]
                v1 = tet_vertex_indices[t, (k + 1) % 3]

                if v == wp.min(v0, v1):
                    other_v = wp.max(v0, v1)
                    if Tetmesh._find_edge(other_v, edge_ends, edge_beg, edge_cur) == -1:
                        edge_ends[edge_cur] = other_v
                        edge_cur += 1

            for k in range(3):
                v0 = tet_vertex_indices[t, k]
                v1 = tet_vertex_indices[t, 3]

                if v == wp.min(v0, v1):
                    other_v = wp.max(v0, v1)
                    if Tetmesh._find_edge(other_v, edge_ends, edge_beg, edge_cur) == -1:
                        edge_ends[edge_cur] = other_v
                        edge_cur += 1

        vertex_start_edge_count[v] = edge_cur - edge_beg

    @wp.kernel
    def _compress_edges_kernel(
        vertex_tet_offsets: wp.array(dtype=int),
        vertex_tet_indices: wp.array(dtype=int),
        tet_vertex_indices: wp.array2d(dtype=int),
        vertex_start_edge_offsets: wp.array(dtype=int),
        vertex_unique_edge_offsets: wp.array(dtype=int),
        vertex_unique_edge_count: wp.array(dtype=int),
        uncompressed_edge_ends: wp.array(dtype=int),
        tet_edge_indices: wp.array2d(dtype=int),
    ):
        v = wp.tid()

        uncompressed_beg = vertex_start_edge_offsets[v]

        unique_beg = vertex_unique_edge_offsets[v]
        unique_count = vertex_unique_edge_count[v]

        tet_beg = vertex_tet_offsets[v]
        tet_end = vertex_tet_offsets[v + 1]

        for tet in range(tet_beg, tet_end):
            t = vertex_tet_indices[tet]

            for k in range(3):
                v0 = tet_vertex_indices[t, k]
                v1 = tet_vertex_indices[t, (k + 1) % 3]

                if v == wp.min(v0, v1):
                    other_v = wp.max(v0, v1)
                    edge_id = (
                        Tetmesh._find_edge(
                            other_v, uncompressed_edge_ends, uncompressed_beg, uncompressed_beg + unique_count
                        )
                        - uncompressed_beg
                        + unique_beg
                    )
                    tet_edge_indices[t][k] = edge_id

            for k in range(3):
                v0 = tet_vertex_indices[t, k]
                v1 = tet_vertex_indices[t, 3]

                if v == wp.min(v0, v1):
                    other_v = wp.max(v0, v1)
                    edge_id = (
                        Tetmesh._find_edge(
                            other_v, uncompressed_edge_ends, uncompressed_beg, uncompressed_beg + unique_count
                        )
                        - uncompressed_beg
                        + unique_beg
                    )
                    tet_edge_indices[t][k + 3] = edge_id

    @wp.func
    def cell_bvh_id(cell_arg: TetmeshCellArg):
        return cell_arg.tet_bvh

    @wp.func
    def cell_bounds(cell_arg: TetmeshCellArg, cell_index: ElementIndex):
        vidx = cell_arg.tet_vertex_indices[cell_index]
        p0 = cell_arg.positions[vidx[0]]
        p1 = cell_arg.positions[vidx[1]]
        p2 = cell_arg.positions[vidx[2]]
        p3 = cell_arg.positions[vidx[3]]

        return wp.min(wp.min(p0, p1), wp.min(p2, p3)), wp.max(wp.max(p0, p1), wp.max(p2, p3))

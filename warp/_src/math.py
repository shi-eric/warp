# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import warp

"""
Math helper functions for vector norms, quaternions, and transforms.
"""

_wp_module_name_ = "warp.math"

__all__ = [
    "norm_huber",
    "norm_l1",
    "norm_l2",
    "norm_pseudo_huber",
    "quat_from_euler",
    "quat_to_euler",
    "quat_to_rpy",
    "quat_twist",
    "quat_twist_angle",
    "smooth_normalize",
    "transform_compose",
    "transform_decompose",
    "transform_from_matrix",
    "transform_to_matrix",
    "transform_twist",
    "transform_wrench",
    "velocity_at_point",
]


@warp.func
def norm_l1(v: Any) -> float:
    """Compute the L1 norm of a vector v.

    .. math:: \\|v\\|_1 = \\sum_i |v_i|

    Args:
        v (Vector[Float, Any]): The vector to compute the L1 norm of.

    Returns:
        float: The L1 norm of the vector.
    """
    n = float(0.0)
    for i in range(len(v)):
        n += warp.abs(v[i])
    return n


@warp.func
def norm_l2(v: Any) -> float:
    """Compute the L2 norm of a vector v.

    .. math:: \\|v\\|_2 = \\sqrt{\\sum_i v_i^2}

    Args:
        v (Vector[Float, Any]): The vector to compute the L2 norm of.

    Returns:
        float: The L2 norm of the vector.
    """
    return warp.length(v)


@warp.func
def norm_huber(v: Any, delta: float = 1.0) -> float:
    """Compute the Huber norm of a vector v with a given delta.

    .. math::
        H(v) = \\begin{cases} \\frac{1}{2} \\|v\\|^2 & \\text{if } \\|v\\| \\leq \\delta \\\\ \\delta(\\|v\\| - \\frac{1}{2}\\delta) & \\text{otherwise} \\end{cases}

    .. image:: /img/norm_huber.svg
        :align: center

    Args:
        v (Vector[Float, Any]): The vector to compute the Huber norm of.
        delta (float): The threshold value, defaults to 1.0.

    Returns:
        float: The Huber norm of the vector.
    """
    a = warp.dot(v, v)
    if a <= delta * delta:
        return 0.5 * a
    return delta * (warp.sqrt(a) - 0.5 * delta)


@warp.func
def norm_pseudo_huber(v: Any, delta: float = 1.0) -> float:
    """Compute the "pseudo" Huber norm of a vector v with a given delta.

    .. math::
        H^\\prime(v) = \\delta \\sqrt{1 + \\frac{\\|v\\|^2}{\\delta^2}}

    .. image:: /img/norm_pseudo_huber.svg
        :align: center

    Args:
        v (Vector[Float, Any]): The vector to compute the Huber norm of.
        delta (float): The threshold value, defaults to 1.0.

    Returns:
        float: The Huber norm of the vector.
    """
    a = warp.dot(v, v)
    return delta * warp.sqrt(1.0 + a / (delta * delta))


@warp.func
def smooth_normalize(v: Any, delta: float = 1.0) -> Any:
    """Normalize a vector using the pseudo-Huber norm.

    See :func:`norm_pseudo_huber`.

    .. math::
        \\frac{v}{H^\\prime(v)}

    Args:
        v (Vector[Float, Any]): The vector to normalize.
        delta (float): The threshold value, defaults to 1.0.

    Returns:
        Vector[Float, Any]: The normalized vector.
    """
    return v / norm_pseudo_huber(v, delta)


@warp.func
def velocity_at_point(qd: "warp.spatial_vector", r: "warp.vec3") -> "warp.vec3":
    """Evaluate the linear velocity of an offset point on a rigid body.

    Given a spatial twist ``qd = (w, v)`` and a point offset ``r`` from the twist
    origin, this computes:

    .. math::
       v_p = v + w \\times r

    Args:
        qd: Spatial velocity ``(angular, linear)`` of the reference frame.
        r: Point position relative to the same frame origin.

    Returns:
        warp.vec3: Linear velocity of the point.
    """
    return warp.spatial_bottom(qd) + warp.cross(warp.spatial_top(qd), r)


@warp.func
def quat_twist(axis: "warp.vec3", q: "warp.quat") -> "warp.quat":
    """Extract the twist quaternion of ``q`` around ``axis``.

    This performs a swing-twist decomposition and returns the component of
    rotation in ``q`` whose rotation axis is parallel to ``axis``.

    Args:
        axis: Twist axis (expected to be normalized).
        q: Input quaternion in ``(x, y, z, w)`` layout.

    Returns:
        warp.quat: Unit twist quaternion.
    """
    a = warp.vec3(q[0], q[1], q[2])
    proj = warp.dot(a, axis)
    a = proj * axis
    return warp.normalize(warp.quat(a[0], a[1], a[2], q[3]))


@warp.func
def quat_twist_angle(axis: "warp.vec3", q: "warp.quat") -> float:
    """Return the twist magnitude of ``q`` around ``axis``.

    This returns an unsigned twist magnitude. For canonicalized quaternions
    (for example from :func:`quat_from_axis_angle` with angles in ``(-pi, pi)``),
    both ``+theta`` and ``-theta`` map to ``|theta|``.

    Args:
        axis: Twist axis (expected to be normalized).
        q: Input quaternion.

    Returns:
        float: Twist angle in radians in ``[0, pi]`` for canonicalized inputs.
    """
    return 2.0 * warp.acos(quat_twist(axis, q)[3])


@warp.func
def quat_to_rpy(q: "warp.quat") -> "warp.vec3":
    """Convert a quaternion to roll-pitch-yaw angles (ZYX convention).

    The output is ``(roll, pitch, yaw)`` in radians using extrinsic rotations
    about X, Y, and Z respectively.

    Args:
        q: Input quaternion in ``(x, y, z, w)`` order.

    Returns:
        warp.vec3: ``(roll, pitch, yaw)`` in radians.
    """

    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = warp.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = warp.clamp(t2, -1.0, 1.0)
    pitch_y = warp.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = warp.atan2(t3, t4)

    return warp.vec3(roll_x, pitch_y, yaw_z)


@warp.func
def quat_to_euler(q: "warp.quat", i: int, j: int, k: int) -> "warp.vec3":
    """Convert a quaternion into Euler angles for an axis sequence.

    The integers ``i``, ``j``, and ``k`` are axis indices in ``{0, 1, 2}``
    with ``i != j`` and ``j != k``. Axis-indexed output (suitable for
    :func:`quat_from_euler`) is supported for Tait-Bryan sequences
    (``i != k``).

    .. caution::

        This function returns an axis-indexed vector where ``e[0]`` is the
        angle about X, ``e[1]`` about Y, and ``e[2]`` about Z.

        That layout is only unambiguous for Tait-Bryan sequences
        (``i != k``), where all three rotations use distinct axis slots.

        For proper Euler sequences (``i == k``), the first and third
        rotations are both about the same axis. In that case, two different
        sequence positions must share one output slot, so the mapping loses
        information and cannot be a clean inverse of :func:`quat_from_euler`.

    Args:
        q: Input quaternion.
        i: Index of the first axis.
        j: Index of the second axis.
        k: Index of the third axis.

    Returns:
        warp.vec3: Euler angles in radians.
    """
    # The conversion formula below is derived for scalar-first quaternions
    # q = (w, x, y, z). Warp stores quaternions as (x, y, z, w), so remap
    # indices into a temporary scalar-first layout before applying it.
    q0 = q[3]
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]

    # i, j, k are assumed to follow 1-based indexing in the original
    # derivation; convert from Warp's 0-based axis indices.
    i += 1
    j += 1
    k += 1
    not_proper = True
    if i == k:
        not_proper = False
        k = 6 - i - j  # because i + j + k = 1 + 2 + 3 = 6
    e = float((i - j) * (j - k) * (k - i)) / 2.0  # Levi-Civita symbol
    a = q0
    b = q1
    c = q1
    d = q1 * e

    if i == 2:
        b = q2
    elif i == 3:
        b = q3

    if j == 2:
        c = q2
    elif j == 3:
        c = q3

    if k == 2:
        d = q2 * e
    elif k == 3:
        d = q3 * e

    if not_proper:
        qj = q1
        qk = q1
        qi = q1
        if j == 2:
            qj = q2
        elif j == 3:
            qj = q3
        if k == 2:
            qk = q2
        elif k == 3:
            qk = q3
        if i == 2:
            qi = q2
        elif i == 3:
            qi = q3

        a -= qj
        b += qk * e
        c += q0
        d -= qi
    t2 = warp.acos(2.0 * (a * a + b * b) / (a * a + b * b + c * c + d * d) - 1.0)
    tp = warp.atan2(b, a)
    tm = warp.atan2(d, c)
    t1 = 0.0
    t3 = 0.0
    if warp.abs(t2) < 1e-6:
        t3 = 2.0 * tp - t1
    elif warp.abs(t2 - warp.pi) < 1e-6:
        t3 = 2.0 * tm + t1
    else:
        t1 = tp - tm
        t3 = tp + tm
    if not_proper:
        t2 -= warp.HALF_PI
        t3 *= e

    # Place the solved angles back into (x, y, z) by axis index.
    # This keeps round-trips stable for Tait-Bryan orders:
    # quat_from_euler(quat_to_euler(q, i, j, k), i, j, k) ~= q.
    ex = t1
    ey = t2
    ez = t3
    if i == 2:
        ex = t2
        ey = t1
    elif i == 3:
        ex = t3
        ez = t1

    if j == 1:
        ex = t2
    elif j == 2:
        ey = t2
    else:
        ez = t2

    if k == 1:
        ex = t3
    elif k == 2:
        ey = t3
    else:
        ez = t3

    return warp.vec3(ex, ey, ez)


@warp.func
def quat_from_euler(e: "warp.vec3", i: int, j: int, k: int) -> "warp.quat":
    """Construct a quaternion from Euler angles and an axis sequence.

    Args:
        e: Euler angles in radians.
        i: Index of the first axis in ``{0, 1, 2}``.
        j: Index of the second axis in ``{0, 1, 2}``.
        k: Index of the third axis in ``{0, 1, 2}``.

    Notes:
        Angles are read by axis index (``e[0]``=X, ``e[1]``=Y, ``e[2]``=Z),
        so use distinct axes ``(i, j, k)`` for an unambiguous inverse with
        :func:`quat_to_euler`.

    Returns:
        warp.quat: Quaternion in ``(x, y, z, w)`` layout.
    """
    ei = e[i]
    ej = e[j]
    ek = e[k]

    hi = 0.5 * ei
    hj = 0.5 * ej
    hk = 0.5 * ek

    qi = warp.quat(0.0, 0.0, 0.0, 1.0)
    qj = warp.quat(0.0, 0.0, 0.0, 1.0)
    qk = warp.quat(0.0, 0.0, 0.0, 1.0)

    si = warp.sin(hi)
    sj = warp.sin(hj)
    sk = warp.sin(hk)
    ci = warp.cos(hi)
    cj = warp.cos(hj)
    ck = warp.cos(hk)

    if i == 0:
        qi = warp.quat(si, 0.0, 0.0, ci)
    elif i == 1:
        qi = warp.quat(0.0, si, 0.0, ci)
    else:
        qi = warp.quat(0.0, 0.0, si, ci)

    if j == 0:
        qj = warp.quat(sj, 0.0, 0.0, cj)
    elif j == 1:
        qj = warp.quat(0.0, sj, 0.0, cj)
    else:
        qj = warp.quat(0.0, 0.0, sj, cj)

    if k == 0:
        qk = warp.quat(sk, 0.0, 0.0, ck)
    elif k == 1:
        qk = warp.quat(0.0, sk, 0.0, ck)
    else:
        qk = warp.quat(0.0, 0.0, sk, ck)

    # For extrinsic rotations about fixed axes i -> j -> k, quaternion
    # composition is q = q_k * q_j * q_i.
    return warp.mul(qk, warp.mul(qj, qi))


@warp.func
def transform_twist(t: "warp.transform", x: "warp.spatial_vector") -> "warp.spatial_vector":
    """Transform a spatial twist between coordinate frames.

    For transform ``t = (R, p)`` and twist ``x = (w, v)``, the mapped twist is:

    .. math::
       w' = R w,\\quad v' = R v + p \\times w'

    Args:
        t: Rigid transform from source frame to destination frame.
        x: Spatial twist ``(angular, linear)`` expressed in the source frame.

    Returns:
        warp.spatial_vector: Twist expressed in the destination frame.
    """

    q = warp.transform_get_rotation(t)
    p = warp.transform_get_translation(t)

    w = warp.spatial_top(x)
    v = warp.spatial_bottom(x)

    w = warp.quat_rotate(q, w)
    v = warp.quat_rotate(q, v) + warp.cross(p, w)

    return warp.spatial_vector(w, v)


@warp.func
def transform_wrench(t: "warp.transform", x: "warp.spatial_vector") -> "warp.spatial_vector":
    """Transform a spatial wrench between coordinate frames.

    For transform ``t = (R, p)`` and wrench ``x = (tau, f)``, the mapped wrench is:

    .. math::
       f' = R f,\\quad \\tau' = R \\tau + p \\times f'

    Args:
        t: Rigid transform from source frame to destination frame.
        x: Spatial wrench ``(torque, force)`` expressed in the source frame.

    Returns:
        warp.spatial_vector: Wrench expressed in the destination frame.
    """

    q = warp.transform_get_rotation(t)
    p = warp.transform_get_translation(t)

    tau = warp.spatial_top(x)
    f = warp.spatial_bottom(x)

    f = warp.quat_rotate(q, f)
    tau = warp.quat_rotate(q, tau) + warp.cross(p, f)

    return warp.spatial_vector(tau, f)


def create_transform_from_matrix_func(dtype):
    mat44 = warp._src.types.matrix((4, 4), dtype)
    vec3 = warp._src.types.vector(3, dtype)
    transform = warp._src.types.transformation(dtype)

    def transform_from_matrix(mat: mat44) -> transform:
        """Construct a transformation from a 4x4 matrix.

        .. math::
            M = \\begin{bmatrix}
            R_{00} & R_{01} & R_{02} & p_x \\\\
            R_{10} & R_{11} & R_{12} & p_y \\\\
            R_{20} & R_{21} & R_{22} & p_z \\\\
            0 & 0 & 0 & 1
            \\end{bmatrix}

        Where:

        * :math:`R` is the 3x3 rotation matrix extracted from the rotational part of the input matrix.
        * :math:`p` is the 3D position vector :math:`[p_x, p_y, p_z]` extracted from the input matrix.

        Args:
            mat (Matrix[Float, Literal[4], Literal[4]]): Matrix to convert.

        Returns:
            Transformation[Float]: The transformation.
        """
        p = vec3(mat[0][3], mat[1][3], mat[2][3])
        q = warp.quat_from_matrix(mat)
        return transform(p, q)

    return transform_from_matrix


transform_from_matrix = warp.func(
    create_transform_from_matrix_func(warp.float32),
    name="transform_from_matrix",
)
warp.func(
    create_transform_from_matrix_func(warp.float16),
    name="transform_from_matrix",
)
warp.func(
    create_transform_from_matrix_func(warp.float64),
    name="transform_from_matrix",
)


def create_transform_to_matrix_func(dtype):
    mat44 = warp._src.types.matrix((4, 4), dtype)
    transform = warp._src.types.transformation(dtype)

    def transform_to_matrix(xform: transform) -> mat44:
        """Convert a transformation to a 4x4 matrix.

        .. math::
            M = \\begin{bmatrix}
            R_{00} & R_{01} & R_{02} & p_x \\\\
            R_{10} & R_{11} & R_{12} & p_y \\\\
            R_{20} & R_{21} & R_{22} & p_z \\\\
            0 & 0 & 0 & 1
            \\end{bmatrix}

        Where:

        * :math:`R` is the 3x3 rotation matrix created from the orientation quaternion of the input transform.
        * :math:`p` is the 3D position vector :math:`[p_x, p_y, p_z]` of the input transform.

        Args:
            xform (Transformation[Float]): Transformation to convert.

        Returns:
            Matrix[Float, Literal[4], Literal[4]]: The matrix.
        """
        p = warp.transform_get_translation(xform)
        q = warp.transform_get_rotation(xform)
        rot = warp.quat_to_matrix(q)
        # fmt: off
        return mat44(
            rot[0][0], rot[0][1], rot[0][2], p[0],
            rot[1][0], rot[1][1], rot[1][2], p[1],
            rot[2][0], rot[2][1], rot[2][2], p[2],
            dtype(0.0), dtype(0.0), dtype(0.0), dtype(1.0),
        )
        # fmt: on

    return transform_to_matrix


transform_to_matrix = warp.func(
    create_transform_to_matrix_func(warp.float32),
    name="transform_to_matrix",
)
warp.func(
    create_transform_to_matrix_func(warp.float16),
    name="transform_to_matrix",
)
warp.func(
    create_transform_to_matrix_func(warp.float64),
    name="transform_to_matrix",
)


def create_transform_compose_func(dtype):
    mat44 = warp._src.types.matrix((4, 4), dtype)
    quat = warp._src.types.quaternion(dtype)
    vec3 = warp._src.types.vector(3, dtype)

    def transform_compose(position: vec3, rotation: quat, scale: vec3):
        """Compose a 4x4 transformation matrix from a 3D position, quaternion orientation, and 3D scale.

        .. math::
            M = \\begin{bmatrix}
            s_x R_{00} & s_y R_{01} & s_z R_{02} & p_x \\\\
            s_x R_{10} & s_y R_{11} & s_z R_{12} & p_y \\\\
            s_x R_{20} & s_y R_{21} & s_z R_{22} & p_z \\\\
            0 & 0 & 0 & 1
            \\end{bmatrix}

        Where:

        * :math:`R` is the 3x3 rotation matrix created from the input quaternion orientation.
        * :math:`p` is the input 3D position vector :math:`[p_x, p_y, p_z]`.
        * :math:`s` is the input 3D scale vector :math:`[s_x, s_y, s_z]`.

        Args:
            position (Vector[Float, Literal[3]]): The 3D position vector.
            rotation (Quaternion[Float]): The quaternion orientation.
            scale (Vector[Float, Literal[3]]): The 3D scale vector.

        Returns:
            Matrix[Float, Literal[4], Literal[4]]: The transformation matrix.
        """
        R = warp.quat_to_matrix(rotation)
        # fmt: off
        return mat44(
            scale[0] * R[0,0], scale[1] * R[0,1], scale[2] * R[0,2], position[0],
            scale[0] * R[1,0], scale[1] * R[1,1], scale[2] * R[1,2], position[1],
            scale[0] * R[2,0], scale[1] * R[2,1], scale[2] * R[2,2], position[2],
            dtype(0.0), dtype(0.0), dtype(0.0), dtype(1.0),
        )
        # fmt: on

    return transform_compose


transform_compose = warp.func(
    create_transform_compose_func(warp.float32),
    name="transform_compose",
)
warp.func(
    create_transform_compose_func(warp.float16),
    name="transform_compose",
)
warp.func(
    create_transform_compose_func(warp.float64),
    name="transform_compose",
)


def create_transform_decompose_func(dtype):
    mat44 = warp._src.types.matrix((4, 4), dtype)
    vec3 = warp._src.types.vector(3, dtype)
    mat33 = warp._src.types.matrix((3, 3), dtype)
    zero = dtype(0.0)

    def transform_decompose(m: mat44):
        """Decompose a 4x4 transformation matrix into 3D position, quaternion orientation, and 3D scale.

        .. math::
            M = \\begin{bmatrix}
            s_x R_{00} & s_y R_{01} & s_z R_{02} & p_x \\\\
            s_x R_{10} & s_y R_{11} & s_z R_{12} & p_y \\\\
            s_x R_{20} & s_y R_{21} & s_z R_{22} & p_z \\\\
            0 & 0 & 0 & 1
            \\end{bmatrix}

        Where:

        * :math:`R` is the 3x3 rotation matrix extracted from the input matrix after removing scale.
        * :math:`p` is the 3D position vector :math:`[p_x, p_y, p_z]` extracted from the input matrix.
        * :math:`s` is the 3D scale vector :math:`[s_x, s_y, s_z]` extracted from the input matrix.

        Args:
            m (Matrix[Float, Literal[4], Literal[4]]): The matrix to decompose.

        Returns:
            Tuple[Vector[Float, Literal[3]], Quaternion[Float], Vector[Float, Literal[3]]]: A tuple containing the position vector, quaternion orientation, and scale vector.
        """
        # extract position
        position = vec3(m[0, 3], m[1, 3], m[2, 3])
        # extract rotation matrix components
        r00, r01, r02 = m[0, 0], m[0, 1], m[0, 2]
        r10, r11, r12 = m[1, 0], m[1, 1], m[1, 2]
        r20, r21, r22 = m[2, 0], m[2, 1], m[2, 2]
        # get scale magnitudes
        sx = warp.sqrt(r00 * r00 + r10 * r10 + r20 * r20)
        sy = warp.sqrt(r01 * r01 + r11 * r11 + r21 * r21)
        sz = warp.sqrt(r02 * r02 + r12 * r12 + r22 * r22)
        # normalize rotation matrix components
        if sx != zero:
            r00 /= sx
            r10 /= sx
            r20 /= sx
        if sy != zero:
            r01 /= sy
            r11 /= sy
            r21 /= sy
        if sz != zero:
            r02 /= sz
            r12 /= sz
            r22 /= sz
        # extract rotation (quaternion)
        rotation = warp.quat_from_matrix(mat33(r00, r01, r02, r10, r11, r12, r20, r21, r22))
        # extract scale
        scale = vec3(sx, sy, sz)
        return position, rotation, scale

    return transform_decompose


transform_decompose = warp.func(
    create_transform_decompose_func(warp.float32),
    name="transform_decompose",
)
warp.func(
    create_transform_decompose_func(warp.float16),
    name="transform_decompose",
)
warp.func(
    create_transform_decompose_func(warp.float64),
    name="transform_decompose",
)


# register API functions so they appear in the documentation

warp._src.context.register_api_function(
    norm_l1,
    group="Vector Math",
)
warp._src.context.register_api_function(
    norm_l2,
    group="Vector Math",
)
warp._src.context.register_api_function(
    norm_huber,
    group="Vector Math",
)
warp._src.context.register_api_function(
    norm_pseudo_huber,
    group="Vector Math",
)
warp._src.context.register_api_function(
    smooth_normalize,
    group="Vector Math",
)
warp._src.context.register_api_function(
    quat_from_euler,
    group="Quaternion Math",
)
warp._src.context.register_api_function(
    quat_to_euler,
    group="Quaternion Math",
)
warp._src.context.register_api_function(
    quat_to_rpy,
    group="Quaternion Math",
)
warp._src.context.register_api_function(
    quat_twist,
    group="Quaternion Math",
)
warp._src.context.register_api_function(
    quat_twist_angle,
    group="Quaternion Math",
)
warp._src.context.register_api_function(
    velocity_at_point,
    group="Spatial Math",
)
warp._src.context.register_api_function(
    transform_from_matrix,
    group="Transformations",
)
warp._src.context.register_api_function(
    transform_to_matrix,
    group="Transformations",
)
warp._src.context.register_api_function(
    transform_compose,
    group="Transformations",
)
warp._src.context.register_api_function(
    transform_decompose,
    group="Transformations",
)
warp._src.context.register_api_function(
    transform_twist,
    group="Spatial Math",
)
warp._src.context.register_api_function(
    transform_wrench,
    group="Spatial Math",
)

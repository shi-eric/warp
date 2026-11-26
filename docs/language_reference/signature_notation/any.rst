.. _builtin-notation-any:

Any notation
============

In an input annotation, ``Any`` is a placeholder for a concrete type or value.
For example, ``Array[Any]`` leaves the array element type open and ``Vector[Scalar, Any]`` leaves
the vector length open. The overload can still impose constraints on the concrete arguments.
In a return annotation, ``Any`` leaves the concrete result type to be determined from the call.

See :ref:`reading-built-in-signatures` for examples of how overloads use this notation.

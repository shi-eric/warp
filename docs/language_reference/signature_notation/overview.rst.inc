.. _reading-built-in-signatures:

Reading Built-In Signatures
---------------------------

Built-in signatures use bracketed notation to describe families of Warp types.
The entries in brackets specify properties such as the element type or shape.
:data:`~warp.Scalar`, :data:`~warp.Float`, and :data:`~warp.Int` specify families of scalar types.

The overload determines whether repeated placeholders must agree and how the inputs affect the result type.

- :func:`~warp.cross` uses ``Vector[Scalar, Literal[3]]`` for both inputs and the result.
  The two inputs must have the same scalar type. For example, two ``wp.vec3f`` inputs produce a ``wp.vec3f``.
- :func:`~warp.outer` uses ``Vector[Scalar, Any]`` for both inputs. The lengths can differ.
  A ``wp.vec2f`` and a ``wp.vec3f`` produce a 2-by-3 matrix of ``wp.float32`` values.
- :func:`~warp.tile_store` takes an ``Array[Any]`` and a ``Tile[Any, tuple[int, ...]]``.
  The array and tile must have the same element type and number of dimensions.

.. toctree::
   :maxdepth: 1
   :caption: Notation terms

   Any <signature_notation/any>
   Literal <signature_notation/literal>
   Vector <signature_notation/vector>
   Matrix <signature_notation/matrix>
   Quaternion <signature_notation/quaternion>
   Transformation <signature_notation/transformation>
   Array <signature_notation/array>
   IndexedArray <signature_notation/indexedarray>
   FabricArray <signature_notation/fabricarray>
   IndexedFabricArray <signature_notation/indexedfabricarray>
   Tile <signature_notation/tile>
   TileStack <signature_notation/tilestack>

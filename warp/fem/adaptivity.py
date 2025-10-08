# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# isort: skip_file

from warp._src.fem.adaptivity import adaptive_nanogrid_from_field as adaptive_nanogrid_from_field


# TODO: Remove after cleaning up the public API.

from warp._src.fem import adaptivity as _adaptivity


def __getattr__(name):
    from warp._src.utils import get_deprecated_api

    return get_deprecated_api(_adaptivity, "wp.fem", name)

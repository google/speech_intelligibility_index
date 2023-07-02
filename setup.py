# Copyright 2023 The speech_intelligibility_index Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2023 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Create the speech_intelligibility_index package files.
"""

import setuptools

with open('README.md', 'r') as fh:
  long_description = fh.read()

setuptools.setup(
    name='speech_intelligibility_index',
    version='1.0.0',
    author='Malcolm Slaney',
    author_email='speech-intelligibility-index-maintainers@googlegroups.com',
    description='Speech Intelligibility Index',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/google/speech_intelligibility_index',
    packages=['speech_intelligibility_index'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'absl-py',
        'numpy',
        # 'jax',  # Optional
    ],
    include_package_data=True,  # Using the files specified in MANIFEST.in
)
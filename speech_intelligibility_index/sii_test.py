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

"""Tests for SII code.

Original Matlab Version Copyright 2003-2005 Hannes Muesch
Translation to Matlab by Malcolm Slaney, Google Sound Understanding Team

Copyright 2022 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from absl.testing import absltest
import matplotlib.pyplot as plt
import numpy as np

import sii


class ExampleTest(absltest.TestCase):

  def example_1(self):
    # Example 1:
    #
    # Talker and listener are facing each other in a free field. There is no
    # background noise. Calculate the SII as a function of the distance between
    # the talker and the listener.
    #
    # Assumptions:
    # Talking effort: "normal" and "shouted" speech
    # Importance function of "average speech" (default)
    # Hearing threshold is 0dB HL (normal hearing -- default)
    num_channels = 2                  # binaural listening
    distances = np.arange(0.5, 20.1, 0.5)  # distances from 0.5m to 20m by 0.5m

    sn_list = []
    ss_list = []
    for distance in distances:
      # Normal vocal effort
      [ssl, nsl, hearing_threshold] = sii.input_5p1(ssl='normal',
                                                    num_channels=num_channels,
                                                    distance=distance)
      sn_list.append(sii.sii(ssl=ssl, nsl=nsl,
                             hearing_threshold=hearing_threshold))

      # Shouted speech
      [ssl, nsl, hearing_threshold] = sii.input_5p1(ssl='shout',
                                                    num_channels=num_channels,
                                                    distance=distance)
      ss_list.append(sii.sii(ssl=ssl, nsl=nsl,
                             hearing_threshold=hearing_threshold))

    plt.plot(distances, sn_list, 'o-b')
    plt.plot(distances, ss_list, 'x-r')
    plt.grid(True)
    plt.title('SII in quiet free-field')
    plt.xlabel('Distance between talker and listener [m]')
    plt.ylabel('SII')
    plt.legend(('normal vocal effort', 'shouted speech'))
    plt.axis([0.4, 20.1, 0, 1])

    plt.savefig('/tmp/example1.png')
    return ss_list, sn_list

  def test_example_1(self):
    ss_list, sn_list = self.example_1()

    # pylint: disable=line-too-long  # Match Matlab output.
    expected_ss = np.array([
        0.8809, 0.9177, 0.9386, 0.9522, 0.9623, 0.9705, 0.9768, 0.9819, 0.9865, 0.9899,
        0.9928, 0.9949, 0.9967, 0.9977, 0.9979, 0.9975, 0.9970, 0.9966, 0.9962, 0.9958,
        0.9954, 0.9950, 0.9947, 0.9943, 0.9940, 0.9937, 0.9934, 0.9931, 0.9929, 0.9926,
        0.9924, 0.9921, 0.9919, 0.9916, 0.9914, 0.9912, 0.9909, 0.9906, 0.9903, 0.9900])

    expected_sn = np.array([
        1.0000, 0.9969, 0.9947, 0.9920, 0.9885, 0.9856, 0.9831, 0.9810, 0.9791, 0.9775,
        0.9760, 0.9746, 0.9722, 0.9698, 0.9677, 0.9657, 0.9638, 0.9619, 0.9600, 0.9583,
        0.9567, 0.9551, 0.9532, 0.9508, 0.9477, 0.9434, 0.9380, 0.9322, 0.9266, 0.9203,
        0.9143, 0.9084, 0.9024, 0.8962, 0.8901, 0.8844, 0.8788, 0.8733, 0.8680, 0.8627])

    np.testing.assert_allclose(np.array(ss_list), expected_ss, atol=1e-4)
    np.testing.assert_allclose(np.array(sn_list), expected_sn, atol=1e-4)

  def example_2(self):
    # Example 2
    #
    # Consider the following communication situation:
    # A talker and a listener face each other and are 1m appart. There is a
    # background of white noise and the talker shouts.
    # Calculate the SII for this communication situation as a function of
    # noise level when the listener wears hearing protection and when he
    # does not wear hearing protection.
    #
    # Assumptions:
    # Attenuation of Elvex Blue (TM) Foam Ear Plugs as listed on the companie's
    # web site (interpolated onto standard 1/3 oct band frequencies)

    # pylint: disable=invalid-name  # Use variable names in ANSI standard.
    A = np.array([36.1, 37.0, 38.0, 37.7, 37.4, 37.2, 37.0, 36.9, 36.7, 36.4,
                  36.1, 35.8, 38.0, 40.3, 40.7, 41.0, 41.5, 42.5])  # dB

    # relative spectral density level of white noise (a flat spectrum):
    N = np.zeros(18)
    # Absolute spectral density level
    absolute_levels = np.arange(0, 80.1, 2)

    # binaural listening.
    b = 2

    shout_list = []
    protected_list = []
    for level in absolute_levels:
      # No hearing protection
      [ssl, nsl, _] = sii.input_5p1(ssl='shout', num_channels=b, nsl=N+level)
      shout_list.append(sii.sii(ssl=ssl, nsl=nsl))

      # With hearing protection
      [ssl, nsl, _] = sii.input_5p1(ssl='shout', num_channels=b, nsl=level,
                                    insertion_gain=-A)
      protected_list.append(sii.sii(ssl=ssl, nsl=nsl))

    plt.plot(absolute_levels, shout_list, 'o-b')
    plt.plot(absolute_levels, protected_list, 'x-r')
    plt.grid(True)
    plt.title('SII in noise with and w/o hearing protection')
    plt.xlabel('Spectral Density Level of White Noise [dB]')
    plt.ylabel('SII')
    plt.legend(('No hearing protection', 'Foam Ear Plugs'))
    plt.axis([0, 80, 0, 1])
    plt.savefig('/tmp/example1.png')

    return shout_list, protected_list

  def test_example_2(self):
    shout_list, protected_list = self.example_2()
    # Expected results from the Matlab command: Example2
    # pylint: disable=line-too-long,bad-whitespace  # Match Matlab output.
    expected_speech = np.array([
        0.9177, 0.9177, 0.9177, 0.9173, 0.9161, 0.9126, 0.9092, 0.9035, 0.8967, 0.8894,
        0.8795, 0.8672, 0.8508, 0.8306, 0.8070, 0.7790, 0.7482, 0.7073, 0.6565, 0.5960,
        0.5372, 0.4783, 0.4225, 0.3669, 0.3117, 0.2604, 0.2105, 0.1657, 0.1250, 0.0865,
        0.0534, 0.0245, 0.0062, 0,      0,      0,      0,      0,      0,      0,
        0])

    expected_protection = np.array([
        0.9400, 0.9400, 0.9400, 0.9400, 0.9400, 0.9400, 0.9400, 0.9400, 0.9400, 0.9384,
        0.9327, 0.9240, 0.9116, 0.8942, 0.8715, 0.8457, 0.8164, 0.7780, 0.7277, 0.6624,
        0.5988, 0.5352, 0.4738, 0.4138, 0.3542, 0.2973, 0.2432, 0.1943, 0.1498, 0.1082,
        0.0709, 0.0367, 0.0126, 0,      0,      0,      0,      0,      0,      0,
        0])

    np.testing.assert_allclose(np.array(shout_list),
                               expected_speech, atol=1e-4)
    np.testing.assert_allclose(np.array(protected_list),
                               expected_protection, atol=1e-4)


if __name__ == '__main__':
  absltest.main()

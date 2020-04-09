# RDFS
### [Effectiveness of random deep feature selection for securing image manipulation detectors against adversarial examples]

Authors: [Mauro Barni](https://scholar.google.it/citations?hl=en&user=ntRScY8AAAAJ), [Ehsan Nowroozi](https://scholar.google.com/citations?user=C0bNkP8AAAAJ&hl=en), [Benedetta Tondi](https://scholar.google.it/citations?hl=en&user=xpNEfq4AAAAJ) and Bowen Zhang

2018-2019 Department of Information Engineering and Mathematics, University of Siena, Italy.

Author: Ehsan Nowroozi (Ehsan.Nowroozi65@gmail.com)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.

If you are using this software, please cite from [Arxiv](https://arxiv.org/abs/1910.12392). Also, you can find this paper in the [IEEE](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9053318).

@article{Barni2019EffectivenessOR,
  title={Effectiveness of random deep feature selection for securing image manipulation detectors against adversarial examples},
  author={Mauro Barni and Ehsan Nowroozi and Benedetta Tondi and Baochang Zhang},
  journal={ArXiv},
  year={2019},
  volume={abs/1910.12392}
}

Abstract- 
We investigate if the random feature selection approach proposed
in [1] to improve the robustness of forensic detectors to targeted attacks,
can be extended to detectors based on deep learning features.
In particular, we study the transferability of adversarial examples
targeting an original CNN image manipulation detector to other detectors
(a fully connected neural network and a linear SVM) that rely
on a random subset of the features extracted from the flatten layer of
the original network. The results we got by considering three image
manipulation detection tasks (resizing, median filtering and adaptive
histogram equalization), two original network architectures and
three classes of attacks, show that feature randomization helps to
hinder attack transferability, even if, in some cases, simply changing
the architecture of the detector, or even retraining the detector is
enough to prevent the transferability of the attacks.



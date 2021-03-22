Multilayer-SSL-NFFT-Examples Version 2.0, 2020-10-07

Multilayer-SSL-NFFT-Examples is a collection of MATLAB code that
performs semi-supervised learning based on a diffuse interface approach 
on multilayer graphs with acceleration for large image datasets 
using the NFFT-based fast summation.

The collection consists of the following files:

README
 - This file.

COPYING
 - A copy of the GPL v2 license.

Example_5_beach_evolution/beach_eighth_evolution.m
- Generates the images of Figure 2 visualizing the evolution of a graph based
  phase-field simulation for image segmentation of the example from Figure 3
  based on a 4-class 2-layer Graph Allen-Cahn classification scheme
  [1, Algorithm 6.1] using the power mean Laplacian with p=1 and the
  NFFT-based fast summation for the eigeninformation computations.
  
Example_8_1_SBM/SBM_example1.m
- Section 8.1 numerical experiment on 3-layer stochastic block model (SBM)
  graphs with each layer carrying the information about one class for the
  Allen-Cahn multiclass classification scheme [1, Algorithm 6.1] using the
  power mean Laplacian with two and three layers respectively as well as
  the three single layer graphs.

Example_8_1_SBM/SBM_example2.m
- Section 8.1 numerical experiment on 2-layer stochastic block model (SBM)
  graphs with and without one noisy layer for the Allen-Cahn multiclass
  classification scheme [1, Algorithm 6.1] using the power mean Laplacian.

Example_8_2_small_multilayer/*_multiple_rng.m
- Section 8.2 numerical experiment on various small but high-dimensional
  single layer and multilayer data sets for the Allen-Cahn multiclass
  classification scheme [1, Algorithm 6.1] using the power mean Laplacian.
  
Example_8_3_Image_data/Fig5_single_beach_image_full.m
- Section 8.3 numerical experiment on image data for the Allen-Cahn multiclass
  classification scheme [1, Algorithm 6.1] using the power mean Laplacian with
  a splitting into 2 layers (RGB color + XY pixel coordinates) using the
  NFFT-based fast summation for the eigeninformation computations.

Example_8_3_Image_data/Fig6_Fig7_two_beach_images_transfer.m
- Section 8.3 numerical experiment on transfer learning from one image to
  another for the Allen-Cahn multiclass classification scheme
  [1, Algorithm 6.1] using the power mean Laplacian with a splitting into
  2 layers (RGB color + XY pixel coordinates) using the NFFT-based fast
  summation for the eigeninformation computations.

Example_8_4_Pavia_center/Pavia_center*.m
- Section 8.4 numerical experiment on the Pavia center data set
  (hyperspectral image classification) for the Allen-Cahn multiclass
  classification scheme [1, Algorithm 6.1] using the power mean Laplacian
  using the NFFT-based fast summation for the eigeninformation computations.

Subroutines/convexity_splitting_vector_modified_fast.m
- Semi-supervised classification scheme using the Allen-Cahn model with a
  smooth potential and convexity splitting, which realize steps 2-4 of
  [1, Algorithm 6.1]. 

Subroutines/dist2.m
- Computes Euclidean distance matrix between two matrices. 

Subroutines/fastsumAdjacencyEigs.m
- MATLAB code to approximate the largest eigenvalues of a graph adjacency matrix
  based on fastsumAdjacencySetup.m and MATLAB's EIGS function.

Subroutines/fastsumAdjacencySetup.m
- MATLAB code to setup fast evaluation of matrix-vector products with the graph
  adjacency matrix or its normalized version using the NFFT3 toolbox.

Subroutines/fh_pml.m
- Calculates the arithmetic mean of a cell of size [1,T] applied to a vector w. 

Subroutines/gen_arnoldi_for_power_of_a_matrix_times_a_vector_modified.m
- Computes an M-orthogonal basis of the generalized Krylov subspace K_k(P, Q, v) = {v, P \ (Q v), ..., (P \ Q)^k v)} well as the generalized Arnoldi decomposition of P and Q.

Subroutines/generate_sbm_graph.m
- Generates a random graph following the prescribed stochastic block model (SBM) distribution. 

Subroutines/MV_fastsum_T_layers.m
- Calculates the arithmetic mean of a cell of NFFT-fastsum objects applied to
  a vector x

Subroutines/power_of_a_matrix.m
- Calculates the power of a matrix based on its spectral decomposition.

Subroutines/sample_idx_per_class.m
- Randomly selects a given equal percentage/number of elements of label information for each class. 

Subroutines/fastsum
- Compiled MATLAB interface of the NFFT-based fast summation x64
  Windows/Linux/macosX, see the matlab/fastsum/ folder within the
  NFFT3 software library available at
    https://www-user.tu-chemnitz.de/~potts/nfft/
      or
    https://github.com/NFFT/nfft

For all MATLAB source files, please refer to the descriptions inside the files
for further information.


REFERENCE
 [1] - Kai Bergermann, Martin Stoll, and Toni Volkmer. Semi-supervised Learning
       for Multilayer Graphs Using Diffuse Interface Methods and Fast Matrix
       Vector Products. Submitted, 2020.
       Preprint available at https://arxiv.org/abs/2007.05239

 [2] - Dominik Alfke, Daniel Potts, Martin Stoll, Toni Volkmer - NFFT meets
       Krylov methods: Fast matrix-vector products for the graph Laplacian of
	   fully connected networks. Front. Appl. Math. Stat. 4:61, 2018.
	   Available at http://dx.doi.org/10.1007/s00211-016-0861-7

PREREQUISITES
 - This software has been tested in MATLAB R2018a, but it should run in older 
   versions as well.
 - The MATLAB interface of the NFFT-based fast summation has been compiled
   for x64 Windows/Linux/macosX using GCC with flag -march=haswell.
   Therefore, it may not work on older CPUs (below Intel i3/i5/i7-4xxx or
   AMD Excavator/4th gen Bulldozer) as well as on some Intel Atom/Pentium CPUs.
   If you want to compile it yourself, please follow these instructions:
   - Download and unpack the toolbox from 
       https://www-user.tu-chemnitz.de/~potts/nfft/
   - Run the 'configure' script with option
       --enable-openmp --with-matlab=/PATH/TO/MATLAB/HOME/FOLDER.
   - Run 'make' and optionally 'make check'.
   - Copy the compiled MATLAB interface matlab/fastsum/fastsummex.mex* to the
     Multilayer-SSL-NFFT-Examples/Subroutines/fastsum/ folder.

LICENSE
 - Most of this software (cf. the header of each file)
   is distributed under the GNU General Public License v2.
   See COPYING for the full license text. If that file is not available, see 
   <http://www.gnu.org/licenses/>.


AUTHORS
 - Dominik Alfke, TU Chemnitz, dominik.alfke(at)math.tu-chemnitz.de
 - Kai Bergermann, TU Chemnitz, kai.bergermann(at)math.tu-chemnitz.de
 - Toni Volkmer, TU Chemnitz, toni.volkmer(at)math.tu-chemnitz.de

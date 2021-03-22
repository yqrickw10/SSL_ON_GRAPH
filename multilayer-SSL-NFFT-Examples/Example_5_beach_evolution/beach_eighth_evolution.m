% Numerical experiment on image data
% (original file 20170423_140149.jpg Copyright (c) 2017 Kai Bergermann)
% for the Allen-Cahn multiclass classification scheme [1, Algorithm 6.1]
% with a splitting into a 2-layer graph (RGB values and XY pixel coordinates)
% using the power mean Laplacian [2] with p=1
% using the NFFT-based fast summation [3] for the eigeninformation computations [4]
% plotting the evolution of the underlying phase-field for pixels assigned 
% to one class with a probability of over 0.66. 
%
% [1] Kai Bergermann, Martin Stoll, and Toni Volkmer. Semi-supervised Learning for Multilayer Graphs Using Diffuse Interface Methods and Fast Matrix Vector Products. Submitted, 2020. 
% [2] Pedro Mercado, Antoine Gautier, Francesco Tudisco, and Matthias Hein. The power mean Laplacian for multilayer graph clustering. In Proceedings of the Twenty-First International Conference on Artificial Intelligence and Statistics, volume 84 of Proceedings of Machine Learning Research, pages 1828â€“1838, 2018.
% [3] Daniel Potts, Gabriele Steidl, and Arthur Nieslony. Fast convolution with radial kernels at nonequispaced knots. Numerische Mathematik 98 (2014), pp. 329–351.
% [4] Dominik Alfke, Daniel Potts, Martin Stoll, and Toni Volkmer. NFFT meets Krylov methods: Fast matrix-vector products for the graph Laplacian of fully connected networks. Frontiers in Applied Mathematics and Statistics, 4:61, 2018.
%
% This software is distributed under the GNU General Public License v2. See 
% COPYING for the full license text. If that file is not available, see 
% <http://www.gnu.org/licenses/>.
%
% Copyright (c) 2019-2020 Kai Bergermann, Toni Volkmer

clear all

% add paths.
addpath('../Subroutines')
addpath('../Subroutines/fastsum')

%% read image data
IM=imread('../Example_8_3_Image_data/Data/20170423_140149_eighth_resolution_painted_scaled.png');
IM_full=imread('../Example_8_3_Image_data/Data/20170423_140149_eighth_resolution.png');

% get the size
nx=size(IM,1); ny=size(IM,2); n=nx*ny;

%% RGB layer
% vetorize each color channel
data=double([reshape(IM_full(:,:,1),[n,1]),reshape(IM_full(:,:,2),[n,1]),reshape(IM_full(:,:,3),[n,1])]);

% center and scale
data = data - repmat(mean(data),size(data,1),1); %feature-wise centering
data = data/max(max(abs(data))); %layer-wise scaling

%% Coordinate layer
% spatial pixel coordinates
[X,Y]=ndgrid(1:nx, 1:ny);
coords=[reshape(X,[n,1]),reshape(Y,[n,1])];

% center and scale
coords = coords - repmat(mean(coords),size(coords,1),1); %feature-wise centering
coords = coords/max(max(abs(coords))); %layer-wise scaling

%% extract painted regions
% tree class: blue
B_log=(IM(:,:,1)<5 & IM(:,:,2)<5 & IM(:,:,3)>250);

% beach class: red
R_log=(IM(:,:,1)>250 & IM(:,:,2)<5 & IM(:,:,3)<5);

% sea class: yellow
Y_log=(IM(:,:,1)>250 & IM(:,:,2)>250 & IM(:,:,3)<5);

% sky class: green
G_log=(IM(:,:,1)<5 & IM(:,:,2)>123 & IM(:,:,2)<133 & IM(:,:,3)<5);

%% initial conditions
% assemble initial conditions
U0=zeros(n,4);

% known labels
U0(:,1)=reshape(B_log,[n,1]); U0(:,2)=reshape(R_log,[n,1]);
U0(:,3)=reshape(Y_log,[n,1]); U0(:,4)=reshape(G_log,[n,1]);

% unknown labels
for i=1:n
    if(sum(U0(i,:))==0)
        U0(i,:)=[1/4,1/4,1/4,1/4];
    end
end

%% NFFT-opts
% color layer
opts1.sigma=1;                  %The scaling parameter in the Gaussian kernel.
opts1.diagonalEntry=0;          %The value on the diagonal of the adjacency matrix, i.e. the weight assigned to all graph loops.
opts1.N=64;                     %The NFFT band width, should be a power of 2.
opts1.m=5;                      %The NFFT window cutoff parameter.
opts1.eps_B=1/16;               %The NFFT regularization length.
opts1.p=5;                      %The NFFT regularization degree.
opts1.N_oversampling=2*opts1.N;  %The NFFT band width for oversampling.
 
% coordinate layer
opts2.sigma=4;                  %The scaling parameter in the Gaussian kernel.
opts2.diagonalEntry=0;          %The value on the diagonal of the adjacency matrix, i.e. the weight assigned to all graph loops.
opts2.N=64;                     %The NFFT band width, should be a power of 2.
opts2.m=5;                      %The NFFT window cutoff parameter.
opts2.eps_B=1/16;               %The NFFT regularization length.
opts2.p=5;                      %The NFFT regularization degree.
opts2.N_oversampling=2*opts2.N;  %The NFFT band width for oversampling.

%% NFFT-fastsumAdjacencySetup
disp('NFFT-based fast summation precompute');
tic
S{1} = fastsumAdjacencySetup(data, opts1);
S{2} = fastsumAdjacencySetup(coords, opts2);
toc

%% eigeninformation computation
% case p=1: see beginning of Section 4.3

k=12; %number of eigenpairs

disp('Lanczos using NFFT-based fast summation for eigenpairs computation');
tic
[phi,mu]=eigs(@(x)MV_fastsum_T_layers(x,S),n,k,'lm'); 
toc
lambda=eye(k)-mu;

%% Allen-Cahn
% Allen-Cahn-parameter definition
omega0=10000;
epsilon=5.0e-03;
dt=0.01;
c=omega0+3/epsilon;
tolit=1.0e-10; 

% iteration
max_iter_array=[0 50 200 1000];

figure

for iiter=1:4
  max_iter = max_iter_array(iiter);
  fprintf('Allen-Cahn max_iter = %d\n', max_iter);
  tic
  [U,it]=convexity_splitting_vector_modified_fast(U0,lambda,phi,omega0,epsilon,dt,c,max_iter,tolit);
  toc
  
  IM_class = uint8(255*ones(size(IM_full)));
  cut_off = 0.66;
  for ix=1:nx
      for iy=1:ny
          val = U((iy-1)*nx+ix,:);
          val_s = sort(val);
          [vv,ii] = max(val);
          if (vv <= cut_off)
              IM_class(ix,iy,:) = uint8([0 0 0]);
              IM_class(ix,iy,:) = uint8([255 255 255]);
          else
              if (ii==1)
                  IM_class(ix,iy,:) = uint8([0,0,255]);
              elseif (ii==2)
                  IM_class(ix,iy,:) = uint8([255,0,0]);
              elseif (ii==3)
                  IM_class(ix,iy,:) = uint8([255,255,0]);
              elseif (ii==4)
                  IM_class(ix,iy,:) = uint8([0,255,0]);
              end
          end
      end
  end

  if isempty(it)
    it = 0;
  end

  subplot(2,2,iiter);
  image(IM_class);
  title(sprintf('Allen-Cahn iterations: %d', it));
  filename_out = sprintf('beach_eighth_evolution_w_it%d.png', it);
  imwrite(IM_class,filename_out);
  fprintf('\n');
end

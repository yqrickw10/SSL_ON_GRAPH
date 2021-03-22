function [lambda,phi,time_eigs,pavia_labels_vec_reduced,labels,n_class] = Pavia_center_run_eigs(bands,use_xy,sigma1,sigma2,k,p)
%PAVIA_CENER_RUN_EIGS Computes eigenpairs for Pavia center data set using specified bands
%
% Helper function for numerical experiments on the Pavia center data set [2]
% for the Allen-Cahn multiclass classification scheme [1, Algorithm 6.1]
% using the power mean Laplacian [3]
% using the NFFT-based fast summation [4] for the eigeninformation computations [5]
%
% [1] Kai Bergermann, Martin Stoll, and Toni Volkmer. Semi-supervised Learning for Multilayer Graphs Using Diffuse Interface Methods and Fast Matrix Vector Products. Submitted, 2020. 
% [2] A. Plaza, J. A. Benediktsson, J. W. Boardman, J. Brazile, L. Bruzzone, G. Camps-Valls, J. Chanussot, M. Fauvel, P. Gamba, A. Gualtieri, et al. Recent advances in techniques for hyperspectral image processing. Remote sensing of environment, 113 (2009), pp. S110-S122.
% [3] Pedro Mercado, Antoine Gautier, Francesco Tudisco, and Matthias Hein. The power mean Laplacian for multilayer graph clustering. In Proceedings of the Twenty-First International Conference on Artificial Intelligence and Statistics, volume 84 of Proceedings of Machine Learning Research, pages 1828-1838, 2018.
% [4] Daniel Potts, Gabriele Steidl, and Arthur Nieslony. Fast convolution with radial kernels at nonequispaced knots. Numerische Mathematik 98 (2014), pp. 329–351.
% [5] Dominik Alfke, Daniel Potts, Martin Stoll, and Toni Volkmer. NFFT meets Krylov methods: Fast matrix-vector products for the graph Laplacian of fully connected networks. Frontiers in Applied Mathematics and Statistics, 4:61, 2018.
%
% This software is distributed under the GNU General Public License v2. See 
% COPYING for the full license text. If that file is not available, see 
% <http://www.gnu.org/licenses/>.
%
% Copyright (c) 2020 Kai Bergermann, Toni Volkmer


%% NFFT-opts
% color layer
opts1.sigma=sigma1;             %The scaling parameter in the Gaussian kernel.
opts1.diagonalEntry=0;          %The value on the diagonal of the adjacency matrix, i.e. the weight assigned to all graph loops.
opts1.N=64;                     %The NFFT band width, should be a power of 2.
opts1.m=3;                      %The NFFT window cutoff parameter.
opts1.eps_B=1/16;               %The NFFT regularization length.
opts1.p=3;                      %The NFFT regularization degree.
opts1.N_oversampling=2*opts1.N;  %The NFFT band width for oversampling.
 
% coordinate layer
opts2.sigma=sigma2;             %The scaling parameter in the Gaussian kernel.
opts2.diagonalEntry=0;          %The value on the diagonal of the adjacency matrix, i.e. the weight assigned to all graph loops.
opts2.N=64;                     %The NFFT band width, should be a power of 2.
opts2.m=3;                      %The NFFT window cutoff parameter.
opts2.eps_B=1/16;               %The NFFT regularization length.
opts2.p=3;                      %The NFFT regularization degree.
opts2.N_oversampling=2*opts2.N;  %The NFFT band width for oversampling.

try
  pavia_data_=load('Data/Pavia.mat');
  pavia_labels_=load('Data/Pavia_gt.mat');
catch err
  warn('Failed to load Pavia center dataset. Please consult the Data/README file.');
  rethrow(err);
end

pavia_labels=double(pavia_labels_.pavia_gt);
m=length(unique(pavia_labels))-1;
n_class=zeros(1,m);
for i=1:m
    n_class(i)=sum(sum(pavia_labels==i));
end

pavia_data=pavia_data_.pavia;
nx=size(pavia_data,1); ny=size(pavia_data,2); n=nx*ny;


%% Coordinate layer
% spatial pixel coordinates
[X,Y]=ndgrid(1:1096, [1:223,605:1096]);
coords=[reshape(X,[n,1]),reshape(Y,[n,1])];

if use_xy
  S = cell(1,length(bands)+1);
else
  S = cell(1,length(bands));
end

data = cell(1,length(bands));
for i=1:length(bands)
  cur_bands = bands{i};
  data{i} = zeros(n,length(cur_bands));
  for j=1:length(cur_bands)
    data{i}(:,j) = reshape(pavia_data(:,:,cur_bands(j)),n,1);
  end
end

% delete 0-class labels (empty)
pavia_labels_vec=reshape(pavia_labels,[n,1]);
ind_remove = find(pavia_labels_vec==0);
for i=1:length(data)
    data_temp=data{i};
    data_temp(ind_remove,:)=[];
    data{i}=data_temp;
end
coords(ind_remove,:)=[];

n_eff = length(coords);

pavia_labels_vec_reduced=pavia_labels_vec;
pavia_labels_vec_reduced(ind_remove,:)=[];

labels=zeros(n_eff,m);
for i=1:n_eff
    labels(i,pavia_labels_vec_reduced(i))=1;
end


% Setup fastsum
% tic
for i=1:length(data)
    S{i}=fastsumAdjacencySetup(data{i},opts1);
end
if use_xy
  S{end}=fastsumAdjacencySetup(coords,opts2);
end
% toc

T = length(S);

tt = tic;
if(p==1)
    [phi,mu]=eigs(@(x)MV_fastsum_T_layers(x,S),n_eff,k,'lm','Display',1);
    lambda=eye(k)-mu;
elseif(p<0)
    max_iter_arnoldi=100;
    tol=1e-6;
    delta=log(1+abs(p));

    Ls_fct=cell(1,T);
    Ls_p_fct=cell(1,T);

    for i=1:T
        Ls_fct{i} = @(x) (1+delta)*x - S{i}.applyNormalizedAdjacency(x); 
        Ls_p_fct{i} = @(v) gen_arnoldi_for_power_of_a_matrix_times_a_vector_modified(@(x) Ls_fct{i}(x), max_iter_arnoldi, v, p, tol);
    end
    [phi,mu]=eigs(@(w) fh_pml(w,Ls_p_fct),n_eff,k,'lm','Display',1); 
    lambda=diag(diag(mu).^(1/p));
else
    error('Choose p=1 or p<0!')
end
time_eigs = toc(tt);

end


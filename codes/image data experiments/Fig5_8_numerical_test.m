% Numerical experiment on image data for multi-class case
% using the power mean Laplacian with a splitting into 2 layers (RGB + XY)
% The code is mainly from this paper:
% [1] Kai Bergermann, Martin Stoll, and Toni Volkmer. Semi-supervised Learning for Multilayer Graphs Using Diffuse Interface Methods and Fast Matrix Vector Products. Submitted, 2020. 

clear all

% add paths.
addpath('../Subroutines')
addpath('../Subroutines/fastsum')

%% k=12 and p=1
k=12; %number of eigenpairs
p=1; %parameter p of power mean Laplacian

%% read image data
IM=imread('Data/epfl_painted.png');
IM_full=imread('Data/epfl.png');

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
% sky: blue
B_log=(IM(:,:,1)<5 & IM(:,:,2)<5 & IM(:,:,3)>250);

% logo: red
R_log=(IM(:,:,1)>250 & IM(:,:,2)<5 & IM(:,:,3)<5);

% building: yellow
Y_log=(IM(:,:,1)>250 & IM(:,:,2)>250 & IM(:,:,3)<5);

% grass: green
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
tic
S{1} = fastsumAdjacencySetup(data, opts1);
S{2} = fastsumAdjacencySetup(coords, opts2);
toc

%% eigeninformation computation

tic
if(p==1)
    [phi,mu]=eigs(@(x)MV_fastsum_T_layers(x,S),n,k,'lm');
    lambda=eye(k)-mu;
elseif(p<0)
    max_iter_arnoldi=100;
    tol=1e-6;
    delta=log(1+abs(p));
    Ls1_fct = @(x) (1+delta)*x - S{1}.applyNormalizedAdjacency(x); 
    Ls2_fct = @(x) (1+delta)*x - S{2}.applyNormalizedAdjacency(x);
    Ls1_p_fct = @(v) gen_arnoldi_for_power_of_a_matrix_times_a_vector_modified(@(x) Ls1_fct(x), max_iter_arnoldi, v, p, tol);
    Ls2_p_fct = @(v) gen_arnoldi_for_power_of_a_matrix_times_a_vector_modified(@(x) Ls2_fct(x), max_iter_arnoldi, v, p, tol);
    [phi,mu]=eigs(@(w) (1/2)*(Ls1_p_fct(w)+Ls2_p_fct(w)),n,k,'lm'); 
    lambda=diag(diag(mu).^(1/p));
else
    error('Choose p=1 or p<0!')
end
toc

%% Allen-Cahn
% Allen-Cahn-parameter definition
omega0=1000;
epsilon=5.0e-03;
dt=0.01;
c=omega0+3/epsilon;
max_iter=500;
tolit=1.0e-06; 

% iteration
tic
[U,it,tol_reached]=convexity_splitting_vector_modified_fast(U0,lambda,phi,omega0,epsilon,dt,c,max_iter,tolit);
toc

%% retrieve solution and plot it
u_sol=zeros(n,4);
[~, I_U] = max(U,[],2);
for i=1:n
  u_sol(i,I_U(i))=1;
end

www = repmat(255,1,3);
figure
subplot(221)
part_grass=IM_full;
u_sol1 = reshape(u_sol(:,1),nx,ny);
for ix=1:nx
  for iy=1:ny
    if ~u_sol1(ix,iy)
      part_grass(ix,iy,:) = www;
    end
  end
end
image(part_grass);
title('sky')% B
imwrite(part_grass,'B_sky.png');
%
subplot(222)
part_log=IM_full;
u_sol2 = reshape(u_sol(:,2),nx,ny);
for ix=1:nx
  for iy=1:ny
    if ~u_sol2(ix,iy)
      part_log(ix,iy,:) = www;
    end
  end
end
image(part_log);
title('logo')% R
imwrite(part_log,'B_logo.png');
%
subplot(223)
part_building=IM_full;
u_sol3 = reshape(u_sol(:,3),nx,ny);
for ix=1:nx
  for iy=1:ny
    if ~u_sol3(ix,iy)
      part_building(ix,iy,:) = www;
    end
  end
end
image(part_building);
title('building')% Y
imwrite(part_building,'B_building.png');
%
subplot(224)
part_grass=IM_full;
u_sol4 = reshape(u_sol(:,4),nx,ny);
for ix=1:nx
  for iy=1:ny
    if ~u_sol4(ix,iy)
      part_grass(ix,iy,:) = www;
    end
  end
end
image(part_grass);
title('grass')% G
imwrite(part_grass,'B_grass.png');

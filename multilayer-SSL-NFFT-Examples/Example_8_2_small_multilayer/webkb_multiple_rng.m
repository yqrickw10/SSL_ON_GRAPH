% Numerical experiment on the single layer WebKB data set [2]
% for the Allen-Cahn multiclass classification scheme [1, Algorithm 6.1]
% using the power mean Laplacian [3].
%
% [1] Kai Bergermann, Martin Stoll, and Toni Volkmer. Semi-supervised Learning for Multilayer Graphs Using Diffuse Interface Methods and Fast Matrix Vector Products. Submitted, 2020. 
% [2] M. Craven, A. McCallum, D. PiPasquo, T. Mitchell, and D. Freitag, Learning to extract symbolic knowledge from the world wide web. Tech. rep., Carnegie-mellon univ pittsburgh pa school of computer Science, 1998.
% [3] Pedro Mercado, Antoine Gautier, Francesco Tudisco, and Matthias Hein. The power mean Laplacian for multilayer graph clustering. In Proceedings of the Twenty-First International Conference on Artificial Intelligence and Statistics, volume 84 of Proceedings of Machine Learning Research, pages 1828-1838, 2018.
%
% This software is distributed under the GNU General Public License v2. See 
% COPYING for the full license text. If that file is not available, see 
% <http://www.gnu.org/licenses/>.
%
% Copyright (c) 2019-2020 Kai Bergermann, 2020 Toni Volkmer

clear all

% ratio of known labels
ratio_known_array=[0.01 0.05:0.05:0.25];

% parameter p for power mean
p_array=[1 -1 -10];

% Choose the number of eigenpairs
k=15; 

addpath('../Subroutines')

%% load and prepare data
load Data/webKB_texas_2.mat

fprintf('WebKB dataset k = %3d, mean error in percent\n', k);
fprintf('============================================\n');
fprintf('known  ');
for rk = ratio_known_array
  fprintf(' | %3d%%', round(rk*100));
end
fprintf('\n');

T=1;
X1=data{1}';
n=size(X1,1);

%% prepare labels
Y=truelabel{1};
m=5; 

%bring labels in matrix form
Y_mat=zeros(n,m);
for i=1:n
    Y_mat(i,Y(i))=1;
end

%% Compute pairwise euclidean distances between data points
S1 = dist2(X1,X1);
scale = 2;

%% Compute the weight matrix W
W1 = exp(-S1/(scale^2));
W1 = W1.*~eye(size(W1));
D1invsqrt=diag(sum(W1).^-0.5);

for p = p_array
fprintf('-------------------------------------------------\n');
    fprintf('p = %3d', p);
    %% create L_p and compute the first k eigenpairs
tic;
    delta=log(1+abs(p));
    if(p==1)
        L1=eye(n)-D1invsqrt*W1*D1invsqrt;
        Lp=(1/T)*(L1);
    else
        L1=(1+delta)*eye(n)-D1invsqrt*W1*D1invsqrt;
        Lp=((1/T)*(L1^p))^(1/p);
    end
    
    [phi,lambda]=eigs(Lp,k,-1); % 'sm' may cause warning/cause error here
    
runtime_eig = toc;
ii=0;
mean_runtimes = zeros(length(ratio_known_array),1);
    for ratio_known = ratio_known_array
ii=ii+1;
        %% Allen-Cahn
        n_rng=10;
        accuracy=zeros(1,n_rng);
        for j=1:n_rng
            s = RandStream('mcg16807','Seed',j); RandStream.setGlobalStream(s);
            idxSample         = sample_idx_per_class(Y, ratio_known, 'percentage');
            U0                 = (1/m)*ones(n,m);
            for i=1:length(idxSample)
                U0(idxSample(i),:)=zeros(1,m);
                U0(idxSample(i),Y(idxSample(i)))=1;
            end
            
            % Allen-Cahn-parameter definition
            omega0=1000;
            epsilon=5.0e-03;
            dt=0.01;
            c=omega0+3/epsilon;
            max_iter=300;
            tolit=1.0e-06;
            
tic;            
            % iteration
            [U,it]=convexity_splitting_vector_modified_fast(U0,lambda,phi,omega0,epsilon,dt,c,max_iter,tolit);
mean_runtimes(ii) = mean_runtimes(ii) + toc;
            
            % retrieve solution and calculate accuracy
            U_sol=zeros(n,m);
            [~, I_U] = max(U,[],2);
            for i=1:n
                U_sol(i,I_U(i))=1;
            end
            accuracy(j)=sum(all(U_sol==Y_mat,2))/n;
        end
        
mean_runtimes(ii) = mean_runtimes(ii) / n_rng;        
        mean_accuracy=mean(accuracy);
        mean_error=1-mean_accuracy;
        fprintf(' | %4.1f', mean_error*100);
    end
    fprintf('\n');
fprintf('time eigs: %.3f\n', runtime_eig);
fprintf('time ac');
for ii=1:length(ratio_known_array); fprintf(' | %4.2f', mean_runtimes(ii)); end; fprintf('\n');
end

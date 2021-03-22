% Numerical experiment on the 6-layer UCI data set [2]
% for the Allen-Cahn multiclass classification scheme [1, Algorithm 6.1]
% using the power mean Laplacian [3].
%
% [1] Kai Bergermann, Martin Stoll, and Toni Volkmer. Semi-supervised Learning for Multilayer Graphs Using Diffuse Interface Methods and Fast Matrix Vector Products. Submitted, 2020. 
% [2] M. van Breukelen, R. P. W. Duin, D. M. J. Tax, and J. E. den Hartog. Handwritten digit recognition by combined classifiers. Kybernetika, 34 (1998), pp. 381-386.
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
k=98; 

addpath('../Subroutines')

%% load and prepare data
load Data/UCI_mfeat.mat

fprintf('UCI dataset k = %3d, mean error in percent\n', k);
fprintf('==========================================\n');
fprintf('known  ');
for rk = ratio_known_array
  fprintf(' | %3d%%', round(rk*100));
end
fprintf('\n');

T=6;
X1=data{1}'; X2=data{2}'; X3=data{3}'; X4=data{4}'; X5=data{5}'; X6=data{6}'; 
n=size(X1,1); 

%% prepare labels
Y=truelabel{1};
m=size(unique(Y),1); 

%bring labels in matrix form
Y_mat=zeros(n,m);
for i=1:n
    Y_mat(i,Y(i)+1)=1;
end

%% Compute pairwise euclidean distances between data points
S1 = dist2(X1,X1);
S2 = dist2(X2,X2);
S3 = dist2(X3,X3);
S4 = dist2(X4,X4);
S5 = dist2(X5,X5);
S6 = dist2(X6,X6);
scale = 10;

%% Compute the weight matrix W
W1 = exp(-S1/(scale^2));
W1 = W1.*~eye(size(W1));
W2 = exp(-S2/(scale^2));
W2 = W2.*~eye(size(W2));
W3 = exp(-S3/(scale^2));
W3 = W3.*~eye(size(W3));
W4 = exp(-S4/(scale^2));
W4 = W4.*~eye(size(W4));
W5 = exp(-S5/(scale^2));
W5 = W5.*~eye(size(W5));
W6 = exp(-S6/(scale^2));
W6 = W6.*~eye(size(W6));
D1=diag(sum(W1));
D2=diag(sum(W2));
D3=diag(sum(W3));
D4=diag(sum(W4));
D5=diag(sum(W5));
D6=diag(sum(W6));

for p = p_array
fprintf('-------------------------------------------------\n');
    fprintf('p = %3d', p);
    %% create L_p and compute the first k eigenpairs
tic;
    L1=zeros(n,n); L2=zeros(n,n); L3=zeros(n,n);
    L4=zeros(n,n); L5=zeros(n,n); L6=zeros(n,n);
    
    delta=log(1+abs(p));
    if(p==1) % D singular!
        for i=1:n
            if(D1(i,i)==0) % isolated nodes
                L1(i,i)=0;
            else
                for j=1:n
                    if(D1(j,j)==0)
                        L1(i,j)=0;
                    else
                        L1(i,j)=-W1(i,j)/(sqrt(D1(i,i))*(sqrt(D1(j,j))));
                    end
                end
                L1(i,i)=1;
            end
        end
        for i=1:n
            if(D2(i,i)==0) % isolated nodes
                L2(i,i)=0;
            else
                for j=1:n
                    if(D2(j,j)==0)
                        L2(i,j)=0;
                    else
                        L2(i,j)=-W2(i,j)/(sqrt(D2(i,i))*(sqrt(D2(j,j))));
                    end
                end
                L2(i,i)=1;
            end
        end
        for i=1:n
            if(D3(i,i)==0) % isolated nodes
                L3(i,i)=0;
            else
                for j=1:n
                    if(D3(j,j)==0)
                        L3(i,j)=0;
                    else
                        L3(i,j)=-W3(i,j)/(sqrt(D3(i,i))*(sqrt(D3(j,j))));
                    end
                end
                L3(i,i)=1;
            end
        end
        for i=1:n
            if(D4(i,i)==0) % isolated nodes
                L4(i,i)=0;
            else
                for j=1:n
                    if(D4(j,j)==0)
                        L4(i,j)=0;
                    else
                        L4(i,j)=-W4(i,j)/(sqrt(D4(i,i))*(sqrt(D4(j,j))));
                    end
                end
                L4(i,i)=1;
            end
        end
        for i=1:n
            if(D5(i,i)==0) % isolated nodes
                L5(i,i)=0;
            else
                for j=1:n
                    if(D5(j,j)==0)
                        L5(i,j)=0;
                    else
                        L5(i,j)=-W5(i,j)/(sqrt(D5(i,i))*(sqrt(D5(j,j))));
                    end
                end
                L5(i,i)=1;
            end
        end
        for i=1:n
            if(D6(i,i)==0) % isolated nodes
                L6(i,i)=0;
            else
                for j=1:n
                    if(D6(j,j)==0)
                        L6(i,j)=0;
                    else
                        L6(i,j)=-W6(i,j)/(sqrt(D6(i,i))*(sqrt(D6(j,j))));
                    end
                end
                L6(i,i)=1;
            end
        end
        Lp=(1/T)*(L1+L2+L3+L4+L5+L6);
    else
        for i=1:n
            if(D1(i,i)==0) % isolated nodes
                L1(i,i)=delta;
            else
                for j=1:n
                    if(D1(j,j)==0)
                        L1(i,j)=0;
                    else
                        L1(i,j)=-W1(i,j)/(sqrt(D1(i,i))*(sqrt(D1(j,j))));
                    end
                end
                L1(i,i)=1+delta;
            end
        end
        for i=1:n
            if(D2(i,i)==0) % isolated nodes
                L2(i,i)=delta;
            else
                for j=1:n
                    if(D2(j,j)==0)
                        L2(i,j)=0;
                    else
                        L2(i,j)=-W2(i,j)/(sqrt(D2(i,i))*(sqrt(D2(j,j))));
                    end
                end
                L2(i,i)=1+delta;
            end
        end
        for i=1:n
            if(D3(i,i)==0) % isolated nodes
                L3(i,i)=delta;
            else
                for j=1:n
                    if(D3(j,j)==0)
                        L3(i,j)=0;
                    else
                        L3(i,j)=-W3(i,j)/(sqrt(D3(i,i))*(sqrt(D3(j,j))));
                    end
                end
                L3(i,i)=1+delta;
            end
        end
        for i=1:n
            if(D4(i,i)==0) % isolated nodes
                L4(i,i)=delta;
            else
                for j=1:n
                    if(D4(j,j)==0)
                        L4(i,j)=0;
                    else
                        L4(i,j)=-W4(i,j)/(sqrt(D4(i,i))*(sqrt(D4(j,j))));
                    end
                end
                L4(i,i)=1+delta;
            end
        end
        for i=1:n
            if(D5(i,i)==0) % isolated nodes
                L5(i,i)=delta;
            else
                for j=1:n
                    if(D5(j,j)==0)
                        L5(i,j)=0;
                    else
                        L5(i,j)=-W5(i,j)/(sqrt(D5(i,i))*(sqrt(D5(j,j))));
                    end
                end
                L5(i,i)=1+delta;
            end
        end
        for i=1:n
            if(D6(i,i)==0) % isolated nodes
                L6(i,i)=delta;
            else
                for j=1:n
                    if(D6(j,j)==0)
                        L6(i,j)=0;
                    else
                        L6(i,j)=-W6(i,j)/(sqrt(D6(i,i))*(sqrt(D6(j,j))));
                    end
                end
                L6(i,i)=1+delta;
            end
        end
        Lp=((1/T)*(L1^p+L2^p+L3^p+L4^p+L5^p+L6^p))^(1/p);
    end
    
    
    [phi,lambda]=eigs(Lp,k,'sm');
    
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
                U0(idxSample(i),Y(idxSample(i))+1)=1; %class numbers start at 0...
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

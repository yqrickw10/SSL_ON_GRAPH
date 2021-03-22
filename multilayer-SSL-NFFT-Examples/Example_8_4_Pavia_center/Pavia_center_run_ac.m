function [accuracy_total,accuracy_classwise,U_sol,it,time_ac] = Pavia_center_run_ac(pavia_labels_vec_reduced,perc_known_labels,lambda,phi,labels,n_class)
%PAVIA_CENTER_RUN_AC Runs Allen-Cahn convexity splitting for Pavia center data set
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

m=9;
n_eff = 148152;
idxSample         = sample_idx_per_class(pavia_labels_vec_reduced, perc_known_labels, 'percentage');
U0                 = (1/m)*ones(n_eff,m);
for i=1:length(idxSample)
    U0(idxSample(i),:)=zeros(1,m);
    U0(idxSample(i),pavia_labels_vec_reduced(idxSample(i)))=1;
end

%% Allen-Cahn
% Allen-Cahn-parameter definition
omega0=10000;
epsilon=5.0e-01;
dt=0.01;
c=omega0+3/epsilon;
max_iter=300;
tolit=1.0e-06; 

% iteration
tt = tic;
[U,it]=convexity_splitting_vector_modified_fast(U0,lambda,phi,omega0,epsilon,dt,c,max_iter,tolit);
time_ac = toc(tt);

% retrieve solution and plot it
U_sol=zeros(n_eff,m);
[~, I_U] = max(U,[],2);
counter=0;
counter_class=zeros(1,m);
for i=1:n_eff
    U_sol(i,I_U(i))=1;
    if all(U_sol(i,:)==labels(i,:))
        counter=counter+1;
        counter_class(I_U(i))=counter_class(I_U(i))+1;
    end
end


accuracy_total=counter/n_eff;
accuracy_classwise=counter_class./n_class;

end


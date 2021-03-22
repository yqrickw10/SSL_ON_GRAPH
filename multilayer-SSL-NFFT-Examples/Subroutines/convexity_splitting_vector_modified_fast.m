function [u,it,tol_reached] = convexity_splitting_vector_modified_fast(u_0, lambda, phi,omega_0,epsilon,dt,c,MAX_ITER,tolit)
% segmentation of an image using the Allen-Cahn model with a smooth potential and convexity splitting
%
% see: [1] C. Garcia-Cardona, E. Merkurjev, A.L. Bertozzi, A. Flenner, A.
% Percus, "Multiclass Data Segmentation using Diffuse Interface Methods on
% Graphs", IEEE, 2014.
%
% Input:
% u_0:      image with predifend points, the supervised part
% lambda:   k eigenvalues of the Graph Laplacian
% phi:      k eigenvectors of the Graph Laplacian
%
% Output:
% u:        segmented image
%
% Stoll/Bosch 2016
%
% Created on: 25.05.2016
%     Author: Jessica Bosch
% 
% 
% Modified: 	Kai Bergermann, 2019
% 		removed random initialization of u_0. Use uniformly distributed u_0 instead.
%               Toni Volkmer , 2020
%               vectorized some code for speedup


%% set parameters
% omega_0=10000;              % fidelity parameter
% epsilon = 1;                % interface parameter
% c = (2/epsilon)+omega_0;    % convexity parameter
% dt = 0.01;                  % time step size
% MAX_ITER = 1000;            % max. number of time steps
% tolit = 1e-5;               % stopping tolerance

lambda = diag(lambda);      % desired eigenvalues
k = size(lambda,1);         % k=number of desired eigenvalues
[n,N] = size(u_0);          % n=number of unknowns, N=number of phases

% fidelity matrix for the fidelty term
omega = zeros(n,1); 
for i=1:n
    if (u_0(i,1)>(1/N) || u_0(i,1)<(1/N))
        omega(i,1)=omega_0;
    end
end

u=u_0;

%% algorithm according to [p. 1605, 1]

Y=zeros(k,n);
for j=1:k
    Y(j,:) = (phi(:,j)/(1 + dt*(epsilon*lambda(j) + c)))';
end

% loop over time
for it = 1:MAX_ITER
    % modified smooth potential (using L1 norm)
    % see [equ. (20), 1]
%     T=zeros(n,N);
% 
%     for i=1:n
%         norm_u=norm(u(i,:),1);
%         for j=1:N
%             for l=1:N
%                 tt_norm = 1;
%                 for m=1:N 
%                     if (m~=l)
%                         tt_norm=tt_norm*0.25*(norm_u-abs(u(i,m))+abs(u(i,m)-1))^(2);
%                     end
%                 end
%                 T(i,j)=T(i,j)+0.5*(1-2*double(j==l))*(norm_u-abs(u(i,l))+abs(u(i,l)-1))*tt_norm;
%             end
%         end
%     end
%     clear i;
    T=zeros(n,N);
    norm_u_vector = sum(abs(u),2);
%     for i=1:n
        for j=1:N
            for l=1:N
                tt_norm = 1;
                for m=1:N 
                    if (m~=l)
                        tt_norm=tt_norm*0.25.*(norm_u_vector-abs(u(:,m))+abs(u(:,m)-1)).^(2);
                    end
                end
                T(:,j)=T(:,j)+0.5*(1-2*double(j==l))*(norm_u_vector-abs(u(:,l))+abs(u(:,l)-1)).*tt_norm;
            end
        end
%     end
    
 
    % fidelity part
    Fid=u-u_0;
%     for j=1:n
%         Fid(j,:)=Fid(j,:)*omega(j);
%     end   
    Fid = Fid .* repmat(omega,1,N);
    
    Z=Y*((1+c*dt)*u-(dt/(2*epsilon))*T-dt*Fid);    
    
    u_new=phi*Z;
%     u_new2 = u_new;
% 
%     % project the solution back to the Gibbs simplex
%     for j=1:n
%         u_new(j,:)=projspx(u_new(j,:));
%     end  

    % project the solution back to the Gibbs simplex
    u_new = projspx_fast(u_new);
    
    % norm for stopping criterion
    norm_diff = sum((u_new-u).^2,2);
    norm_new = sum((u_new).^2,2);
%     norm_diff=zeros(n,1);
%     norm_new=zeros(n,1);
%     for j=1:n
%         norm_diff(j)=norm((u_new(j,:)-u(j,:))')^(2);
%         norm_new(j)=norm((u_new(j,:))')^(2);
%     end
   
    % update old solution
    u=u_new;

    % test stopping criterion
%     max(norm_diff)/max(norm_new)
    tol_reached = max(norm_diff)/max(norm_new);
    if (tol_reached<tolit)
        break;
    end    

end

end


function projy=projspx(y)

% projection of y onto the Gibbs simplex
% see: Y. Chen and X. Ye, "Projection onto A Simplex", arXiv preprint, 2011

    n=length(y);
    y_sort=sort(y);
    th_set=0;
    
    for i=(n-1):-1:1
    
        ti=0;
        for j=(i+1):n
            ti=ti+y_sort(j);
        end    
        
        if ((ti-1)/(n-i)>=y_sort(i))
            th=(ti-1)/(n-i);
            th_set=1;
            break;
        end
        
    end
    
    if (th_set < 0.5)
        th=0;
        for j=1:n
            th=th+y(j);
        end
        th=(th-1)/n;
    end
    
    projy=y-th;
    for j=1:n
        if (projy(j)<0)
            projy(j)=0;
        end
    end
            
end

function projy=projspx_fast(y_matrix)

% projection of y onto the Gibbs simplex
% see: Y. Chen and X. Ye, "Projection onto A Simplex", arXiv preprint, 2011

    n=size(y_matrix,2);
%     y_sort=sort(y);
    y_matrix_sort = sort(y_matrix,2);

    th=zeros(size(y_matrix,1),1);
    th_set=zeros(size(y_matrix,1),1);
    
    for i=(n-1):-1:1
        ti = sum(y_matrix_sort(:,i+1:n),2);
%         ti = sum(y_sort(i+1:n));
%         ti=0;
%         for j=(i+1):n
%             ti=ti+y_sort(j);
%         end    
        ind = find((th_set==0) .* ((ti-1)/(n-i)>=y_matrix_sort(:,i)));
        th(ind)=(ti(ind)-1)/(n-i);
        th_set(ind)=1;

%         if ((ti-1)/(n-i)>=y_sort(i))
%             th=(ti-1)/(n-i);
%             th_set=1;
%             break;
%         end
        
    end

    ind = find(th_set < 0.5);
    th(ind) = (sum(y_matrix(ind,:),2)-1)/n;
%     if (th_set < 0.5)
%       th = (sum(y)-1)/n;
% %         th=0;
% %         for j=1:n
% %             th=th+y(j);
% %         end
% %         th=(th-1)/n;
%     end
    
    projy=y_matrix-repmat(th,1,n);
    projy(projy < 0) = 0;
%     for j=1:n
%         if (projy(j)<0)
%             projy(j)=0;
%         end
%     end
            
end

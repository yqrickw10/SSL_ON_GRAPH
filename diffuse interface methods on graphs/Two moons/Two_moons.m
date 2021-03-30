%Author: Yuqi Wang (yuqi.wang@epfl.ch)
%Created Date: 2021/03/25
%Matlab Version of Diffuse Interface Method on data

%read data
%Note: data is created in Gen_two_moons.m
load('2Moons_v2.mat');

%plot two moons data
% figure, 
% plot(X(Y==1,1),X(Y==1,2),'bo')
% hold on 
% plot(X(Y==2,1),X(Y==2,2),'ro')


%create graph
%create K-nearest neighbor graph
%we use KNN algorithmm

n_samples = 2000;
K = 10; %this his not the number of eigenvectors but the number of KNN 
IDX = zeros(2000,10);
Sigma = zeros(2000,1);
%idx = knnsearch(X(Reference,:),X(1,:),'K',10);

%create graph
for index=1:2000
    %find K-nearest neighbor
    idx = knnsearch(X,X(index,:),'K',K+1);
    IDX(index,:) = idx(2:end);
    Sigma(index) = norm(X(index,:)-X(IDX(index,K),:));
end

Weight = zeros(2000,2000);
for x=1:2000
    for y=1:K
        weight = exp(-norm(X(x,:)-X(IDX(x,y),:))^2/(Sigma(x)*Sigma(IDX(x,y))));
        Weight(x,IDX(x,y)) = weight;
        Weight(IDX(x,y),x) = weight;        
    end
end

%create degree matrix
degree_vec = zeros(n_samples,1);
for x=1:n_samples
    degree_vec(x) = sum(Weight(x,:));
end
Degree = diag(degree_vec);
%Weight = Weight;

Laplacian = Degree - Weight;
D_sqrt_inv = pinv(sqrtm(Degree));
sym_Laplacian = eye(n_samples)-D_sqrt_inv*Weight*D_sqrt_inv;

%eigendecomposition
%[V,D] = eig(Laplacian);
[V,D] = eig(sym_Laplacian);

%only keeps the real values
V = real(V);
D = real(D);


%set experiments parameters
c = 2;
epsilon =0.25;
dt = 1;
M = 500;
n_vectors = 20;%number of vector

%initialization
u = zeros(n_samples,M+1);
u(:,1) = sign(V(:,2)-mean(V(:,2)));
%u(:,1) = (u(:,1)-mean(u(:,1)))/abs(max(u(:,1)));

%without any fidelity terms
%convex splitting

a=zeros(n_vectors,M+1);
b=zeros(n_vectors,M+1);
d=zeros(n_vectors,M+1);
D1 = zeros(n_vectors,1);

for k=1:n_vectors
    a(k,1) = dot(u(:,1),V(:,k));
    b(k,1) = dot(u(:,1).^3,V(:,k));
    d(k,1) = 0;
    D1(k) = 1+dt*(epsilon*D(k,k)+c);    
end

for x=1:M
    for k=1:n_vectors
        a_next = 1/D1(k)*((1+dt/epsilon+c*dt)*a(k,x)-dt/epsilon*b(k,x)-dt*d(k,x));
        %update a
        a(k,x+1) = a_next;
    end
    
    u(:,x+1)= sum(a(:,x+1)'.*V(:,1:n_vectors),2);
    
    
    %constraint
    %u(:,x+1) = (u(:,x+1) - mean(u(:,x+1)))/abs(max(u(:,x+1)));
    %how to constraint?
    

    %u(:,x+1) = u(:,x+1) - mean(u(:,x+1));
    %u(:,x+1)= normalize(u(:,x+1));
    
    for k=1:n_vectors
        b_next = dot(u(:,x+1).^3,V(:,k));
        d_next = 0;
        %update
        b(k,x+1) = b_next;
        d(k,x+1) = d_next;        
    end
    
end

predict = sign(u(:,end));
%predict = sign(u(:,1));
predict(predict==1)=2;
predict(predict==-1)=1;

accuracy = sum(predict==Y)/n_samples;
disp("Accuracy:");
disp(accuracy);

%plot two moons data
figure;
plot(X(predict==1,1),X(predict==1,2),'bo');
hold on;
plot(X(predict==2,1),X(predict==2,2),'ro');











%Author: Yuqi Wang (yuqi.wang@epfl.ch)
%Created Date: 2021/03/11
%Matlab Version of Diffuse Interface Method on data

%Need to work on: using graph representation in the matlab

%Read Data
data = readtable('house-votes-84.csv','ReadVariableNames',false);


%set parameters
tau = 0.3;


c = 1;
dt = 0.1;
epsilon = 2;
M = 500; %number of iterations

lambda_known = 1;
lambda_unknown = 0;

%create fully connected graph by using adjacency matrix representation
votes = data(:,2:end);
votes_array = table2array(votes);

%replace values
votes_array = strrep(votes_array,'y','1');
votes_array = strrep(votes_array,'n','-1');
votes_array = strrep(votes_array,'?','0');
votes_mat = str2double(votes_array);
[n,m] = size(votes_mat);

%create weigt matrix
Weight = zeros(n,n);
for x=1:n
    for y=x:n
        %calculate weight
        Weight(x,y) = exp(-norm(votes_mat(x,:)-votes_mat(y,:))^2/tau);
        Weight(y,x) = Weight(x,y);
    end
end

%create degree matrix
degree_vec = zeros(n,1);
for x=1:n
    degree_vec(x) = sum(Weight(x,:));
end
Degree = diag(degree_vec);

Laplacian = Degree - Weight;
D_sqrt_inv = sqrtm(Degree);
sym_Laplacian = D_sqrt_inv*Laplacian*D_sqrt_inv;

%eigendecomposition
[V,D] = eig(sym_Laplacian);

%only keeps the real values
V = real(V);
D = real(D);


%initialization
u=zeros(n,M+1);
u(1,1)=-1;u(2,1)=-1;u(3,1)=1;u(4,1)=1;u(5,1)=1;

lambda = zeros(n,1);
lambda(1)=1;lambda(2)=1;lambda(3)=1;lambda(4)=1;lambda(5)=1;

%convex splitting

%the vectors store the final value
a=zeros(n,1);
b=zeros(n,1);
d=zeros(n,1);
D1 = zeros(n,1);

%intialization
for k=1:20
    a(k) = dot(u(:,1),V(:,k));
    b(k) = dot(u(:,1).^3,V(:,k));
    d(k) = 0;
    D1(k) = 1+dt*(epsilon*D(k,k)+c);
end

for k=1:20
    %iterate
    for x=1:M
        a_next = 1/D1(k)*((1+dt/epsilon+c*dt)*a(k)-dt/epsilon*b(k)-dt*d(k));
        %update a
        a(k) = a_next;
        u(:,x+1)= sum(a.*V,2);
        b_next = dot(u(:,x+1).^3,V(:,k));
        d_next = dot(lambda.*(u(:,x+1)-u(:,1)),V(:,k));
        %update
        b(k) = b_next;
        d(k) = d_next;
    end
end












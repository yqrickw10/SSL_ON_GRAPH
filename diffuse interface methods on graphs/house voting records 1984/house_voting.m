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

%n is the number of datapoints, m is the number of features

%create weigt matrix
Weight = zeros(n,n);
for x=1:n
    for y=x:n
        %calculate weight
        if x~=y
            Weight(x,y) = exp(-norm(votes_mat(x,:)-votes_mat(y,:))^2/tau);
            Weight(y,x) = Weight(x,y);
        else
            Weight(x,y) = 0;
        end
    end
end

%create degree matrix
degree_vec = zeros(n,1);
for x=1:n
    degree_vec(x) = sum(Weight(x,:));
end
Degree = diag(degree_vec);

Laplacian = Degree - Weight;
D_sqrt_inv = pinv(sqrtm(Degree));
sym_Laplacian = eye(n)-D_sqrt_inv*Weight*D_sqrt_inv;

%eigendecomposition
%[V,D] = eig(Laplacian);
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
a=zeros(n,M+1);
b=zeros(n,M+1);
d=zeros(n,M+1);
D1 = zeros(n,1);




%intialization
for k=1:n
    a(k,1) = dot(u(:,1),V(:,k));
    b(k,1) = dot(u(:,1).^3,V(:,k));
    d(k,1) = 0;
    D1(k) = 1+dt*(epsilon*D(k,k)+c);
end

for x=1:M
    for k=1:n
        a_next = 1/D1(k)*((1+dt/epsilon+c*dt)*a(k,x)-dt/epsilon*b(k,x)-dt*d(k,x));
        %update a
        a(k,x+1) = a_next;
        b_next = dot(u(:,x+1).^3,V(:,k));
        d_next = dot(lambda.*(u(:,x+1)-u(:,1)),V(:,k));
        %update
        b(k,x) = b_next;
        d(k,x) = d_next;
    end
    u(:,x+1)= sum(a(:,x+1).*V,2);
end


class_data = data(:,1);
class_array = table2array(class_data);
class_array = strrep(class_array,'republican','-1');
class_array = strrep(class_array,'democrat','1');
class = str2double(class_array);

%accuracy
accuracy = sum(class==sign(u(:,end)))/n;










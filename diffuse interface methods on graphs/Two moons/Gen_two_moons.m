% 20210325
% two-moon dataset generator

center1_x = 0;
center1_y = 0;
center2_x = 1;
center2_y = 0.5;
nb_sample = 2000;
radius = 1;
theta = linspace(0,pi,nb_sample);
noise = randn(1,nb_sample)*0.15;
%noise = rand(1,nb_sample)*0.6;
semi_up = [(radius+noise).*cos(theta) + center1_x;(radius+noise).*sin(theta)+center1_y];
semi_down = [(radius+noise).*cos(-1*theta) + center2_x; (radius+noise).*sin(-1*theta)+center2_y];
figure;
plot(semi_up(1,:),semi_up(2,:),'bo')
hold on
plot(semi_down(1,:),semi_down(2,:),'ro')

x = [semi_up,semi_down]';
y = [ones(length(semi_up),1);-1*ones(length(semi_down),1)];
idx = randperm(2*nb_sample,2*nb_sample);
xt = x(idx(1:nb_sample),:);
yt = y(idx(1:nb_sample));
x = x(idx(nb_sample+1:end),:);
y = y(idx(nb_sample+1:end));

% theta = 0;
% theta = theta*pi/180;
% rotation = [  [cos(theta), sin(theta)]; [-sin(theta), cos(theta)] ];
% x = x * rotation;
% xt = xt * rotation;

dataMean = mean(x);
X = 2*bsxfun(@minus, x, dataMean);
X_adapt = 2*bsxfun(@minus, xt, dataMean);
% theta = 0;
% theta = theta*pi/180;
% rotation = [  [cos(theta), sin(theta)]; [-sin(theta), cos(theta)] ];
% X_adapt = X_adapt*rotation;
Y = ones(numel(y),1).*(y==1)+ 2*ones(numel(y),1).*(y==-1);

figure, 
plot(X(Y==1,1),X(Y==1,2),'bo')
hold on 
plot(X(Y==2,1),X(Y==2,2),'ro')
%plot(X_adapt(yt==1,1),X_adapt(yt==1,2),'ro')
%plot(X_adapt(yt==-1,1),X_adapt(yt==-1,2),'r+')
axis equal

save('2Moons_v2','X','X_adapt','Y','x','xt','y','yt');
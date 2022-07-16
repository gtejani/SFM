% %{
close all;
clear all; clc;

run('vlfeat-0.9.21/toolbox/vl_setup')
addpath('Givenfunctions');

% Select the Data - 1 for given_data, 0 for my_data
data = 1;
% data = 0;

if data
    path = 'Data/'
    type = '*.JPG';
    K = [ 1698.873755 0.000000     971.7497705; %intrinsic matrix
    0.000000    1698.8796645 647.7488275;
    0.000000    0.000000     1.000000 ];
else
    path = 'Photos/'
    type = '*.jpg';
    K = [ 3139.52364 0.000000     1945.33128; %intrinsic matrix
    0.000000    3128.63723 1551.30943;
    0.000000    0.000000     1.000000 ];
end

% number of images in the dataset
cd(path)
images = dir(type); 
N = length(images); % number of pictures
cd ..

% Loads images in image cell-array
image{N,1} = [];
for i = 1:N
   image{i} = imread(strcat(path,images(i).name)); 
end
disp('images laoded. Number of images: '), disp(N)

% parameters initialization for 5 point algorithm
t = 0.0000001;              % Ransac threshold for d
ransac_iter = 1000;         % number of Ransac iteration

% parameters initialization for P3P algorithm
inter_ransac = 10000;       % number of Ransac iteration
th = 1;                     % Ransac threshold for d

%% Feature extraction and matching
% Load images, extract features and find correspondences.
disp('Features extraction. It takes a while!')
features{N,1} = [];
descriptor{N,1} = [];
for i = 1:N
    img = single(rgb2gray(image{i}));
    [f,d] = vl_sift(img,'PeakThresh',3,'EdgeThresh',10,'FirstOctave',0);
    features{i} = f;
    descriptor{i} = d;
end
matches{N,N} = [];
for i = 1:N
   for j = 1:N
       if i~=j
        [matches{i,j}, ~] = vl_ubcmatch(descriptor{i},descriptor{j});
       end
   end
end
match_matrix = zeros(N,N);
for i = 1 : N
    for j = 1 : N
        match_matrix(i,j) =length(matches{i,j}) ;
        if i==j
            match_matrix(i,j) = 0;
        end
    end
end

% plotting sample feature maps
disp('Plotting a Sample Feature Map') 
k = 5;
figure, imshow(image{k}), hold on
f = features{k};
perm = randperm(size(f,2)) ;
h1 = vl_plotframe(f(1:2,perm)) ;
set(h1,'color','r','linewidth',3) ;
hold off
title('Features Extraction')

%% maximum feature matching ref and pair 
[val, idx] = max(match_matrix);
[val2, pair] = max(val);
ref = idx(pair);
disp('best reference-pair images:')
disp(ref), disp(pair)

%% Estimate E using 8,7-point algorithm or calibrated 5-point algorithm and RANSAC
paired_feat = matches{ref,pair}; % matched pair's features 

% unnormalized (x,y) coordinates
ref_cords = [features{ref}(1:2,:); ones(1,size(features{ref},2))];
pair_cords = [features{pair}(1:2,:); ones(1,size(features{pair},2))];

% normalized (x,y) coordinates
ref_cords_norm = K\ref_cords;
pair_cords_norm = K\pair_cords;

% matching coressponding feature coordinates in ref and pair
ref_match_cords = ref_cords_norm(:,paired_feat(1,:));
pair_match_cords = pair_cords_norm(:,paired_feat(2,:));

% Plotting Matched features on The Pair of Images for visualization
figure;imshow(image{ref}); hold on
f =  K*ref_match_cords;
perm = randperm(size(f,2)) ;
sel = perm(1:50);
h1 = vl_plotframe(f(1:2,sel)) ;
set(h1,'color','r','linewidth',3) ;
hold off
title('Feature Matching of Best ref-pair')

figure;imshow(image{pair}); hold on
f2 = K*pair_match_cords; 
h2 = vl_plotframe(f2(1:2,sel)) ;
set(h2,'color','r','linewidth',3) ;
title('Feature Matching of Best ref-pair')

% RANSAC iterations on 5-point algorithm
num_inliers = 0;
for i = 1:ransac_iter   
    % random sampling of 5 points
    perm = randsample(size(ref_match_cords,2),5);
    p1 = ref_match_cords(:,perm);
    p2 = pair_match_cords(:,perm);
       
    % 5-point algorithm 
    Evec = calibrated_fivepoint(p1,p2);
    % number of solutions
    n = size(Evec,2); 
    E = mat2cell(permute(reshape(Evec,3,3,n),[2,1,3]),3,3,ones(1,n));
    nE = length(E);
    for j = 1:nE
        E_x1 = E{j}*ref_match_cords;
        E_x2 = E{j}'*pair_match_cords;
        x2_E_x1 = sum(pair_match_cords.*E_x1);
        d = (x2_E_x1).^2.*(1./(E_x1(1,:).^2+E_x1(2,:).^2)+1./(E_x2(1,:).^2+E_x2(2,:).^2));
        inliers = find(abs(d) < t);        
        if length(inliers)>num_inliers
            best_E = E{j};
            num_inliers = length(inliers);         
        end
    end    
end

disp('Essential Matrix: best_E = '), disp(best_E)

%% color extraction 
col_cords = round(ref_cords(:,paired_feat(1,:)));
len = length(col_cords);
colornew = zeros(3,len);
for i = 1 : len  
    color = image{ref};
    colornew(:,i) = [color(col_cords(2,i),col_cords(1,i),1),...
        color(col_cords(2,i),col_cords(1,i),2),...
        color(col_cords(2,i),col_cords(1,i),3) ];
end

%% Finding Camera Matrix 
% Decompose E into [Rp, Tp]
[U,~,V] = svd(best_E);
diagonal = [1 0 0;0 1 0;0 0 0];
mod_E = U*diagonal*V';
[U,~,V] = svd(mod_E);

%finding four possibe solutions
W = [0 -1 0;1 0 0;0 0 1];
Cp = zeros(4,4,4);
Cp(:,:,1) = [U*W*V' U(:,3);0 0 0 1];
Cp(:,:,2) = [U*W*V' -U(:,3);0 0 0 1];
Cp(:,:,3) = [U*W'*V' U(:,3);0 0 0 1];
Cp(:,:,4) = [U*W'*V' -U(:,3);0 0 0 1];

% Making sure each Rotation is legal R matrix
for k = 1:length(Cp)
   if det(Cp(1:3,1:3,k))<0
      Cp(1:3,1:3,k) = -Cp(1:3,1:3,k); 
   end      
end

% test points for finding right solution - can choose any point
q1 = ref_match_cords(:,1);
q2 = pair_match_cords(:,1);
% Triangulation test
M1 = [eye(3) zeros(3,1)];
q1x = [0 -q1(3,1) q1(2,1);q1(3,1) 0 -q1(1,1);-q1(2,1) q1(1,1) 0]; % skew matrices
q2x = [0 -q2(3,1) q2(2,1);q2(3,1) 0 -q2(1,1);-q2(2,1) q2(1,1) 0];
for i = 1:4
   H = inv(Cp(:,:,i));
   M2 = H(1:3,1:4);
   A = [q1x*M1;q2x*M2];
   [u,d,v] = svd(A);
   P = v(:,4); 
   p1est = P./P(4);
   p2est = Cp(:,:,i)\p1est;
   if p1est(3)>0 && p2est(3)>0
      cor_Cp = Cp(:,:,i);
      break;
   end 
end

%% Two-view Reconstruction
p1 = ref_match_cords;
p2 = pair_match_cords;
% Rotation and Translation
Rp = cor_Cp(1:3,1:3); 
Tp = cor_Cp(1:3,4); 
P = [Rp Tp];
Xtemp = zeros(4,size(p1,2));
M2 = cor_Cp(1:3,1:4);
for i=1:size(p1,2)
    A = [p1(1,i)*M1(3,:) - M1(1,:); p1(2,i)*M1(3,:) - M1(2,:); p2(1,i)*M2(3,:) - M2(1,:);  p2(2,i)*M2(3,:) - M2(2,:) ];
    [U,~,V] = svd(A);
    Xtemp(:,i) = V(:,4);
end
Xtemp = Xtemp./Xtemp(4,:);
X = Xtemp(1:3,:);
X = X(1:3,:);               % world points
idx = find(X(3,:)>=0);
X_exist = X(:,idx);         % points that exist in positive Z direction
col_exist = colornew(:,idx);
X_exist = [X_exist; col_exist/255];
figure;pcshow(X_exist(1:3,:)',X_exist(4:6,:)','MarkerSize',20);shg;
xlabel('x');ylabel('y');zlabel('z');
axis([-4 4 -4 4 0 20]);shg
title('Two View SFM')

% Save 3D points to PLY
if data
    filename = 'given_data_2view.ply';
else
    filename = 'my_data_2view.ply';
end
% filename = sprintf('%02dviews.ply', 2);
SavePLY(filename, X_exist);
% %}
% Storing Camera Projection Matrices for each image
C_mat = zeros(3,4,N);
C_mat(:,:,ref) = M1;
C_mat(:,:,pair) = P;
% Storing 2D image points, 3D world points & paired_feats indices
X3d{1,N} = [];
X3d{1,ref} = [ref_match_cords(:,idx);X(:,idx);paired_feat(1,idx)];
X3d{1,pair} = [pair_match_cords(:,idx);X(:,idx);paired_feat(2,idx)];

old_imgs = [ref pair];
img_color = col_exist;
Xpt = X3d{1,ref}(4:6,:);

%% Growing step ( for more than 2 views )
for pics = 3:N
    [best_img,old] = next_pair(match_matrix,old_imgs); % next ref-pair
    match_feats = matches{old,best_img}; % matched pair's features
    
    % unnormalized (x,y) coordinates
    best_cords = [features{best_img}(1:2,:); ones(1,size(features{best_img},2))];
    old_cords = [features{old}(1:2,:); ones(1,size(features{old},2))];
    
    % normalized (x,y) coordinates
    best_cords_norm = K\best_cords;
    old_cords_norm = K\old_cords;
    
    % matching coressponding feature coordinates in ref and pair
    old_match_cords = old_cords_norm(:,match_feats(1,:));
    best_match_cords = best_cords_norm(:,match_feats(2,:));
    
    old_feat = X3d{1,old}(7,:);
    [vals, ind] = ismembertol(old_feat,match_feats(1,:));
    ind1 = find(vals(1,:));
    ref_feat_match = old_feat(1,ind1);
    
    % subset image point in old image
    old_points = old_cords_norm(:,ref_feat_match(1,:));
    
    % subset image point in new image
    newim_feats = match_feats(2,ind(ind>0));
    ips = best_cords_norm(:,newim_feats(1,:));
    
    % subset World point in new image
    wps = X3d{1,old}(4:6,ind1);
    np = size(wps,2);
    data_pnp = [ips;wps]';
    if(np<3)
        break;
    end
    disp('Growing Step - Image Number: ');disp(pics);
    X3d{1,best_img} = [ips;wps;newim_feats];
    n_inliers = 0;
    for i = 1:inter_ransac
       perm = randsample(np,3);
       datas = data_pnp(perm,:);
       RT = PerspectiveThreePoint(datas);
       
       num_RT = size(RT,1)/4;
       for j = 1:num_RT
          Rt = RT(4*j-3:4*j,:); 
          Rt = Rt(1:3,:);
          x3d = [wps;ones(1,np)];
          kp1x = K*Rt*x3d;
          kp1x = kp1x./kp1x(3,:);
          kp2x = K*C_mat(:,:,old)*x3d;
          kp2x = kp2x./kp2x(3,:);         
          d1 = (K*ips - kp1x).^2;
          d2 = (K*old_points - kp2x).^2;
          d1 = d1(1,:) + d1(2,:);
          d2 = d2(2,:) + d2(2,:);
          d = sqrt(d1+d2);
          
          inliers = find(abs(d)<th);
          if length(inliers) > n_inliers
            n_inliers = length(inliers);
            best_RT = Rt;
          end
       end      
    end
    C_mat(:,:,best_img) = best_RT;
    old_imgs = [old_imgs best_img];
    
    col_cord = round(old_cords(:,match_feats(1,:)));
    len = length(col_cord);
    colornew1 = zeros(3,len);
    for i = 1 : len
        color = image{old};
        colornew1(:,i) = [color(col_cord(2,i),col_cord(1,i),1),...
            color(col_cord(2,i),col_cord(1,i),2),...
            color(col_cord(2,i),col_cord(1,i),3) ];
    end
    
    p1 = old_match_cords;
    p2 = best_match_cords;
    M1 = C_mat(:,:,old);
    Xtemp = zeros(4,size(p1,2));
    M2 = best_RT;
    for i=1:size(p1,2)
        A = [p1(1,i)*M1(3,:) - M1(1,:); p1(2,i)*M1(3,:) - M1(2,:); p2(1,i)*M2(3,:) - M2(1,:);  p2(2,i)*M2(3,:) - M2(2,:) ];
        [U,~,V] = svd(A);
        Xtemp(:,i) = V(:,4);
    end
    Xtemp = Xtemp./Xtemp(4,:);
    Xx = Xtemp(1:3,:);
    idx = find(Xx(3,:)>=0);
    Xx_exist = Xx(:,idx);
    col1_exist = colornew1(:,idx);
    X3d{1,best_img} = [best_match_cords(:,idx);Xx_exist;match_feats(2,idx)];
    
    % duplicate points' removal
    check = round(Xx_exist(1:3,:)*5);
    [~,indx,~]= unique(check','rows');
    Xx_exist = Xx_exist(1:3,indx);
    col1_exist = col1_exist(:,indx);
    Xpt = [Xpt,Xx_exist];
    img_color= [img_color col1_exist];   
end

color_pts = img_color;
Xp1 = Xpt(1:3,:);
X_exist_pts = Xp1(:,find(Xp1(3,:)>=0));
color_exist_pts = color_pts(:,find(Xp1(3,:)>=0));
X_exist_pts = [X_exist_pts; color_exist_pts/255];

figure;pcshow(X_exist_pts(1:3,:)',X_exist_pts(4:6,:)','MarkerSize',20);shg;
axis([-20 20 -20 20 0 30]);xlabel('x');ylabel('y');zlabel('z');shg

title('Full Grown Image from all the images');
hold on
for k = 1:size(C_mat,3)
if ~isempty(C_mat(:,:,k))
pcshow(C_mat(:,4,k)','MarkerSize',500)
numbers = num2str(k);
text(C_mat(1,4,k)'+ 0.5, C_mat(2,4,k)'+ 0.5, C_mat(3,4,k)'+ 0.5 ,numbers);
end
end
axis([-20 20 -20 20 0 30])
hold off
%}
disp('Saving Ply file')
if data
    filename = 'given_data_SfM.ply';
else
    filename = 'my_data_SfM.ply';
end
SavePLY(filename, X_exist_pts);

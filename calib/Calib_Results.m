% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly executed under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 3139.523643784334126 ; 3128.637229192162522 ];

%-- Principal point:
cc = [ 1945.331278977470447 ; 1551.309428308991983 ];

%-- Skew coefficient:
alpha_c = 0.000000000000000;

%-- Distortion coefficients:
kc = [ 0.290160791720078 ; -0.557554842153788 ; 0.006199142864102 ; -0.007303370167804 ; 0.000000000000000 ];

%-- Focal length uncertainty:
fc_error = [ 17.508645466162264 ; 18.272107806920307 ];

%-- Principal point uncertainty:
cc_error = [ 22.510703214809801 ; 20.255746788027981 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.000000000000000;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.017898244730119 ; 0.069789038694799 ; 0.003487909789769 ; 0.003252805418869 ; 0.000000000000000 ];

%-- Image size:
nx = 4032;
ny = 3024;


%-- Various other variables (may be ignored if you do not use the Matlab Calibration Toolbox):
%-- Those variables are used to control which intrinsic parameters should be optimized

n_ima = 13;						% Number of calibration images
est_fc = [ 1 ; 1 ];					% Estimation indicator of the two focal variables
est_aspect_ratio = 1;				% Estimation indicator of the aspect ratio fc(2)/fc(1)
center_optim = 1;					% Estimation indicator of the principal point
est_alpha = 0;						% Estimation indicator of the skew coefficient
est_dist = [ 1 ; 1 ; 1 ; 1 ; 0 ];	% Estimation indicator of the distortion coefficients


%-- Extrinsic parameters:
%-- The rotation (omc_kk) and the translation (Tc_kk) vectors for every calibration image and their uncertainties

%-- Image #1:
omc_1 = [ -2.160173e+00 ; -2.170453e+00 ; 1.609961e-01 ];
Tc_1  = [ -8.038714e-01 ; -5.703608e-01 ; 2.748170e+00 ];
omc_error_1 = [ 6.489339e-03 ; 5.953449e-03 ; 1.391452e-02 ];
Tc_error_1  = [ 1.943995e-02 ; 1.793867e-02 ; 1.766009e-02 ];

%-- Image #2:
omc_2 = [ 2.946038e+00 ; -7.061225e-01 ; -3.209035e-01 ];
Tc_2  = [ -3.431908e-01 ; 6.902404e-01 ; 2.730713e+00 ];
omc_error_2 = [ 8.700006e-03 ; 2.877639e-03 ; 1.241511e-02 ];
Tc_error_2  = [ 1.937243e-02 ; 1.777441e-02 ; 1.764939e-02 ];

%-- Image #3:
omc_3 = [ -2.833964e+00 ; -8.755402e-01 ; 6.436188e-01 ];
Tc_3  = [ -1.157228e+00 ; 1.664369e-01 ; 2.723653e+00 ];
omc_error_3 = [ 7.562601e-03 ; 2.103346e-03 ; 1.093327e-02 ];
Tc_error_3  = [ 1.897333e-02 ; 1.812090e-02 ; 1.747890e-02 ];

%-- Image #4:
omc_4 = [ 2.945585e+00 ; -9.690161e-01 ; -4.826401e-01 ];
Tc_4  = [ -3.999129e-01 ; 8.096560e-01 ; 2.483781e+00 ];
omc_error_4 = [ 7.926858e-03 ; 2.051238e-03 ; 1.129458e-02 ];
Tc_error_4  = [ 1.755287e-02 ; 1.621713e-02 ; 1.619556e-02 ];

%-- Image #5:
omc_5 = [ 1.686687e+00 ; -1.691189e+00 ; -7.726099e-01 ];
Tc_5  = [ 8.791837e-01 ; 2.573673e-01 ; 2.811985e+00 ];
omc_error_5 = [ 6.219877e-03 ; 5.769620e-03 ; 8.836499e-03 ];
Tc_error_5  = [ 2.125581e-02 ; 1.880569e-02 ; 1.995130e-02 ];

%-- Image #6:
omc_6 = [ -2.493895e+00 ; -1.457925e+00 ; 6.973507e-01 ];
Tc_6  = [ -8.224644e-01 ; -2.957233e-01 ; 3.365447e+00 ];
omc_error_6 = [ 7.342723e-03 ; 3.669322e-03 ; 1.124388e-02 ];
Tc_error_6  = [ 2.360734e-02 ; 2.173734e-02 ; 1.966610e-02 ];

%-- Image #7:
omc_7 = [ -1.740458e+00 ; -1.814868e+00 ; 6.950211e-01 ];
Tc_7  = [ -6.131567e-01 ; -5.988633e-01 ; 2.490949e+00 ];
omc_error_7 = [ 6.084973e-03 ; 4.544332e-03 ; 8.562140e-03 ];
Tc_error_7  = [ 1.763196e-02 ; 1.611007e-02 ; 1.364211e-02 ];

%-- Image #8:
omc_8 = [ -2.251406e+00 ; -2.025732e+00 ; 2.580669e-01 ];
Tc_8  = [ -7.217226e-01 ; -4.456580e-01 ; 2.939578e+00 ];
omc_error_8 = [ 6.823183e-03 ; 5.921332e-03 ; 1.420976e-02 ];
Tc_error_8  = [ 2.069687e-02 ; 1.901323e-02 ; 1.845375e-02 ];

%-- Image #9:
omc_9 = [ -2.419808e+00 ; -8.485264e-01 ; 1.340933e+00 ];
Tc_9  = [ -4.900141e-01 ; -1.682806e-01 ; 2.829081e+00 ];
omc_error_9 = [ 7.579171e-03 ; 3.620274e-03 ; 9.172751e-03 ];
Tc_error_9  = [ 2.004561e-02 ; 1.811559e-02 ; 1.352442e-02 ];

%-- Image #10:
omc_10 = [ -2.058234e+00 ; -1.996694e+00 ; -2.415263e-01 ];
Tc_10  = [ -6.923825e-01 ; -5.276347e-01 ; 2.258838e+00 ];
omc_error_10 = [ 5.537308e-03 ; 5.883648e-03 ; 1.059897e-02 ];
Tc_error_10  = [ 1.625876e-02 ; 1.496827e-02 ; 1.493295e-02 ];

%-- Image #11:
omc_11 = [ 1.097689e-02 ; -2.636990e+00 ; -2.503215e-01 ];
Tc_11  = [ -2.686379e-01 ; -6.351943e-01 ; 1.841789e+00 ];
omc_error_11 = [ 4.068789e-03 ; 7.589817e-03 ; 1.024718e-02 ];
Tc_error_11  = [ 1.325035e-02 ; 1.168028e-02 ; 1.263844e-02 ];

%-- Image #12:
omc_12 = [ -2.370706e+00 ; 1.360841e-01 ; 1.795829e-01 ];
Tc_12  = [ -5.961884e-01 ; 5.409731e-01 ; 2.428124e+00 ];
omc_error_12 = [ 6.697876e-03 ; 3.710439e-03 ; 8.036579e-03 ];
Tc_error_12  = [ 1.710831e-02 ; 1.573638e-02 ; 1.338710e-02 ];

%-- Image #13:
omc_13 = [ 5.537133e-01 ; -2.816321e+00 ; -5.690651e-01 ];
Tc_13  = [ 6.127023e-01 ; -7.796731e-02 ; 1.951704e+00 ];
omc_error_13 = [ 3.788154e-03 ; 6.444567e-03 ; 1.024643e-02 ];
Tc_error_13  = [ 1.436999e-02 ; 1.276565e-02 ; 1.177953e-02 ];


%% Model 2 of the 10th SPE Comparative Solution Project
% The data set was originally posed as a benchmark for upscaling methods.
% The 3-D geological model consists of 60x220x85 grid cells, each of size
% 20ftx10ftx2ft. The model is a geostatistical realization from the
% Jurassic Upper Brent formations, in which one can find the giant North
% Sea fields of Statfjord, Gullfaks, Oseberg, and Snorre. In this specific
% model, the top 70 ft (35 layers) represent the shallow-marine Tarbert
% formation and the lower 100 ft (50 layers) the fluvial Ness formation.
% The data can be obtained from the SPE website:
% http://www.spe.org/web/csp/
% The data set is used extensively in the literature, and for this reason,
% MRST has a special module that will download and provide easy access to
% the data set.
%
% In this example, we will inspect the SPE10 model in more detail.

mrstModule add spe10

%% Load the model
% The first time you access the model, the MRST dataset system will prompt
% you to download the model from the official website of the comparative
% solution project. Depending upon your internet connection this may take
% quite a while--even several minutes--so please be patient. Notice that
% |getSPE10rock| returns permeabilities in strict SI units (i.e., m^2), and
% the petrophysical data of this model may therefore be used directly in
% simulations in MRST.
rock = getSPE10rock();
p = reshape(rock.poro,60,220,85);
Kx = reshape((rock.perm(:,1)),60,220,85) ;
for i=1:85
    Kx_layer = Kx(:,:,i);
    save(['Kx_' num2str(i) '.mat'],'Kx_layer');
end



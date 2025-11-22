% Copyright © 2022-2030, Tianyuan Li, Department of Electronic Engineering, Tsinghua University. All Rights Reserved.
% mod in 20xxxxxx
clear all;

codeBase = 'xx/ptycho-workstation'; % absolute path
addpath(codeBase);

% set the input parameters here
cfg_in = struct;

% === outputs ===
cfg_in.path_saveOutputs = 'xx'; % absolute path
if ~exist(cfg_in.path_saveOutputs, 'dir')
    mkdir(cfg_in.path_saveOutputs);
end
% prepare the log file
logFileName = fullfile(cfg_in.path_saveOutputs, 'log.txt'); % or 'xx.txt'
logFileID = fopen(logFileName, 'a+');
io.DispAndLog(logFileID, char(datetime));
% copy config file to outputs
configFile = [mfilename('fullpath'), '.m'];
copyfile(configFile, cfg_in.path_saveOutputs);

% === Computing Device ===
cfg_in.gpuID = 1; % index of gpu device
gpuDev = gpuDevice(cfg_in.gpuID);
io.DispAndLog(logFileID, ['GPU AvailableMemory: ', num2str(gpuDev.AvailableMemory / 1e9, '%.2f'), ' GB']);

% === Exp Paras ===
% = Accelerator =
cfg_in.voltage = 300000; % [V] accelerating voltage

% = Electron Beam =
cfg_in.defocus = 1.5e-6; % [m]
cfg_in.Cs = 0.0e-3; % [m]
cfg_in.semiAngle = 3.69e-3; % [rad]

% = Detector =
cfg_in.pSizeFOrigin = 0.005570e+10; % [1/m] pixel size in fourier space, corresponding to camera length (mm)
cfg_in.diffN = 128; % number of pixels on the detector along Y/X axis
cfg_in.ADU = 1; % count per e-, to calculate dose (e-/A2)
semiAngleCamera = utils.diskRadiusPix2semiAngle(floor(cfg_in.diffN / 2), ...
    utils.calWaveLength(cfg_in.voltage), cfg_in.pSizeFOrigin); % [rad]
io.DispAndLog(logFileID, ['Camera SemiAngle: ', num2str(semiAngleCamera*1e3, '%.2f'), ' mrad']);
cfg_in.darkReference = 8 * ones(cfg_in.diffN, cfg_in.diffN, 'single'); % background noise
cfg_in.gainReference = ones(cfg_in.diffN, cfg_in.diffN, 'single'); % pixels are not identical

% = Scan =
% by default, grid scan
cfg_in.scanNYX = 300; % number of scans along Y/X axis
cfg_in.scanArea = 304.988 * 1e-9; % [m] the side length of scan area, corresponding to magnification (kx, Mx)
cfg_in.rotAngle = 6.03; % [degree] clockwise + / anti-clockwise -, rotation around the center of the scan area
% alternative: load scan from file (custom scan)
cfg_in.path_scanToLoad = ''; % if empty, then not to load scanfile
% scanfile(.mat) includes scanCoors
% scanCoors in scanfile: (scanN, 2) 2:[m] (y, x)
% (y, x) is the scan spot's coordinate,
% with the origin at the center of the scan area

% === Recon Paras === 
% = image =
cfg_in.updownsamplingF = 1; % up(>1)/down-sampling in fourier space, reduce/enlarge pSizeF == enlarge/reduce imgSizeR
cfg_in.updownsamplingR = 1; % up(>1)/down-sampling in real space, enlarge/reduce nyquist frequency == reduce/enlarge pSize
% the probN will be automatically calculated with diffN and updownsamplingF/R

% = scan positions =
% - crop a subArea to reconstruct -
cfg_in.subAreaScanIDs = (1:cfg_in.scanNYX^2).'; % the list of the indice of the scan to be reconstructed
% the ids is row-by-row counted, each id is a integer between 1 and scanN 
% for example: if using all the scans, set subAreaScanIDs to (1:scanN).'
% if reconstructing a square sub-area, use the following pesudo-codes
% scanIDMat = reshape(1:cfg_in.scanNYX^2, cfg_in.scanNYX, cfg_in.scanNYX).'; % row-by-row
% cfg_in.subAreaScanIDs = reshape(scanIDMat(1:300, 12:2:300).', [], 1);
cfg_in.isRecenterAfterCrop = true; % if calculating FRC, please set it to false
% - add random offset (gaussian noise) to the coordinates of the scans -
cfg_in.scanPerturbSigma = 0.0; % [m] e.g. 0.5 * cfg_in.scanArea / (cfg_in.scanNYX - 1)

% = cbed =
cfg_in.path_cbeds = 'xx.raw'; % absolute path
cfg_in.flipMode = 1;
% flip: 0-no flip, 1-flip vertically, 2-flip horizontally, 3-flip vertically + flip horizontally 
% 4-transpose, 5-transpose+flip vertically, 6-transpose+flip horizontally, 7-transpose+flip vertically+flip horizontally
cfg_in.nZeroEdges = 0; % set nZeroEdges rows/cols at the edge of CBED to 0.0
cfg_in.cbedMask = ones(cfg_in.diffN, cfg_in.diffN, 'single');
cfg_in.isLoadingCBED = false;
cfg_in.isSavingCBED = false;
cfg_in.savedCBEDFile = '';
cfg_in.isCBEDOnGPU = false; % whether to load the measured CBED data onto gpu

% estimate the cbedTrans
% cbedTrans: pix [tx, ty], moving torwards right/down is positive, size: (1, 2) or (subAreaScanN, 2)
oCBED = cCBED(cfg_in.diffN, cfg_in.scanNYX^2, ...
    'updownsamplingF', 1, 'updownsamplingR', 1, ...
    'subAreaScanIDs', cfg_in.subAreaScanIDs); % without oversampling
oCBED.loadAndPreprocess(cfg_in.path_cbeds, ...
    'nZeroEdges', cfg_in.nZeroEdges, ...
    'darkReference', cfg_in.darkReference, 'gainReference', cfg_in.gainReference, ...
    'ADU', cfg_in.ADU, 'flipMode', cfg_in.flipMode);
center = floor(cfg_in.diffN / 2) + 1;
sYX1 = oCBED.ind2YXInds(cfg_in.scanNYX, cfg_in.scanNYX);
CoMs_fit = utils.fitDiskDrift(oCBED, sYX1, 'plane');
tmp = center - CoMs_fit; % (subAreaScanN, 2), 2: (v, u)
cfg_in.cbedTrans = [tmp(:, 2), tmp(:, 1)]; % (subAreaScanN, 2), 2: (u, v)
io.DispAndLog(logFileID, ['CBED Trans (x, y): (', num2str(mean(cfg_in.cbedTrans(:, 1)), '%.3f'), ', ', num2str(mean(cfg_in.cbedTrans(:, 2)), '%.3f'), ')']);
io.DispAndLog(logFileID, ['Beam Tilt | Max Trans(ABS) (x, y): (', num2str(max(abs(cfg_in.cbedTrans(:, 1)-mean(cfg_in.cbedTrans(:, 1))), [], 'all'), '%.3f'), ', ', num2str(max(abs(cfg_in.cbedTrans(:, 2)-mean(cfg_in.cbedTrans(:, 2))), [], 'all'), '%.3f'), ')']);
oCBED.alignment(cfg_in.cbedTrans, gpuDev);
vP = sqrt(mean(oCBED.cbed, 3));
vPFile = fullfile(cfg_in.path_saveOutputs, 'vP.mrc');
io.WriteMRC(vP, 1, vPFile, 2, 1);

% = obj =
cfg_in.objModel = 'complex'; % 'complex' or 'potential'
% --- there are 2 ways to determine the objNYX ---
% 1. set mannually
cfg_in.objNYXSet = 3000; % >0 is valid, please make sure the set objNYX is sufficient for the reconstruction
% 2. calculate by the scanArea/pixel size/nPad (only if cfg_in.objNYXSet < 0)
cfg_in.nPad = -1; % pad nPad pixels around the obj
% ---
cfg_in.sliThick = 0.0; % [m] thickness of each slice
cfg_in.nSlices = 1;
cfg_in.path_objToLoad = {'', ''}; % {'path to abs', 'path to angle / pot'} absolute path. If empty, then not to load objfile
cfg_in.nPadForLoad = 0;

% = multi-modes of prob =
% --- there are 2 ways to get probs ---
% 1. initialize from scratch
cfg_in.upsamplingProbFFactor = 1; % >= 1, upsampling in fourier space, enlarge the imgSizeR, then cropping in real space
cfg_in.nModes = 1; % number of modes of the probe
cfg_in.slaveModeWeight = 0.0;
    % baseMode's intensity: totalIntensity * (1 - slaveModeWeight * (nModes - 1)),
    % slaveMode's intensity: totalIntensity * slaveModeWeight, 
    % slaveModeWeight: [0, 1)
    % if there is only base probe, it's better to set slaveModeWeight to 0.0
cfg_in.path_vaccumProbe = vPFile;
% 2. load from file
cfg_in.path_probToLoad = {'', ''}; % {'path to abs', 'path to angle'} absolute path
cfg_in.isAdjustProbIntensity = false;

% = iteration =
cfg_in.nRounds = 30; % the number of rounds in reconstruction
cfg_in.iRoundsUpdateObj = 1:cfg_in.nRounds; % in which round to update object function
cfg_in.iRoundsUpdateProb = 1:cfg_in.nRounds; % in which round to update probe function
% use the iRoundsUpdatexx to control whether to update obj/prob
cfg_in.everyNRoundsToSaveOutputs = 30;

% = beats =
% 1 beat involves nSpotsParallel spots, 1 round involves nBeats beats
cfg_in.nSpotsParallel = 4096; % these spots are calculated simultaneously
cfg_in.path_beatToLoad = ''; % beatfile(.mat) includes beats / spotsWeightObj / spotsWeightProb (nSpotsParallel, nBeats)

% = recon algorithm setting =
cfg_in.nSubSpotsParallelM = 4096; % in case of cuda memory is not sufficient, each beat may be split into several subbeats
cfg_in.stepSizeFactor = [0.25, 0.25]; % from 0 to 1, a factor plus stepsize, [alpha, beta], alpha -> object, beta -> probe
probN = ceil(ceil(cfg_in.diffN * cfg_in.updownsamplingF) * cfg_in.updownsamplingR);
cfg_in.gradFMask = ones(probN, probN, 'single');

% = postprocessing =
cfg_in.isShrinkageObjOn = false;
cfg_in.isCenterizeProb = false;

% = monitoring log-likelihood =
cfg_in.isMonitorLikelihoods = false;
cfg_in.nSubSpotsParallelL = -1; % in case of cuda memory is not sufficient, each beat may be split into several subbeats
cfg_in.monitorLikelihoodsMask = ones(probN, probN, 'single');

% = program controlling =
cfg_in.isTicToc = true;
cfg_in.saveInit = true;

% === main function ===
reset(gpuDev);
mains.runPtychoReconGPU(cfg_in, logFileID, gpuDev);
reset(gpuDev);
io.DispAndLog(logFileID, 'The program is finished. We are cool!');
fclose(logFileID);
gpuDevice([]); % deselects the GPU device and clears its memory of gpuArray and CUDAKernel variables

clear all;
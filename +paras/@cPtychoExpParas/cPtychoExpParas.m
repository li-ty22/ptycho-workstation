% Declare and configure the experimental parameters in Ptychography
% Use SI base units, like m/kg/A/s etc.
classdef cPtychoExpParas < handle
    properties
        % Accelerator
        voltage % [V] accelerating voltage
        waveLength % [m]
        sigmaObj % interaction constant in the transmission function

        % Electron Beam
        defocus % [m]
        % In ptychography, defocus means the distance between the upper surface of
        % the sample and the convergent spot of the eletron beam,
        % Be sure to confirm the plus or minus: 
        % 'Positive defocus' means the convergent spot is in behind of 
        % the front surface of the sample along the optical axis, 
        % and 'Negative defocus' means the convergent spot is in front of
        % the front surface of the sample along the optical axis
        Cs % [m]
        semiAngle % [rad] convergent semiangle
        % semiAngle = atan(diskRadiusPix * pSizeF * waveLength)

        % Detector
        pSizeFOrigin % [1/m] pixel size in fourier space, corresponding to [camera length] and [waveLength]
        diffN % number of pixels on the detector along Y/X axis, origin image size
        ADU % [count/e-]
        darkReference % background noise, mrc file
        gainReference % pixels are not identical, mrc file

        % Grid Scan(default)
        scanN % total number of scans
        scanNYX % number of scans along Y/X axis
        scanArea % [m] the side length of scan area, corresponding to magnification
        rotAngle % [degree] clockwise + / anti-clockwise -, rotation around the center of the scan area
        scanCoors % coordinates of all the scan spots
        % (scanN, 2) 2:[m] (y, x)
        % (y, x) is the scan spot's coordinate,
        % with the origin at the center of the scan area
        stepSize % [m]
        % scanN/scanCoors are important
        % load scan
        % scanfile(.mat) includes scanCoors
        path_scanToLoad % scanfile (.mat for now)
    end
    methods
        % initialize the parameters
        function init(obj, cfg_in)
            % cfg_in: a struct to hold all the input parameters, in which,

            % Accelerator
            obj.voltage = cfg_in.voltage;
            obj.waveLength = utils.calWaveLength(obj.voltage);
            obj.sigmaObj = utils.calSigmaObj(obj.voltage);
            % Electron Beam
            obj.defocus = cfg_in.defocus;
            obj.Cs = cfg_in.Cs;
            obj.semiAngle = cfg_in.semiAngle;
            % Detector
            obj.pSizeFOrigin = cfg_in.pSizeFOrigin;
            obj.diffN = cfg_in.diffN; % origin image size
            obj.ADU = cfg_in.ADU;
            obj.darkReference = cfg_in.darkReference;
            obj.gainReference = cfg_in.gainReference;
            % scan
            obj.path_scanToLoad = cfg_in.path_scanToLoad;
            isLoadScan = ~isempty(obj.path_scanToLoad);
            if isLoadScan
                tmp = load(obj.path_scanToLoad); % (.mat)
                obj.scanNYX = nan;
                obj.scanArea = nan;
                obj.rotAngle = nan;
                obj.scanCoors = tmp.scanCoors; % (scanN, 2) 2:[m] (y, x)
                obj.stepSize = nan;
                obj.scanN = size(obj.scanCoors, 1);
                clear tmp;
                % scanN and scanCoors are important
                % if we choose the custom scan, scanNYX/scanArea/rotAngle/stepsize are irrelevant, for they are corresponding to the grid scan 
            else
                % Grid Scan (default)
                obj.scanNYX = cfg_in.scanNYX;
                obj.scanArea = cfg_in.scanArea;
                obj.rotAngle = cfg_in.rotAngle;
                [obj.scanCoors, obj.stepSize, obj.scanN] = scans.gridScan(obj.scanNYX, obj.scanArea, obj.rotAngle);
                % scanCoors: (scanN, 2) 2:[m] (y, x)
                % scanN and scanCoors are important
            end
        end
    end
end
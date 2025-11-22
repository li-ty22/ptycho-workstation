% CBED data block and relevant methods
classdef cCBED < handle
    properties
        cbed % cbed data block (diffN/probN, diffN/probN, subAreaScanN)
        % before updownsampling, the size is (diffN, diffN, subAreaScanN)
        % after updownsampling, that would be (probN, probN, subAreaScanN)
        totElectrons % the total number of the detected electrons in this dataset

        % the size feature of the dataset
        % different from the raw dataset, this dataset could include partial data or zero-padded/oversampled data
        updownsamplingF % up(>1)/down-sampling in fourier space, reduce/enlarge pSizeF
        updownsamplingR % up(>1)/down-sampling in real space, enlarge/reduce nyquist frequency
        probN % number of pixels of each cbed image along Y/X axis after the preprocessing
        subAreaScanIDs % the list of the indice of the scan to be operated or reconstructed
        subAreaScanN % the number of the scans to be operated or reconstructed
        
        % the size feature of the raw dataset
        diffN % number of pixels on the detector along Y/X axis
        scanN % number of the scans in the experiment

        % other records
        path_cbeds
        nZeroEdges
        darkReference
        gainReference
        ADU
        flipMode
        cbedTrans
        cbedMask
    end
    methods
        % constructor
        function obj = cCBED(diffN, scanN, varargin)
            % varargin: 
            % updownsamplingF, updownsamplingR,
            % subAreaScanIDs
            p = inputParser;
            addParameter(p, 'updownsamplingF', 1);
            addParameter(p, 'updownsamplingR', 1);
            addParameter(p, 'subAreaScanIDs', 1:scanN);
            parse(p, varargin{:});
            
            obj.diffN = diffN;
            obj.scanN = scanN;
            obj.updownsamplingF = p.Results.updownsamplingF;
            obj.updownsamplingR = p.Results.updownsamplingR;
            obj.probN = ceil(ceil(obj.diffN * obj.updownsamplingF) * obj.updownsamplingR);
            obj.subAreaScanIDs = p.Results.subAreaScanIDs(:);
            obj.subAreaScanN = length(obj.subAreaScanIDs);
            obj.cbed = [];
        end

        function loadAndPreprocess(obj, file, varargin)
            % varargin:
            % nZeroEdges
            % darkReference, gainReference
            % ADU: count/e-
            % flip: 0-no flip, 1-flip vertically, 2-flip horizontally, 3-flip vertically + flip horizontally 
            % 4-transpose, 5-transpose+flip vertically, 6-transpose+flip horizontally, 7-transpose+flip vertically+flip horizontally 
            p = inputParser;
            addParameter(p, 'nZeroEdges', 0);
            addParameter(p, 'darkReference', zeros(obj.diffN, obj.diffN, 'single'));
            addParameter(p, 'gainReference', ones(obj.diffN, obj.diffN, 'single'));
            addParameter(p, 'ADU', 1);
            addParameter(p, 'flipMode', 0);
            parse(p, varargin{:});
            obj.nZeroEdges = p.Results.nZeroEdges;
            obj.darkReference = p.Results.darkReference;
            obj.gainReference = p.Results.gainReference;
            obj.ADU = p.Results.ADU;
            obj.flipMode = p.Results.flipMode;
            obj.path_cbeds = file;

            % load cbed raw data from .raw file or .mrc/.mrcs file
            [~, ~, ext] = fileparts(file);
            switch ext
                case '.raw'
                    cbedFromFile = io.ReadRaw(file, obj.diffN, obj.diffN, obj.scanN, false);           
                case '.mrcs'
                    cbedFromFile = io.ReadMRC(file);
                case '.mrc'
                    cbedFromFile = io.ReadMRC(file);
                case '.mat' % zip cbed for now
                    tmp = load(file);
                    ind = tmp.ind;
                    value = tmp.value;
                    cbedFromFile = utils.unzipCBED(ind, value, [obj.diffN, obj.diffN, obj.scanN]);
                otherwise
                    error('unsupported extension for cbed file');
            end % (diffN, diffN, scanN)
            % select data in ROI
            obj.cbed = cbedFromFile(:, :, obj.subAreaScanIDs); % (diffN, diffN, subAreaScanN)
            clear cbedFromFile;
            % set edges to zero
            obj.cbed(1:obj.nZeroEdges, :, :) = 0.0;
            obj.cbed(:, 1:obj.nZeroEdges, :) = 0.0;
            obj.cbed((end-obj.nZeroEdges+1):end, :, :) = 0.0;
            obj.cbed(:, (end-obj.nZeroEdges+1):end, :) = 0.0;
            % adjust Dark&Gain, scale with ADU
            obj.cbed = (obj.cbed - obj.darkReference) .* obj.gainReference / obj.ADU;
            obj.cbed(obj.cbed < 0.0) = 0.0;
            obj.totElectrons = sum(obj.cbed, 'all'); % e-
            % flip: 0-no flip, 1-flip vertically, 2-flip horizontally, 3-flip vertically + flip horizontally 
            % 4-transpose, 5-transpose+flip vertically, 6-transpose+flip horizontally, 7-transpose+flip vertically+flip horizontally
            switch obj.flipMode
                % 0 - no flip
                case 0
                    % do nothing
                case 1 % 1 - vertically
                    obj.cbed = flip(obj.cbed, 1);
                case 2 % 2 - horizontally
                    obj.cbed = flip(obj.cbed, 2);
                case 3 % 3 - vertically + horizontally
                    obj.cbed = flip(obj.cbed, 1);
                    obj.cbed = flip(obj.cbed, 2);
                case 4 % 4 - transpose
                    obj.cbed = permute(obj.cbed, [2, 1, 3]);
                case 5 % 5 - transpose + vertically
                    obj.cbed = permute(obj.cbed, [2, 1, 3]);
                    obj.cbed = flip(obj.cbed, 1);
                case 6 % 6 - transpose + horizontally
                    obj.cbed = permute(obj.cbed, [2, 1, 3]);
                    obj.cbed = flip(obj.cbed, 2);
                case 7 % 7 - transpose + horizontally + vertically
                    obj.cbed = permute(obj.cbed, [2, 1, 3]);
                    obj.cbed = flip(obj.cbed, 2);
                    obj.cbed = flip(obj.cbed, 1);
                otherwise
                    error('wrong flip mode');
            end
        end

        % translation
        function alignment(obj, tran, gpuDev)
            % tran: pix [tx, ty], moving torwards right/down is positive
            % or (subAreaScanN, 2), translate cbed 1 by 1
            % check the inputs
            obj.cbedTrans = tran;
            if (size(tran, 1) ~= 1 && size(tran, 1) ~= obj.subAreaScanN) || size(tran, 2) ~= 2
                error('Wrong Input (tran)');
            end
            % translation
            if size(tran, 1) == 1 % apply same translation to the whole dataset
                if sqrt(sum(abs(tran).^2, 'all')) > 0.01
                    disp('CBED Alignment: apply same translation to the whole dataset');
                    obj.cbed = imtranslate(obj.cbed, tran, 'linear', 'OutputView', 'same', 'FillValues', 0.0);
                end
                % from doc: 
                %   If A has more than two dimensions and translation is a 2-element vector, 
                %   then imtranslate applies the 2-D translation to each plane of A.
            else % translate cbed 1 by 1
                disp('CBED Alignment: apply different translation to each frame in the dataset (May be time-consuming)');
                if isempty(gpuDev)
                    % bilinear, cpu
                    for iScan = 1:obj.subAreaScanN
                        if sqrt(sum(abs(tran(iScan, :)).^2, 'all')) > 0.01
                            obj.cbed(:, :, iScan) = imtranslate(obj.cbed(:, :, iScan), tran(iScan, :), 'linear', 'OutputView', 'same', 'FillValues', 0.0);
                        end
                    end
                    % odd!!! imtranslate does not support gpu
                else
                    % bilinear, gpu
                    N = size(obj.cbed, 1);
                    [X, Y] = meshgrid(1:N);
                    XG = gpuArray(single(X)); YG = gpuArray(single(Y));
                    for iScan = 1:obj.subAreaScanN
                        tmpCBEDG = gpuArray(obj.cbed(:, :, iScan));
                        tmpCBEDG = interp2(XG, YG, tmpCBEDG, XG - tran(iScan, 1), YG - tran(iScan, 2), 'linear', 0.0);
                        obj.cbed(:, :, iScan) = gather(tmpCBEDG);
                    end
                end
                
            end
            obj.cbed(obj.cbed < 0.0) = 0.0;
        end

        function maskCBED(obj, cbedMask)
            obj.cbedMask = cbedMask;
            obj.cbed = obj.cbed .* cbedMask;
        end

        % up/down sampling
        function updownsampling(obj)
            % diffN -> updownsampledDiffN, reduce/enlarge pSizeF, interpolation
            updownsampledDiffN = ceil(obj.diffN * obj.updownsamplingF);
            if obj.updownsamplingF == 1
                cbed_updownsampledDiffN = obj.cbed;
            else
                cbed_updownsampledDiffN = zeros(updownsampledDiffN, updownsampledDiffN, obj.subAreaScanN, 'single');
                for iScan = 1:obj.subAreaScanN
                    cbed_updownsampledDiffN(:, :, iScan) = imresize(obj.cbed(:, :, iScan), obj.updownsamplingF, "bilinear");
                    % from doc: imresize uses the ceil function when calculating the output image size
                end
                cbed_updownsampledDiffN(cbed_updownsampledDiffN < 0.0) = 0.0;
            end
            % updownsampledDiffN -> probN, enlarge/reduce nyquist frequency, padding zero or crop
            if obj.updownsamplingR == 1
                % updownsampledDiffN = probN
                obj.cbed = cbed_updownsampledDiffN;
            elseif obj.updownsamplingR > 1 % pad 0, enlarge nyquist frequency 
                % probN > updownsampledDiffN
                ul = floor(obj.probN / 2) + 1 - floor(updownsampledDiffN / 2);
                br = ul + updownsampledDiffN - 1;
                obj.cbed = zeros(obj.probN, obj.probN, obj.subAreaScanN, 'single');
                obj.cbed(ul:br, ul:br, :) = cbed_updownsampledDiffN;
            else % crop image, reduce nyquist frequency
                % updownsampledDiffN > probN
                ul = floor(updownsampledDiffN / 2) + 1 - floor(obj.probN / 2);
                br = ul + obj.probN - 1;
                obj.cbed = cbed_updownsampledDiffN(ul:br, ul:br, :);
            end
        end

        % get a subset of CBEDs
        function subCBEDs = getSubset(obj, IDs)
            % IDs: (nSpots, 1)
            % subCBEDs: (probN, probN, nSpots)
            subCBEDs = obj.cbed(:, :, IDs(:));
        end
        
        % get the average image of all the CBEDs
        function avgImg = getAvgCBED(obj)
            avgImg = mean(obj.cbed, 3);
        end

        % statistic on CBED
        % 1. check the intensity of each frame
        function intensity1by1 = calIntensityEachFrame(obj)
            intensity1by1 = squeeze(sum(obj.cbed, [1 2]));
        end

        % parameter estimation on CBED
        % 1. estimate the radius(pix) of the disk
        function diskRadiusPix = calDiskRadiusPix(obj)
            sqrtAvgCBED = sqrt(obj.getAvgCBED());
            th = rms(sqrtAvgCBED, 'all'); % root mean square
            mask = sqrtAvgCBED > th;
            diskRadiusPix = sqrt(sum(mask, 'all') / pi);
        end
        % 2. estimate the translation of the disk (by calculating the mass center)
        function tran = calDiskTransPix(obj)
            % tran: [tx, ty], moving torwards right/down is positive 
            % avgCBED
            avgImg = obj.getAvgCBED();
            M = sum(avgImg, 'all');
            % calculate the mass center
            imgN = size(avgImg, 1);
            center = floor(imgN / 2) + 1;
            [U, V] = meshgrid(1:imgN);
            m_u = sum((U - center) .* avgImg, 'all') / M;
            m_v = sum((V - center) .* avgImg, 'all') / M;
            tran = -[m_u, m_v];
        end

        % use CoM to estimate the translation of CBED
        function CoMs = calCoMs(obj)
            % calculate the mass center of each cbed
            % CoMs: (subAreaScanN, 2), 2: (v, u)
            Ms = sum(obj.cbed, [1, 2]); % (1, 1, subAreaScanN)
            nYX = size(obj.cbed, 1);
            [U, V] = meshgrid(1:nYX); % (nYX, nYX)
            m_u = sum(U .* obj.cbed, [1 2]) ./ Ms; % (1, 1, subAreaScanN)
            m_v = sum(V .* obj.cbed, [1 2]) ./ Ms; % (1, 1, subAreaScanN)
            CoMs = [m_v(:), m_u(:)];
        end

        % visualization
        % 1. average CBED and the radius of disk
        % varargin: diskRadiusPix log
        displayAvgCBED(obj, varargin);

        % todo: upgrade
        % 2. get the video(gif) of a subset of cbeds
        videoCBED(obj, fileVideo, IDs);

        % utils
        function [YXInds] = ind2YXInds(obj, scanNY, scanNX)
            % YXInds: (subAreaScanN, 2), 2: (scanY, scanX)
            [sX, sY] = meshgrid(1:scanNX, 1:scanNY);
            sX = sX.'; sY = sY.';
            sX1 = sX(obj.subAreaScanIDs);
            sY1 = sY(obj.subAreaScanIDs);
            YXInds = [sY1, sX1];
        end

        function out = checkParas(obj, in_diffN, in_scanN, ...
                in_updownsamplingF, in_updownsamplingR, ...
                in_subAreaScanIDs, ...
                in_path_cbeds, ...
                in_nZeroEdges, ...
                in_darkReference, in_gainReference, ...
                in_ADU, in_flipMode, ...
                in_cbedTrans, ...
                in_cbedMask)
            out = true;
            if ~isequal(in_diffN, obj.diffN)
                out = false;
                return;
            end
            if ~isequal(in_scanN, obj.scanN)
                out = false;
                return;
            end
            if ~isequal(in_updownsamplingF, obj.updownsamplingF)
                out = false;
                return;
            end
            if ~isequal(in_updownsamplingR, obj.updownsamplingR)
                out = false;
                return;
            end
            if ~isequal(in_subAreaScanIDs, obj.subAreaScanIDs)
                out = false;
                return;
            end
            if ~isequal(in_path_cbeds, obj.path_cbeds)
                out = false;
                return;
            end
            if ~isequal(in_nZeroEdges, obj.nZeroEdges)
                out = false;
                return;
            end
            if ~isequal(in_darkReference, obj.darkReference)
                out = false;
                return;
            end
            if ~isequal(in_gainReference, obj.gainReference)
                out = false;
                return;
            end
            if ~isequal(in_ADU, obj.ADU)
                out = false;
                return;
            end
            if ~isequal(in_flipMode, obj.flipMode)
                out = false;
                return;
            end
            if ~isequal(in_cbedTrans, obj.cbedTrans)
                out = false;
                return;
            end
            if ~isequal(in_cbedMask, obj.cbedMask)
                out = false;
                return;
            end
        end
    end
end

classdef cSpot < handle
    properties
        % --- basic function ---
        % coordsXX: center coordinates of each scan, with the origin at the center of the scan area,
        % spots' order is the same as the order in oReconParas.beats (nSpotsParallel, nBeats)
        coordsInit % the init coordinates (2, 1, 1, nSpotsParallel, nBeats)
        % 2: [m] (y0, x0), which are used to restrict the searching area
        % dim "1" for nMC and nTransitions
        coordsCurrent % the coordinates to calculate reconstruction (2, nSpotsParallel, nBeats)
        % 2: [m] (y, x)
        nSpotsParallel
        nBeats
        beats % same as the beats in oReconParas

        status % -1: uninitialized, 0: basic function, 1: MCMC
    end
    methods
        %% --- initialization ---
        function obj = cSpot()
            obj.status = -1;
        end
        % init coords of each spots, arrange them as the order in beats
        function setBeats(obj, coordsInit, beats)
            % coordsInit: [m] (nSpots, 2), spots' order is the same as the order in oCBED (row by row)
            % beats: same as the beats in cReconParas (nSpotsParallel, nBeats)
            obj.beats = beats;
            obj.nSpotsParallel = size(beats, 1);
            obj.nBeats = size(beats, 2);
            obj.coordsCurrent = zeros(2, obj.nSpotsParallel, obj.nBeats, 'single');
            obj.coordsCurrent(1, :) = coordsInit(beats(:), 1).'; % y
            obj.coordsCurrent(2, :) = coordsInit(beats(:), 2).'; % x
            obj.coordsInit = reshape(obj.coordsCurrent, 2, 1, 1, obj.nSpotsParallel, obj.nBeats);
            % add 2 dims for nMC and nTransitions, (2, 1, 1, nSpotsParallel, nBeats)
            obj.status = 0;
        end
        
        %% --- basic function ---
        % get the coordinates for reconstruction
        function [cys, cxs, sigmaY, sigmaX] = getBeatScanCoors(obj, iBeat)
            % cys/cxs/sigmaY/sigmaX: [m] (nSpotsParallel, 1)
            coordsCurrent1Beat = obj.coordsCurrent(:, :, iBeat).';
            cys = coordsCurrent1Beat(:, 1);
            cxs = coordsCurrent1Beat(:, 2);
            sigmaY = NaN;
            sigmaX = NaN;
        end
        function [cys, cxs, sigmaY, sigmaX] = getSubBeatScanCoors(obj, iBeat, iSubBeat, nSubSpotsParallel)
            % cys/cxs/sigmaY/sigmaX: [m] (nSpotsParallel, 1)
            [lind, uind] = obj.subBeatInds(iSubBeat, nSubSpotsParallel);
            coordsCurrent1SubBeat = obj.coordsCurrent(:, lind:uind, iBeat).';
            cys = coordsCurrent1SubBeat(:, 1);
            cxs = coordsCurrent1SubBeat(:, 2);
            sigmaY = NaN;
            sigmaX = NaN;
        end
        function setSubBeatScanCoors(obj, cys, cxs, iBeat, iSubBeat, nSubSpotsParallel)
            % cys/cxs/sigmaY/sigmaX: [m] (nSpotsParallel, 1)
            [lind, uind] = obj.subBeatInds(iSubBeat, nSubSpotsParallel);
            coordsCurrent1SubBeat = obj.coordsCurrent(:, lind:uind, iBeat).';
            coordsCurrent1SubBeat(:, 1) = cys;
            coordsCurrent1SubBeat(:, 2) = cxs;
            % coordsCurrent1SubBeat(:, 3) = sigmaY;
            % coordsCurrent1SubBeat(:, 4) = sigmaX;
            obj.coordsCurrent(:, lind:uind, iBeat) = coordsCurrent1SubBeat.';
        end
        function [cysInit, cxsInit] = getSubBeatInitScanCoors(obj, iBeat, iSubBeat, nSubSpotsParallel)
            % cys/cxs/sigmaY/sigmaX: [m] (nSpotsParallel, 1)
            [lind, uind] = obj.subBeatInds(iSubBeat, nSubSpotsParallel);
            coordsInit1SubBeat = reshape(obj.coordsInit(:, 1, 1, lind:uind, iBeat), 2, []).';
            cysInit = coordsInit1SubBeat(:, 1);
            cxsInit = coordsInit1SubBeat(:, 2);
        end

        %% --- utils function ---
        function [lind, uind] = subBeatInds(obj, iSubBeat, nSubSpotsParallel)
            lind = (iSubBeat - 1) * nSubSpotsParallel + 1;
            uind = iSubBeat * nSubSpotsParallel;
            if lind < 1 || uind > obj.nSpotsParallel
                error('Wrong subbeat ind!');
            end
        end

        %% --- visualization ---
        function saveCoors(obj, file, scanN, subAreaScanIDs)
            % file: .mat
            subAreaScanCoors = zeros(numel(obj.beats), 2, 'single');
            subAreaScanCoors(obj.beats(:), :) = obj.coordsCurrent(1:2, :).'; % (y, x), (subAreaScanN, 2)
            scanCoors = nan(scanN, 2, 'single');
            scanCoors(subAreaScanIDs(:), :) = subAreaScanCoors;
            save(file, "scanCoors");
        end
    end
end
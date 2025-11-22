% represent the object function on GPU, etc
classdef cObjG < handle
    properties
        dataG % (nY, nX, nSlices) volume of the object unction
        % 'single' or 'complex single' on GPU
        pSize % [m] pixel size on the Y-X plane
        sliThick % [m] thickness of each slice along the Z axis
        oriIndY % ind of the origin along Y axis (at the center of the scanArea)
        oriIndX % ind of the origin along X axis (at the center of the scanArea)
    end
    methods
        % initialization
        % 1. initialize 1.0
        function initOnes(obj, nY, nX, nSlices, pSize, sliThick, oriIndY, oriIndX, typename)
            % typename = 'single' or 'complex'
            switch typename
                case 'single'
                    obj.dataG = ones(nY, nX, nSlices, 'single', 'gpuArray');
                case 'complex'
                    obj.dataG = complex(ones(nY, nX, nSlices, 'single', 'gpuArray'), ...
                        zeros(nY, nX, nSlices, 'single', 'gpuArray'));
                otherwise
                    disp('Wrong typename');
            end
            obj.pSize = pSize;
            obj.sliThick = sliThick;
            obj.oriIndY = oriIndY;
            obj.oriIndX = oriIndX;
        end
        % 2. initialize 0.0
        function initZeros(obj, nY, nX, nSlices, pSize, sliThick, oriIndY, oriIndX, typename)
            % typename = 'single' or 'complex'
            switch typename
                case 'single'
                    obj.dataG = zeros(nY, nX, nSlices, 'single', 'gpuArray');
                case 'complex'
                    obj.dataG = complex(zeros(nY, nX, nSlices, 'single', 'gpuArray'), ...
                        zeros(nY, nX, nSlices, 'single', 'gpuArray'));
                otherwise
                    disp('Wrong typename');
            end
            obj.pSize = pSize;
            obj.sliThick = sliThick;
            obj.oriIndY = oriIndY;
            obj.oriIndX = oriIndX;
        end
        % 3. input an data
        function initFromData(obj, data, pSize, sliThick, oriIndY, oriIndX)
            if isgpuarray(data)
                obj.dataG = single(data);
            else
                obj.dataG = gpuArray(single(data));
            end
            obj.pSize = pSize;
            obj.sliThick = sliThick;
            obj.oriIndY = oriIndY;
            obj.oriIndX = oriIndX;
        end

        % set dataG to 0.0 (set the accumulated variable to 0.0)
        function zerolize(obj)
            obj.dataG = obj.dataG * zeros('like', obj.dataG);
        end
        
        function [anchorULYs, anchorULXs, disYs, disXs] = coordConversion(obj, cys, cxs, nYX)
            % cys/cxs: [m] (nSpotsParallel, 1) coordinates of the sub-image centers on each slice
            % nYX: size of the sub image on each slice
            % anchorULYs/anchorULXs/disYs/disXs: [pix] (nSpotsParallel, 1)
            % anchor: the pixel ind at the upper-left of the first pixel (1, 1) in sub-image
            % dis: the distance(pix) between the anchor and the first pixel (1, 1) in sub-image
            
            % change coordinates' unit from m to ind
            cIndYs = cys / obj.pSize + obj.oriIndY;
            cIndXs = cxs / obj.pSize + obj.oriIndX;
            % calculate the crop region in the objPot
            anchorULYs = round(cIndYs - floor(nYX / 2)); % the nearest pixel of the first pixel (1, 1) in sub-image
            anchorULXs = round(cIndXs - floor(nYX / 2));
            % anchorULYs = floor(cIndYs - floor(nYX / 2)); % the pixel ind at the upper-left of the first pixel (1, 1) in sub-image
            % anchorULXs = floor(cIndXs - floor(nYX / 2));
            % calculate the linear combination factor
            disXs = cIndXs - round(cIndXs);
            disYs = cIndYs - round(cIndYs);
            % disXs = cIndXs - floor(cIndXs);
            % disYs = cIndYs - floor(cIndYs); % distence bewteen the center and its' upper-left pixel
        end

        % crop nSpotsParallel sub-volumes out of dataG
            % [in]:
            % cys/cxs: [m] (nSpotsParallel, 1) central coordinates of the sub-images
            % with origin (0, 0) at (oriIndY, oriIndX) 
            % nYX: size of the sub-image
            % varargin: isInterpolation, if false, subImage = data(anchorULY:anchorBRY-1, anchorULX:anchorBRX-1)
            % [out]:
            % subVolumesG: (nYX, nYX, nSpotsParallel, nSlices) gpu
        subVolumesG = cropVolumesGPU(obj, cys, cxs, nYX, varargin);

        % fuse-add nSpotsParallel sub-volumes into dataG
            % [in]:
            % cys/cxs: [m] (nSpotsParallel, 1) central coordinates of the sub-images
            % with origin (0, 0) at (oriIndY, oriIndX) 
            % subVolumesG: (nYX, nYX, nSpotsParallel, nSlices) gpu
            % weightsG: (nSpotsParallel, 1)
            % varargin: isInterpolation, if false, data(anchorULY:anchorBRY-1, anchorULX:anchorBRX-1) += subImage;
        fuseAddVolumesGPU(obj, subVolumesG, cys, cxs, weightsG, varargin);
        
        % pad n vals around the obj on each slice
        function padGPU(obj, n, val)
            obj.dataG = padarray(obj.dataG, [n, n, 0], val, 'both');
            obj.oriIndY = obj.oriIndY + n;
            obj.oriIndX = obj.oriIndX + n;
        end

        % nY
        function n = nY(obj)
            n = size(obj.dataG, 1);
        end
        % nX
        function n = nX(obj)
            n = size(obj.dataG, 2);
        end
        % nSlices
        function n = nSlices(obj)
            n = size(obj.dataG, 3);
        end
        % pSizeFObj
        function df = pSizeFObj(obj)
            % assumed nY = nX
            df = 1 / obj.pSize / obj.nY();
        end

        % saving
        % --- complex function ---
        function save(obj, absMRCFile, angleMRCFile)
            pSize_A = obj.pSize * 1e10;
            dataCPU = gather(obj.dataG);
            isSaveAbs = ~isempty(absMRCFile);
            isSaveAngle = ~isempty(angleMRCFile);
            if isSaveAbs
                io.WriteMRC(abs(dataCPU), pSize_A, absMRCFile, 2, obj.nSlices());
            end
            if isSaveAngle
                io.WriteMRC(angle(dataCPU), pSize_A, angleMRCFile, 2, obj.nSlices());
            end
            if ~isSaveAbs && ~isSaveAngle
                warning('No file was saved!');
            end
        end
        % --- real function ---
        function savePot(obj, potMRCFile)
            pSize_A = obj.pSize * 1e10;
            dataCPU = gather(obj.dataG);
            io.WriteMRC(dataCPU, pSize_A, potMRCFile, 2, obj.nSlices());
        end
        
    end
end
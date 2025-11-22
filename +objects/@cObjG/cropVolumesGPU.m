function subVolumesG = cropVolumesGPU(obj, cys, cxs, nYX, varargin)
% crop nSpotsParallel sub-volumes out of dataG
% [in]:
% cys/cxs: [m] (nSpotsParallel, 1) central coordinates of the sub-images
% with origin (0, 0) at (oriIndY, oriIndX) 
% nYX: size of the sub-image
% varargin: isInterpolation, if false, subImage = data(anchorULY:anchorBRY-1, anchorULX:anchorBRX-1)
% [out]:
% subVolumesG: (nYX, nYX, nSpotsParallel, nSlices) gpu
    
    p = inputParser;
    addParameter(p, 'isInterpolation', true);
    parse(p, varargin{:});
    isInterpolation = p.Results.isInterpolation;
    
    % allocate memory
    nSpotsParallel = length(cys);
    isComplex = ~isreal(obj.dataG);
    
    % coordinates conversion
    [anchorULYs, anchorULXs, disYs, disXs] = obj.coordConversion(cys, cxs, nYX);
    anchorBRYs = anchorULYs + nYX; % the pixel ind at the bottom-right of the last pixel (nYX, nYX) in sub-image
    anchorBRXs = anchorULXs + nYX;
    
    if isInterpolation % bilinear interpolation
        if isComplex == true % complex
            subVolumesG = complex(zeros(nYX, nYX, obj.nSlices(), nSpotsParallel, 'single', 'gpuArray'), ...
                zeros(nYX, nYX, obj.nSlices(), nSpotsParallel, 'single', 'gpuArray'));
        else % single
            subVolumesG = zeros(nYX, nYX, obj.nSlices(), nSpotsParallel, 'single', 'gpuArray');
        end
        for iSpot = 1:nSpotsParallel
            anchorULY = anchorULYs(iSpot);
            anchorBRY = anchorBRYs(iSpot);
            anchorULX = anchorULXs(iSpot);
            anchorBRX = anchorBRXs(iSpot);
            sp1 = disXs(iSpot);
            sq1 = disYs(iSpot);
            sp = 1 - sp1;
            sq = 1 - sq1;
            subVolumesG(:, :, :, iSpot) = ...
                obj.dataG(anchorULY:anchorBRY-1, anchorULX:anchorBRX-1, :) .* (sq * sp) + ...
                obj.dataG(anchorULY+1:anchorBRY, anchorULX:anchorBRX-1, :) .* (sq1 * sp) + ...
                obj.dataG(anchorULY:anchorBRY-1, anchorULX+1:anchorBRX, :) .* (sq * sp1) + ...
                obj.dataG(anchorULY+1:anchorBRY, anchorULX+1:anchorBRX, :) .* (sq1 * sp1);
        end
        subVolumesG = permute(subVolumesG, [1 2 4 3]); % (nYX, nYX, nSpotsParallel, nSlices)
    else
        if isComplex == true % complex
            subVolumesG = complex(zeros(nYX, nYX, nSpotsParallel, obj.nSlices(), 'single', 'gpuArray'), ...
                zeros(nYX, nYX, nSpotsParallel, obj.nSlices(), 'single', 'gpuArray'));
        else % single
            subVolumesG = zeros(nYX, nYX, nSpotsParallel, obj.nSlices(), 'single', 'gpuArray');
        end
        if isComplex
            subVolumesG = myCUDA.cropComplexGPU_mex(subVolumesG, obj.dataG, uint32(anchorULYs), uint32(anchorULXs));
        else
            subVolumesG = myCUDA.cropRealGPU_mex(subVolumesG, obj.dataG, uint32(anchorULYs), uint32(anchorULXs));
        end
    end % for-loop in if-else, rather than if-else in for-loop
end
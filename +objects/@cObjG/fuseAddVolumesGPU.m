function fuseAddVolumesGPU(obj, subVolumesG, cys, cxs, weightsG, varargin)
% fuse-add nSpotsParallel sub-volumes into dataG
% [in]:
% cys/cxs: [m] (nSpotsParallel, 1) central coordinates of the sub-images
% with origin (0, 0) at (oriIndY, oriIndX) 
% subVolumesG: (nYX, nYX, nSpotsParallel, nSlices) gpu
% weightsG: (nSpotsParallel, 1)
% varargin: isInterpolation, if false, data(anchorULY:anchorBRY-1, anchorULX:anchorBRX-1) += subImage;
    
    p = inputParser;
    addParameter(p, 'isInterpolation', true);
    parse(p, varargin{:});
    isInterpolation = p.Results.isInterpolation;
    
    % features
    nSpotsParallel = length(cys);
    nYX = size(subVolumesG, 1);
    isComplex = ~isreal(obj.dataG);
    
    % coordinates conversion
    [anchorULYs, anchorULXs, disYs, disXs] = obj.coordConversion(cys, cxs, nYX);
    anchorBRYs = anchorULYs + nYX; % the pixel ind at the bottom-right of the last pixel (nYX, nYX) in sub-image
    anchorBRXs = anchorULXs + nYX;
    
    if isInterpolation
        % calculate the linear combination factor
        qs = 1.0 - disYs; % distence bewteen the center and its' bottom-right pixel
        ps = 1.0 - disXs;
        % bilinear interpolation
        sqs_ = 1 - qs;
        sps_ = 1 - ps;
        sq1s_ = qs;
        sp1s_ = ps;
        ul = single(sps_ .* sqs_);
        bl = single(sps_ .* sq1s_);
        ur = single(sp1s_ .* sqs_);
        br = single(sp1s_ .* sq1s_);
        
        if isComplex == true % complex
            subVolumesG = padarray(subVolumesG, [1 1 0 0], complex(0.0, 0.0), 'both');
            obj.dataG = myCUDA.fuseAddBIComplexGPU_mex(subVolumesG, obj.dataG, uint32(anchorULYs), uint32(anchorULXs), ...
                ul, bl, ur, br);
        else % single
            subVolumesG = padarray(subVolumesG, [1 1 0 0], 0.0, 'both');
            obj.dataG = myCUDA.fuseAddBIRealGPU_mex(subVolumesG, obj.dataG, uint32(anchorULYs), uint32(anchorULXs), ...
                ul, bl, ur, br);
        end
        % subVolumesG: (nYX+2, nYX+2, nSpotsParallel, nSlices)
    else
        if isComplex
            obj.dataG = myCUDA.fuseAddComplexGPU_mex(subVolumesG, obj.dataG, uint32(anchorULYs), uint32(anchorULXs));
        else
            obj.dataG = myCUDA.fuseAddRealGPU_mex(subVolumesG, obj.dataG, uint32(anchorULYs), uint32(anchorULXs));
        end
    end
end


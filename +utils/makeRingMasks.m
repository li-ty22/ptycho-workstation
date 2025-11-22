function masks = makeRingMasks(nYX, radiusInners, radiusOuters, varargin)
    % make a ring mask with given radiusInners/Outers(pix), the size is nYX x nYX (YX)
    % use raised-cosine, by default, beta is 0.0
    % if radiusInners/Outers is an array, then generate multiple masks (nYX, nYX, nMasks)
    % optional parameter:
    % beta: for raised-cosine, roll-off factor
    % center: (y, x), all the masks share the same center
    
    centerYX = floor(nYX / 2) + 1;
    p = inputParser;
    addParameter(p, 'center', [centerYX, centerYX]);
    addParameter(p, 'beta', 0.0);
    parse(p, varargin{:});
    center = p.Results.center;
    beta = p.Results.beta;

    nMasks = numel(radiusInners);
    if nMasks ~= numel(radiusOuters)
        error('The number of the masks is wrong!');
    end

    import utils.makeCircleMasks;

    innerMasks = makeCircleMasks(nYX, radiusInners, 'center', center, 'beta', beta);
    outerMasks = makeCircleMasks(nYX, radiusOuters, 'center', center, 'beta', beta);
    masks = outerMasks - innerMasks;
end


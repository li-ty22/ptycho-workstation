function masks = makeShellMasks(nYX, radiusInner, radiusOuter, varargin)
    % make a shell mask with given radiusInners/Outers(pix), the size is nYX x nYX x nYX (YXZ)
    % use raised-cosine, by default, beta is 0.0
    % if radiusInners/Outers is an array, then generate multiple masks (nYX, nYX, nYX, nMasks)
    % optional parameter:
    % beta: for raised-cosine, roll-off factor
    % center: (y, x, z), all the masks share the same center

    centerYXZ = floor(nYX / 2) + 1;
    p = inputParser;
    addParameter(p, 'center', [centerYXZ, centerYXZ, centerYXZ]);
    addParameter(p, 'beta', 0.0);
    parse(p, varargin{:});
    center = p.Results.center;
    beta = p.Results.beta;

    import utils.makeSphereMasks;

    innerMasks = makeSphereMasks(nYX, radiusInner, 'center', center, 'beta', beta);
    outerMasks = makeSphereMasks(nYX, radiusOuter, 'center', center, 'beta', beta);
    masks = outerMasks - innerMasks;
end


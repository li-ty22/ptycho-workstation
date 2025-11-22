function masks = makeCircleMasks(nYX, radiuses, varargin)
    % make a circle mask with given radiuses(pix), the size is nYX x nYX (YX)
    % use raised-cosine, by default, beta is 0.0
    % if radiuses is an array, then generate multiple masks (nYX, nYX, nMasks)
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
    
    nMasks = numel(radiuses);
    radiuses = reshape(radiuses, 1, 1, nMasks);

    [X, Y] = meshgrid(1:nYX);
    X = X - center(2);
    Y = Y - center(1);
    dists = sqrt(X.^2 + Y.^2) ./ radiuses;
    
    masks = zeros(nYX, nYX, nMasks, 'single');
    masks(dists <= (1 - beta)) = 1.0;
    index = dists > (1 - beta) & dists < (1 + beta);
    masks(index) = 0.5 * (1 + cos(pi / 2 / beta * (dists(index) - (1 - beta))));
    
end


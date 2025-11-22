function masks = makeSphereMasks(nYX, radiuses, varargin)
    % make a sphere mask with given radiuses(pix), the size is nYX x nYX x nYX (YXZ)
    % use raised-cosine, by default, beta is 0.0
    % if radiuses is an array, then generate multiple masks (nYX, nYX, nYX, nMasks)
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

    nMasks = numel(radiuses);
    radiuses = reshape(radiuses, 1, 1, 1, nMasks);
    
    [X, Y, Z] = meshgrid(1:nYX, 1:nYX, 1:nYX);
    X = X - center(2);
    Y = Y - center(1);
    Z = Z - center(3);
    dist = sqrt(X.^2 + Y.^2 + Z.^2) ./ radiuses;
    
    masks = zeros(nYX, nYX, nYX, nMasks, 'single');
    masks(dist <= (1 - beta)) = 1.0;
    index = dist > (1 - beta) & dist < (1 + beta);
    masks(index) = 0.5 * (1 + cos(pi / 2 / beta * (dist(index) - (1 - beta))));
end


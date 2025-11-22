function initBase(obj, probN, pSizeF, waveLength, defocus, Cs, semiAngle, vacuumProbe, varargin)
    % varargin:
    % upsamplingProbFFactor: >= 1, upsampling in fourier space, enlarge the imgSizeR, then cropping in real space
    % betaRaisedCosine

    p = inputParser;
    addParameter(p, 'upsamplingFFactor', 1);
    addParameter(p, 'betaRaisedCosine', 0.0);
    parse(p, varargin{:});
    upsamplingFFactor = p.Results.upsamplingFFactor;
    betaRaisedCosine = p.Results.betaRaisedCosine;

    % allocate memory
    prob = complex(zeros(probN, probN, 1, 'single'), zeros(probN, probN, 1, 'single'));
    % init properties
    obj.pSize = 1 / pSizeF / probN;
    obj.waveLength = waveLength;
    obj.defocus = defocus;
    obj.Cs = Cs;
    obj.semiAngle = semiAngle;

    % init base probe
    imgN = ceil(upsamplingFFactor * probN);
    imgPSizeF = (pSizeF * probN) / imgN;
    center = floor(imgN / 2) + 1;
    % base probeF's Angle
    [U, V] = meshgrid(1:imgN); % if using single, it would be overflow!
    K = ((U - center) * imgPSizeF).^2 + ((V - center) * imgPSizeF).^2;
    part1 = pi * defocus * waveLength .* K; % C1:defocus
    part2 = pi / 2.0 * Cs * waveLength^3 .* K.^2; % C3:spherical aberration
    theta = part1 + part2;
    % base probeF's Abs
    imgR = utils.semiAngle2diskRadiusPix(semiAngle, waveLength, imgPSizeF);
    if isempty(vacuumProbe)
        probFAbs = utils.makeCircleMasks(imgN, imgR, 'beta', betaRaisedCosine);
    else
        if upsamplingFFactor > 1
            vacuumProbe = imresize(vacuumProbe, upsamplingFFactor, 'bilinear');
        end
        probFAbs = vacuumProbe;
    end
    % base prob after de-alias
    initBaseProbF = probFAbs .* exp(1i * single(theta));
    baseProb = fftshift(ifft2(ifftshift(initBaseProbF)));
    % crop probN
    ul = floor(imgN / 2) + 1 - floor(probN / 2);
    br = ul + probN - 1;
    prob(:, :, 1) = baseProb(ul:br, ul:br);
    % normalization: set the total intensity of base probe to 1.0
    prob(:, :, 1) = prob(:, :, 1) / sqrt(sum(abs(prob(:, :, 1)).^2, 'all'));
    obj.probG = gpuArray(prob);

    obj.initUVG();
end
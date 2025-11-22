function initMultiModes(obj, nModes)
    % need to init base probe first
    % use Laguerre-Gaussian method to generate different modes
    % all the modes' intensities are set to 1.0

    % allocate memory and set the base probe
    baseProb = obj.getProbIm(1); % CPU
    probN = obj.nYX();
    prob = complex(zeros(probN, probN, nModes, 'single'), zeros(probN, probN, nModes, 'single'));

    % the mass center of the base probe
    [X, Y] = meshgrid(single(0:probN-1));
    totalIntensityBaseProbe = sum(abs(baseProb).^2, 'all');
    mcY = sum(Y .* abs(baseProb).^2, 'all') / totalIntensityBaseProbe;
    mcX = sum(X .* abs(baseProb).^2, 'all') / totalIntensityBaseProbe;

    % the second momentum of the base probe
    smY = sum((Y - mcY).^2 .* abs(baseProb).^2, 'all') / totalIntensityBaseProbe;
    smX = sum((X - mcX).^2 .* abs(baseProb).^2, 'all') / totalIntensityBaseProbe;

    % Laguerre-Gaussian
    modeTEM = ceil(sqrt(nModes));
    nModeDone = 0;
    secondaryBreak = false;
    for m = 0:modeTEM-1
        for n = 0:modeTEM-1
            if nModeDone < nModes
                iMode = nModeDone + 1;
                prob(:, :, iMode) = (Y - mcY).^m .* (X - mcX).^n .* baseProb;
                if nModeDone == 0 % base
                    prob(:, :, 1) = prob(:, :, 1) / sqrt(sum(abs(prob(:, :, 1)).^2, 'all'));
                else % slave
                    prob(:, :, iMode) = prob(:, :, iMode) .* ...
                        exp(- (Y-mcY).^2 / (2*smY)) .* exp(- (X-mcX).^2 / (2*smX));
                    prob(:, :, iMode) = prob(:, :, iMode) / sqrt(sum(abs(prob(:, :, iMode)).^2, 'all'));
                end
            else
                secondaryBreak = true;
                break;
            end
            nModeDone = nModeDone + 1;
        end
        if secondaryBreak == true
            break;
        end
    end

    % move data to GPU
    obj.probG = gpuArray(prob);

end


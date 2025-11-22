function [cbed] = unzipCBED(ind, value, s)
    % s means cbed's size (nVU, nVU, nSpots)
    cbed = zeros(s, 'single');
    cbed(ind) = value;
end


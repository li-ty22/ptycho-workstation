function [ind, value] = zipCBED(cbedIn)
    % only record non-zero pixel
    ind = find(cbedIn);
    value = cbedIn(ind);
end


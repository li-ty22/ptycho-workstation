function [dataOut, zeroInd] = safeDivide(data1, data2, val)
% dataOut = data1 ./ data2
% if an element in data2 is equal to 0.0, return val to dataOut's element corresponding to that in data2
    zeroInd = abs(abs(data2) - 0.0) < eps(0.5);
    dataOut = data1;
    dataOut(zeroInd) = val;
    dataOut(~zeroInd) = data1(~zeroInd) ./ data2(~zeroInd);
end


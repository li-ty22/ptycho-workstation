function [QG, RG] = mgsGPU(AG)
% modified Gram-Schmidt method / QR decomposition of GPU version
% input example: AG = [1, 2; 3, 4; 5, 6], orthogonalize the vector [1;3;5]
% with the vector [2;4;6]
    [m, n] = size(AG);
    QG = zeros(m, n, 'single', 'gpuArray');
    QG(:, 1) = AG(:, 1);
    RG = zeros(n, 'single', 'gpuArray');
    RG(1, 1) = 1;
    for k = 1:n
        RG(k, k) = norm(AG(:, k));
        QG(:, k) = AG(:, k) / RG(k, k);
        for j = k+1:n
            RG(k, j) = QG(:, k)' * AG(:, j);
            AG(:, j) = AG(:, j) - RG(k, j) * QG(:, k);
        end
    end
end


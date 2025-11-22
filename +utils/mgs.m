function [Q, R] = mgs(A)
% modified Gram-Schmidt method / QR decomposition
% input example: A = [1, 2; 3, 4; 5, 6], orthogonalize the vector [1;3;5]
% with the vector [2;4;6]
    [m, n] = size(A);
    Q = zeros(m, n, 'single');
    Q(:, 1) = A(:, 1);
    R = zeros(n, 'single');
    R(1, 1) = 1;
    for k = 1:n
        R(k, k) = norm(A(:, k));
        Q(:, k) = A(:, k) / R(k, k);
        for j = k+1:n
            R(k, j) = Q(:, k)' * A(:, j);
            A(:, j) = A(:, j) - R(k, j) * Q(:, k);
        end
    end
end


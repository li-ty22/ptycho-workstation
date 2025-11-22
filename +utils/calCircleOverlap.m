function overlap = calCircleOverlap(radius, stepSize)
    if stepSize >= 2 * radius
        overlap = 0.0;
    elseif stepSize < 0.0 || radius <= 0.0
        error('Wrong stepSize or radius');
    else
        alphaRad = 2 * acos(stepSize / 2 / radius);
        sectorArea = alphaRad / (2 * pi) * pi * radius^2;
        triangleArea = stepSize / 2 * (radius * sin(alphaRad / 2));
        overlapArea = 2 * (sectorArea - triangleArea);
        overlap = overlapArea / (pi * radius^2);
    end
end


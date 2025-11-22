function shrinkageObjGPU(obj, oObjG, object_shrinkage_maskG)
%SHRINKAGEOBJGPU
    % object_shrinkage_maskG: (objNY, objNX, 1), 0 or 1
    switch obj.objModel
        case 'complex'
            objAbsG = abs(oObjG.dataG);
            objAngG = angle(oObjG.dataG);
            objAngG = objAngG - sum(objAngG .* object_shrinkage_maskG, [1 2]) ./ sum(object_shrinkage_maskG, [1 2]);
            oObjG.dataG = objAbsG .* exp(1i .* objAngG);
        case 'potential'
            oObjG.dataG = oObjG.dataG - sum(oObjG.dataG .* object_shrinkage_maskG, [1 2]) ./ sum(object_shrinkage_maskG, [1 2]);
        otherwise
            error('Wrong object model!');
    end
end


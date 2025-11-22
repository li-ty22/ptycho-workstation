function updateGPU(obj, oObjG, oProbG, updateObj, updateProb)
    if updateObj
        switch obj.objModel
            case 'potential'
                oObjG.dataG = oObjG.dataG + obj.oObjGradAllPotG.dataG ./ max(obj.oObj_probe_normalizationG.dataG, [], [1 2]);
            case 'complex'
                oObjG.dataG = oObjG.dataG + obj.oObjGradAllCG.dataG ./ max(obj.oObj_probe_normalizationG.dataG, [], [1 2]);
            otherwise
                error('Wrong object model!');
        end
    end
    if updateProb
        oProbG.probG = oProbG.probG + obj.oProbGradAllG.probG  ./ max(obj.object_normalizationG, [], [1 2]);
    end
end
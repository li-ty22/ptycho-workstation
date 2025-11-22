classdef cCBEDReconGSimplified < handle
    properties
        cbedBeatsG % cell (nBeats, 1), each element is a (nYX, nYX, nSpotsParallel) gpuArray
        % use ifftshift-ed CBEDs to reduce fftshift/ifftshift operation in forwarding/backwarding calculation (fftshift/ifftshift are time-consuming)
        cbedBeats % like cbedSubBlocksG, cpu
        beats % same as the beats in cPtychoReconParas, (nSpotsParallel, nBeats)
        isCBEDOnGPU
    end
    methods
        function obj = cCBEDReconGSimplified()
            obj.beats = [];
            obj.cbedBeatsG = {};
            obj.cbedBeats = {};
            obj.isCBEDOnGPU = true;
        end
        function setBeats(obj, oCBED, beats, isCBEDOnGPU)
            obj.beats = beats;
            nBeats = size(obj.beats, 2);
            obj.isCBEDOnGPU = isCBEDOnGPU;
            if obj.isCBEDOnGPU
                obj.cbedBeatsG = cell(nBeats, 1);
            else
                obj.cbedBeats = cell(nBeats, 1);
            end
            for iBeat = 1:nBeats
                cbeds1Beat = oCBED.getSubset(obj.beats(:, iBeat)); % (nYX, nYX, nSpotsParallel)
                cbeds1Beat = ifftshift(ifftshift(cbeds1Beat, 1), 2); % ifftshift
                if obj.isCBEDOnGPU
                    obj.cbedBeatsG{iBeat} = gpuArray(cbeds1Beat); % (nYX, nYX, 1, nSpotsParallel), gpu
                else
                    obj.cbedBeats{iBeat} = cbeds1Beat;
                end
            end
        end
        function dataG = getBeatG(obj, iBeat)
            if obj.isCBEDOnGPU
                dataG = obj.cbedBeatsG{iBeat};
            else
                dataG = gpuArray(obj.cbedBeats{iBeat});
            end
        end
        function dataG = getSubBeatG(obj, iBeat, iSubBeat, nSubSpotsParallel)
            lind = (iSubBeat - 1) * nSubSpotsParallel + 1;
            uind = iSubBeat * nSubSpotsParallel;
            nSpotsParallel = size(obj.beats, 1);
            if lind < 1 || uind > nSpotsParallel
                error('Wrong subbeat ind!');
            end
            if obj.isCBEDOnGPU
                data1BeatG = obj.cbedBeatsG{iBeat};
                dataG = data1BeatG(:, :, lind:uind);
            else
                data1Beat = obj.cbedBeats{iBeat};
                dataG = gpuArray(data1Beat(:, :, lind:uind));
            end
        end
    end
end
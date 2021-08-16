function lgraph = addAndConnectResNetSection( ...
    lgraph, ...
    layerToConnectFrom, ...
    sectionName, ...
    numF1x1first, ...
    numF3x3, ...
    numF1x1last, ...
    firstStride, ...
    residualUnits)

stride = firstStride;
for i = 1:length(residualUnits)   
    layerRoot = sectionName+residualUnits{i};
    bnRoot = "bn"+extractAfter(sectionName,"res")+residualUnits{i};

    resnetBlock = [
        convolution2dLayer(1,numF1x1first,"Stride",stride,"Name",layerRoot+"_branch2a","BiasLearnRateFactor",0)
        batchNormalizationLayer("Name",bnRoot+"_branch2a")
        reluLayer("Name",layerRoot+"_branch2a_relu")
        
        convolution2dLayer(3,numF3x3,"Padding",1,"Name",layerRoot+"_branch2b","BiasLearnRateFactor",0);
        batchNormalizationLayer("Name",bnRoot+"_branch2b")
        reluLayer("Name",layerRoot+"_branch2b_relu")
        
        convolution2dLayer(1,numF1x1last,"Name",layerRoot+"_branch2c","BiasLearnRateFactor",0)
        batchNormalizationLayer("Name",bnRoot+"_branch2c")
        
        additionLayer(2,"Name",layerRoot)
        reluLayer("Name",layerRoot+"_relu")];
    
    lgraph = addLayers(lgraph,resnetBlock);
    
    if i == 1
        projectionLayers = [
            convolution2dLayer(1,numF1x1last,"Stride",stride,"Name",layerRoot+"_branch1","BiasLearnRateFactor",0)
            batchNormalizationLayer("Name",bnRoot+"_branch1")];
        lgraph = addLayers(lgraph,projectionLayers);
        
        lgraph = connectLayers(lgraph,layerToConnectFrom,layerRoot+"_branch2a");
        lgraph = connectLayers(lgraph,layerToConnectFrom,layerRoot+"_branch1");
        lgraph = connectLayers(lgraph,bnRoot+"_branch1",layerRoot+"/in2");
    else
        lgraph = connectLayers(lgraph,sectionName+residualUnits{i-1}+"_relu",sectionName+residualUnits{i}+"_branch2a");
        lgraph = connectLayers(lgraph,sectionName+residualUnits{i-1}+"_relu",sectionName+residualUnits{i}+"/in2");
    end
    
    stride = 1;
end
end



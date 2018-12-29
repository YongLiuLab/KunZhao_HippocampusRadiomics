function G=GLAM(IT,nL)
  %calculaition matrix in 2d
  % implemented 10 directions
 warning off %#ok<WNOFF>
 sz=size(IT);
  
  d=1;  %delta: distance
  
  dao=[0 d; -d d; -d 0; -d -d]; %offset: distance and orientation
  numMatrix=size(dao,1);  %number of DAOs
  glcmXY=zeros(nL,nL,numMatrix,sz(3));       
  parfor i=1:sz(3)  %xy plane
    cIT=squeeze(IT(:,:,i)); %xy plane
    [glcmXY(:,:,:,i),~] = graycomatrix(cIT,'NumLevels',nL,'Offset',dao,'G',[],'Symmetric',true);    %%灰度的一个共生矩阵，考验的是灰度之间的差异性
  end
  
  dao=[0 d; -d d; -d -d]; %offset: distance and orientation
  numMatrix=size(dao,1);  %number of DAOs
  glcmYZ=zeros(nL,nL,numMatrix,sz(1));
  parfor i=1:sz(1)  %yz plane
    cIT=squeeze(IT(i,:,:)); 
    [glcmYZ(:,:,:,i),~] = graycomatrix(cIT,'NumLevels',nL,'Offset',dao,'G',[],'Symmetric',true);  
  end
  %每一层上求一个灰度共生矩阵，然后求一下均值
  dao=[-d d; -d 0; -d -d]; %offset: distance and orientation
  numMatrix=size(dao,1);  %number of DAOs
  glcmXZ=zeros(nL,nL,numMatrix,sz(2));
  parfor i=1:sz(2)  %xz plane
    cIT=squeeze(IT(:,i,:)); %xy plane
    [glcmXZ(:,:,:,i),~] = graycomatrix(cIT,'NumLevels',nL,'Offset',dao,'G',[],'Symmetric',true);  
  end
  
  meanG=cat(3,mean(glcmXY,4),mean(glcmYZ,4),mean(glcmXZ,4));%mean over all slices
  GLCMall = GLCM_Features4(meanG);
  
%   G.Autocorrelation=GLCMall.autoc;
%   G.ClusterProminence=GLCMall.cprom;
%   G.ClusterShade=GLCMall.cshad;
%   G.ClusterTendency=GLCMall.ctend;
%   G.Contrast=GLCMall.contr;
%   G.Correlation=GLCMall.corrp;
%   G.DifferenceEntropy=GLCMall.denth;
%   G.Dissimilarity=GLCMall.dissi;
%   G.Energy=GLCMall.energ;
%   G.Entropy=GLCMall.entro;
%   G.Homogeneity1=GLCMall.homom;
%   G.Homogeneity2=GLCMall.homop;
%   G.IMC1=GLCMall.inf1h;
%   G.IMC2=GLCMall.inf2h;
%   G.IDMN=GLCMall.idmnc;
%   G.IDN=GLCMall.indnc;
%   G.InverseVariance=GLCMall.invva;
%   G.MaximumProbability=GLCMall.maxpr;
%   G.SumAverage=GLCMall.savgh;
%   G.SumEntropy=GLCMall.senth;
%   G.SumVariance=GLCMall.svarh;
%   G.Variance=GLCMall.sosvh;

  G.Autocorrelation=mean(GLCMall.autoc); 
  G.ClusterProminence=mean(GLCMall.cprom);
  G.ClusterShade=mean(GLCMall.cshad);
  G.ClusterTendency=mean(GLCMall.ctend);
  G.Contrast=mean(GLCMall.contr);
  G.Correlation=mean(GLCMall.corrp);
  G.DifferenceEntropy=mean(GLCMall.denth);
  G.Dissimilarity=mean(GLCMall.dissi);
  G.Energy=mean(GLCMall.energ);
  G.Entropy=mean(GLCMall.entro);
  G.Homogeneity1=mean(GLCMall.homom);
  G.Homogeneity2=mean(GLCMall.homop);
  G.IMC1=mean(GLCMall.inf1h);
  G.IMC2=mean(GLCMall.inf2h);
  G.IDMN=mean(GLCMall.idmnc);
  G.IDN=mean(GLCMall.indnc);
  G.InverseVariance=mean(GLCMall.invva);
  G.MaximumProbability=mean(GLCMall.maxpr);
  G.SumAverage=mean(GLCMall.savgh);
  G.SumEntropy=mean(GLCMall.senth);
  G.SumVariance=mean(GLCMall.svarh);
  G.Variance=mean(GLCMall.sosvh);
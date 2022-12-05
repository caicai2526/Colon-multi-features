function num = Wilkcoxnew(datafile, trainlabel, features_num)
%   This file prints an array of the discriminatory features
%   datafile = 'ovarian_61902.data'
%   xvalues = 'ovarian_61902.names2.csv'

data = datafile;
ind1 = find(trainlabel ==1);
ind0= find(trainlabel ==0);
controldata = data;
cancerdata = data;
controldata(ind1,:) = [];
cancerdata(ind0,:) = [];
lx = size(datafile,2);
s1 = zeros(1,lx);
for i=1:lx
    [P,H, stat] = ranksum(controldata(:,i),cancerdata(:,i));
    [P,H, stat2] = ranksum(cancerdata(:,i),controldata(:,i));
    s1(i) = min(stat.ranksum,stat2.ranksum);
    s1(i) = P;
end
[a b] = sort(s1,'ascend');
num = b(:,1:features_num);

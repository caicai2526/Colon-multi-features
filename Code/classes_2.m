clear;
clc;

[num, txt, raw] = xlsread('./survival_data/2011_3.xlsx', 'A1:L283');
image_path  = '../3000X3000_change_name/';
file = dir([image_path, '*.tif']);
% for i = 1 : numel(file)
%     image = imread([image_path,file(i).name]);
%     imwrite(image,['../3000X3000_change_name/',file(i).name(1:9), '.tif'])
% end
for i = 1 : numel(num(:,9))
    try
        if num(i, 9)>=60
            name = raw{i+1,1};
            image = imread([image_path,name,'.tif']);
            imwrite(image, ['../data_2_classes/long_sur/', name, '.tif']);
        elseif num(i,9)<60
            name = raw{i+1,1};
            image = imread([image_path,name,'.tif']);
            imwrite(image, ['../data_2_classes/short_sur/', name, '.tif']);
        end
    end
end

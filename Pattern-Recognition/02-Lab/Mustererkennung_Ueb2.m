close all;
clear;
clc;

% Autor: Preslava Peshkova

% Mustererkennung Übung 2

%% Einlesen der Bilder

% Ordner navigieren
addpath(genpath('src/'))
addpath(genpath('data/'))

% Bild Nummer
Bild_Training = '3_12';
Bild_Test = '3_13';

% RGB
% RGBIR
d_RGBIR_training = imread(['data/4_Ortho_RGBIR/top_potsdam_',Bild_Training,'_RGBIR.tif']);
d_RGBIR_test = imread(['data/4_Ortho_RGBIR/top_potsdam_',Bild_Test,'_RGBIR.tif']);
% nDSM
d_nDSM_training = imread(['data/1_DSM_normalisation/dsm_potsdam_0',Bild_Training,'_normalized_lastools.jpg']);
d_nDSM_test = imread(['data/1_DSM_normalisation/dsm_potsdam_0',Bild_Test,'_normalized_lastools.jpg']);
% gt
d_GT_training = imread(['data/5_Labels_all/top_potsdam_',Bild_Training,'_label.tif']);
d_GT_test = imread(['data/5_Labels_all/top_potsdam_',Bild_Test,'_label.tif']);


%% Datentyp und Bildgröße

% Anpassen des Datentyps und Normierung
RGBIR_training = single(d_RGBIR_training)/255;
RGBIR_test = single(d_RGBIR_test)/255;
RGB_training = RGBIR_training(:,:,1:3);
RGB_test = RGBIR_test(:,:,1:3);
IR_training = RGBIR_training(:,:,4);
IR_test = RGBIR_test(:,:,4);

nDSM_training = single(d_nDSM_training);
nDSM_test = single(d_nDSM_test);
nDSM_training = (nDSM_training - min(nDSM_training(:))) / (max(nDSM_training(:)) - min(nDSM_training(:)));
nDSM_test = (nDSM_test - min(nDSM_test(:))) / (max(nDSM_test(:)) - min(nDSM_test(:)));

% Get class labels
gt_training = rgbLabel2classLabel(d_GT_training);
gt_test = rgbLabel2classLabel(d_GT_test);
d_GT_training = double(d_GT_training);
d_GT_test = double(d_GT_test);

% Number of label class
num_class = 6;

% Run Time
Run_time = zeros(2,1);    % Chess, SLIC

% Bildgröße wird bei Ueb2 nicht verändert!

% Chessboard
segment_size = 30; %[pixel]

% SLIC
num_seg = 50000; % Planned Segment Count

% Random forest
num_trees = 7;

%% Chessboard
% Calculate chessboard label mask
num_rows = size(gt_training,1) / segment_size;
num_cols = size(gt_training,2) / segment_size;
numbers = reshape(1:num_rows * num_cols, num_rows, num_cols)';
mask = single(kron(numbers, ones(segment_size))');

% Label idx
idx_chess = label2idx(mask);
pixel = size(gt_training,1) * size(gt_training,2);
mean_rgb = RGB_training;
maj_label_training = gt_training;
chess_feature_training = zeros(num_rows * num_cols, 5);
chess_label_training = zeros(num_rows * num_cols, 1);

for i = 1:num_rows * num_cols
    % Calculate features
    mean_r = mean(RGB_training(idx_chess{i}));
    mean_g = mean(RGB_training(idx_chess{i} + pixel));
    mean_b = mean(RGB_training(idx_chess{i} + 2*pixel));
    mean_ir = mean(IR_training(idx_chess{i}));
    mean_dsm = mean(nDSM_training(idx_chess{i}));
    
    % RGB feature image
    mean_rgb(idx_chess{i}) = mean_r;
    mean_rgb(idx_chess{i} + pixel) = mean_g;
    mean_rgb(idx_chess{i} + 2*pixel) = mean_b;
    
    % Calculate feature vector
    chess_feature_training(i,:) = [mean_r mean_g mean_b mean_ir mean_dsm];
    
    % Calculate majority vote
	h_counts = histcounts(gt_training(idx_chess{i}), 1/2:1:num_class+1/2);
	[~, col] = max(h_counts);
	maj_label_training(idx_chess{i}) = col;
    
    % Label vector
    chess_label_training(i) = col;
end

% Label2RGB
maj_label_chess_training = classLabel2rgbLabel(maj_label_training);

% Plot Label Image
figure;
imshow(maj_label_chess_training);
title('Chessboard: Label Vector Image');
if ~isfile('Chessboard_Label_Vektor.png')
saveas(gcf,'Chessboard_Label_Vektor.png');
end

maj_label_test = gt_test;
chess_feature_test = zeros(num_rows * num_cols, 5);
chess_label_test = zeros(num_rows * num_cols, 1);

for i = 1:num_rows * num_cols
    % Calculate features
    mean_r = mean(RGB_test(idx_chess{i}));
    mean_g = mean(RGB_test(idx_chess{i} + pixel));
    mean_b = mean(RGB_test(idx_chess{i} + 2*pixel));
    mean_ir = mean(IR_test(idx_chess{i}));
    mean_dsm = mean(nDSM_test(idx_chess{i}));
    
    % RGB feature image
    mean_rgb(idx_chess{i}) = mean_r;
    mean_rgb(idx_chess{i} + pixel) = mean_g;
    mean_rgb(idx_chess{i} + 2*pixel) = mean_b;
    
    % Calculate feature vector
    chess_feature_test(i,:) = [mean_r mean_g mean_b mean_ir mean_dsm];
    
    % Calculate majority vote
	h_counts = histcounts(gt_test(idx_chess{i}), 1/2:1:num_class+1/2);
	[~, col] = max(h_counts);
	maj_label_test(idx_chess{i}) = col;
    
    % Label vector
    chess_label_test(i) = col;
end

% Label2RGB
maj_label_chess_test = classLabel2rgbLabel(maj_label_test);

% Plot Label Image
figure;
imshow(maj_label_chess_test);
title('Chessboard: Label Vector Image');
if ~isfile('Chessboard_Label_Vektor_test.png')
saveas(gcf,'Chessboard_Label_Vektor_test.png');
end


%% SLIC
% Calculate SLIC label mask
[mask_training, num_seg_training] = superpixels(RGB_training, num_seg);

% Feature and Label vector
slic_feature_training = zeros(num_seg_training, 3);
slic_label_training = zeros(num_seg_training, 1);

% Mask
maj_label_training = gt_training;

% Label idx
idx_slic_training = label2idx(mask_training);

for i = 1:num_seg_training
    % Calculate features
    mean_r = mean(RGB_training(idx_slic_training{i}));
    mean_g = mean(RGB_training(idx_slic_training{i} + pixel));
    mean_b = mean(RGB_training(idx_slic_training{i} + 2*pixel));
    
    % RGB feature image
    mean_rgb(idx_slic_training{i}) = mean_r;
    mean_rgb(idx_slic_training{i} + pixel) = mean_g;
    mean_rgb(idx_slic_training{i} + 2*pixel) = mean_b;
    
    % Calculate feature vector
    slic_feature_training(i,:) = [mean_r mean_g mean_b];
    
    % Calculate majority vote
	h_counts = histcounts(gt_training(idx_slic_training{i}), 1/2:1:num_class+1/2);
	[~, col] = max(h_counts);
	maj_label_training(idx_slic_training{i}) = col;
    
    % Label vector
    slic_label_training(i) = col;
end

% Label2RGB
maj_label_slic_training = classLabel2rgbLabel(maj_label_training);

% Plot Label Image
figure;
imshow(maj_label_slic_training);
title('SLIC: Label Vector Image');
if ~isfile('slic_Label_Vektor.png')
saveas(gcf,'slic_Label_Vektor.png');
end

% Calculate SLIC label mask
[mask_test, num_seg_test] = superpixels(RGB_test, num_seg);

% Feature and Label vector
slic_feature_test = zeros(num_seg_test, 3);
slic_label_test = zeros(num_seg_test, 1);

% Mask
maj_label_test = gt_test;

% Label idx
idx_slic_test = label2idx(mask_test);

for i = 1:num_seg_test
    % Calculate features
    mean_r = mean(RGB_test(idx_slic_test{i}));
    mean_g = mean(RGB_test(idx_slic_test{i} + pixel));
    mean_b = mean(RGB_test(idx_slic_test{i} + 2*pixel));
    
    % RGB feature image
    mean_rgb(idx_slic_test{i}) = mean_r;
    mean_rgb(idx_slic_test{i} + pixel) = mean_g;
    mean_rgb(idx_slic_test{i} + 2*pixel) = mean_b;
    
    % Calculate feature vector
    slic_feature_test(i,:) = [mean_r mean_g mean_b];
    
    % Calculate majority vote
	h_counts = histcounts(gt_test(idx_slic_test{i}), 1/2:1:num_class+1/2);
	[~, col] = max(h_counts);
	maj_label_test(idx_slic_test{i}) = col;
    
    % Label vector
    slic_label_test(i) = col;
end

% Label2RGB
maj_label_slic_test = classLabel2rgbLabel(maj_label_test);

% Plot Label Image
figure;
imshow(maj_label_slic_test);
title('SLIC: Label Vector Image');
if ~isfile('slic_Label_Vektor_test.png')
saveas(gcf,'slic_Label_Vektor_test.png');
end

%% Random Forest
% Chess
tic;
chess_train_mdl = TreeBagger(num_trees, chess_feature_training, chess_label_training); % Training
Run_time(1) = toc;
% Slic
tic;
slic_train_mdl = TreeBagger(num_trees, slic_feature_training, slic_label_training); % Training
Run_time(2) = toc;

% Prediction
% Chess
tic;
chess_prediction_train = predict(chess_train_mdl, chess_feature_training);
chess_prediction_train = str2num(cell2mat(chess_prediction_train));
chess_prediction_test = predict(chess_train_mdl, chess_feature_test);
chess_prediction_test = str2num(cell2mat(chess_prediction_test));
Run_time(1) = Run_time(1) + toc;

% Slic
tic;
slic_prediction_train = predict(slic_train_mdl, slic_feature_training);
slic_prediction_train = str2num(cell2mat(slic_prediction_train));
slic_prediction_test = predict(slic_train_mdl, slic_feature_test);
slic_prediction_test = str2num(cell2mat(slic_prediction_test));
Run_time(2) = Run_time(2) + toc;

% Create image from labels
chess_label_train = zeros(size(maj_label_training));
slic_label_train = zeros(size(maj_label_training));
chess_label_test2 = zeros(size(maj_label_test));
slic_label_test2 = zeros(size(maj_label_test));

tic;
for i = 1:num_rows * num_cols
    % Chess
    chess_label_train(idx_chess{i}) = chess_prediction_train(i);
    chess_label_test2(idx_chess{i}) = chess_prediction_test(i);
end
Run_time(1) = Run_time(1) + toc;
tic;
for i = 1:num_seg_training
    % Slic
    slic_label_train(idx_slic_training{i}) = slic_prediction_train(i);
end
for i = 1:num_seg_test
    % Slic
    slic_label_test2(idx_slic_test{i}) = slic_prediction_test(i);
end
Run_time(2) = Run_time(2) + toc;

% Convert to label
slic_label_train_rgb = classLabel2rgbLabel(slic_label_train);
chess_label_train_rgb = classLabel2rgbLabel(chess_label_train);

slic_label_test_rgb = classLabel2rgbLabel(slic_label_test2);
chess_label_test_rgb = classLabel2rgbLabel(chess_label_test2);

% Plot Label Image
figure;
imshow(chess_label_train_rgb);
title('Chess: Predicted Image');
if ~isfile('chess_predicted_train.png')
saveas(gcf,'chess_predicted_train.png');
end

% Plot Label Image
figure;
imshow(chess_label_test_rgb);
title('Chess: Predicted Image');
if ~isfile('chess_predicted_test.png')
saveas(gcf,'chess_predicted_test.png');
end

% Plot Label Image
figure;
imshow(slic_label_train_rgb);
title('SLIC: Predicted Image');
if ~isfile('slic_predicted_train.png')
saveas(gcf,'slic_predicted_train.png');
end

% Plot Label Image
figure;
imshow(slic_label_test_rgb);
title('SLIC: Predicted Image');
if ~isfile('slic_predicted_test.png')
saveas(gcf,'slic_predicted_test.png');
end

% Evaluation: Chessboard
d_chess_train = abs(chess_label_train - gt_training);
d_chess_test = abs(chess_label_test2 - gt_test);

acc_chess_train = length(find(d_chess_train == 0)) / (size(d_chess_train, 1) * size(d_chess_train, 2));
acc_chess_test = length(find(d_chess_test == 0)) / (size(d_chess_test, 1) * size(d_chess_test, 2));

d_chess_train = classLabel2rgbLabel(d_chess_train + 1);
d_chess_test = classLabel2rgbLabel(d_chess_test + 1);


% Evaluation: SLIC
d_slic_train = abs(slic_label_train - gt_training);
d_slic_test = abs(chess_label_test2 - gt_test);

acc_slic_train = length(find(d_slic_train == 0)) / (size(d_slic_train, 1) * size(d_slic_train, 2));
acc_slic_test = length(find(d_slic_test == 0)) / (size(d_slic_test, 1) * size(d_slic_test, 2));
                              
d_slic_train = classLabel2rgbLabel(d_slic_train + 1);
d_slic_test = classLabel2rgbLabel(d_slic_test + 1);


% Plot Image
figure;
imshow(d_slic_test);
title('SLIC: Difference Image');
if ~isfile('slic_diff_test.png')
saveas(gcf,'slic_diff_test.png');
end
figure;
imshow(d_slic_train);
title('SLIC: Difference Image');
if ~isfile('slic_diff_train.png')
saveas(gcf,'slic_diff_train.png');
end

figure;
imshow(d_chess_test);
title('SLIC: Difference Image');
if ~isfile('chess_diff_test.png')
saveas(gcf,'chess_diff_test.png');
end
figure;
imshow(d_chess_train);
title('SLIC: Difference Image');
if ~isfile('chess_diff_train.png')
saveas(gcf,'chess_diff_train.png');
end


%% Confusion Matrix

label_vector = chess_label_training;
prediction_vector = chess_prediction_train;
colormap_labels = {'Impervious surface', 'Building', 'Low vegetation', 'Tree', 'Car', 'Clutter'};
method = 'Chess Training';

figure('Name',['Confusion Matrix - ', method])
hold on
title(['Confusion Matrix - ', method])
cm = confusionmat(label_vector,prediction_vector);

% Totals
total_r = sum(cm,2);
total_c = sum(cm)';
total_diag = sum(diag(cm));
total_cm = zeros(num_class + 1,num_class + 1);
total_cm(1:num_class,1:num_class) = cm;
total_cm(end,1:num_class) = total_r;
total_cm(1:num_class,end) = total_c;
total_cm(end,end) = total_diag;

prod_acc = diag(cm) ./ total_r * 100;
user_acc = diag(cm) ./ total_c * 100;

% Percentages
cm_percent = zeros(num_class + 1,num_class + 1);
cm_percent(1:num_class,1:num_class) = cm./repmat(sum(cm, 1),num_class,1) * 100;
cm_percent(end,1:num_class) = user_acc;
cm_percent(1:num_class,end) = prod_acc;
cm_percent(end,end) = total_diag/sum(total_c) * 100;


imagesc(cm_percent([(num_class + 1):-1:1], [1:(num_class + 1)]));
Mycolors=[1 1 1; 0.3 0.3 0.3];
colormap(Mycolors);
textStrings = num2str([cm_percent(:)], '%.1f%%\n');
textStrings_total = num2str([total_cm(:)], '%i\n');

textStrings = strtrim(cellstr(textStrings));
textStrings_total = strtrim(cellstr(textStrings_total));


[x,y] = meshgrid(1:(num_class + 1),num_class + 1:-1:1);
x = [x(:), x(:)];
y = [y(:)-0.2, y(:)+0.2];
hStrings = text(x(:),y(:),[textStrings(:),textStrings_total(:)], ...
    'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));
textColors = repmat(cm_percent(:) > midValue,1,3);
set(gca,'xaxisLocation','top')
set(hStrings,{'Color'},repmat(num2cell(textColors,2),2,1));
set(gca,'XTick',1:(num_class + 1),...
    'XTickLabel',[colormap_labels,{'Producer´s Accuracy'}],... 
    'YTick',1:(num_class + 1),... 
    'YTickLabel',[{'User´s Accuracy'}, colormap_labels(end:-1:1)],...  
    'TickLength',[0 0]);
xlabel(gca, 'Classification = Predictions');
ylabel(gca, 'Ground Truth = Reference Source');
plot([6.5 6.5],get(gca, 'XLim'), 'Color', [0, 0, 0],'LineWidth',2);
plot(get(gca, 'XLim'),[1.5 1.5], 'Color', [0, 0, 0],'LineWidth',2);
xtickangle(45)

if ~isfile('chess_train.png')
saveas(gcf,'chess_train.png');
end

%
label_vector = chess_label_test;
prediction_vector = chess_prediction_test;
method = 'Chess Test';

figure('Name',['Confusion Matrix - ', method])
hold on
title(['Confusion Matrix - ', method])
cm = confusionmat(label_vector,prediction_vector);

% Totals
total_r = sum(cm,2);
total_c = sum(cm)';
total_diag = sum(diag(cm));
total_cm = zeros(num_class + 1,num_class + 1);
total_cm(1:num_class,1:num_class) = cm;
total_cm(end,1:num_class) = total_r;
total_cm(1:num_class,end) = total_c;
total_cm(end,end) = total_diag;

prod_acc = diag(cm) ./ total_r * 100;
user_acc = diag(cm) ./ total_c * 100;

% Percentages
cm_percent = zeros(num_class + 1,num_class + 1);
cm_percent(1:num_class,1:num_class) = cm./repmat(sum(cm, 1),num_class,1) * 100;
cm_percent(end,1:num_class) = user_acc;
cm_percent(1:num_class,end) = prod_acc;
cm_percent(end,end) = total_diag/sum(total_c) * 100;


imagesc(cm_percent([(num_class + 1):-1:1], [1:(num_class + 1)]));
Mycolors=[1 1 1; 0.3 0.3 0.3];
colormap(Mycolors);
textStrings = num2str([cm_percent(:)], '%.1f%%\n');
textStrings_total = num2str([total_cm(:)], '%i\n');

textStrings = strtrim(cellstr(textStrings));
textStrings_total = strtrim(cellstr(textStrings_total));


[x,y] = meshgrid(1:(num_class + 1),num_class + 1:-1:1);
x = [x(:), x(:)];
y = [y(:)-0.2, y(:)+0.2];
hStrings = text(x(:),y(:),[textStrings(:),textStrings_total(:)], ...
    'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));
textColors = repmat(cm_percent(:) > midValue,1,3);
set(gca,'xaxisLocation','top')
set(hStrings,{'Color'},repmat(num2cell(textColors,2),2,1));
set(gca,'XTick',1:(num_class + 1),...
    'XTickLabel',[colormap_labels,{'Producer´s Accuracy'}],... 
    'YTick',1:(num_class + 1),... 
    'YTickLabel',[{'User´s Accuracy'}, colormap_labels(end:-1:1)],...  
    'TickLength',[0 0]);
xlabel(gca, 'Classification = Predictions');
ylabel(gca, 'Ground Truth = Reference Source');
plot([6.5 6.5],get(gca, 'XLim'), 'Color', [0, 0, 0],'LineWidth',2);
plot(get(gca, 'XLim'),[1.5 1.5], 'Color', [0, 0, 0],'LineWidth',2);
xtickangle(45)

if ~isfile('chess_test.png')
saveas(gcf,'chess_test.png');
end


%
label_vector = slic_label_training;
prediction_vector = slic_prediction_train;
method = 'SLIC Training';

figure('Name',['Confusion Matrix - ', method])
hold on
title(['Confusion Matrix - ', method])
cm = confusionmat(label_vector,prediction_vector);

% Totals
total_r = sum(cm,2);
total_c = sum(cm)';
total_diag = sum(diag(cm));
total_cm = zeros(num_class + 1,num_class + 1);
total_cm(1:num_class,1:num_class) = cm;
total_cm(end,1:num_class) = total_r;
total_cm(1:num_class,end) = total_c;
total_cm(end,end) = total_diag;

prod_acc = diag(cm) ./ total_r * 100;
user_acc = diag(cm) ./ total_c * 100;

% Percentages
cm_percent = zeros(num_class + 1,num_class + 1);
cm_percent(1:num_class,1:num_class) = cm./repmat(sum(cm, 1),num_class,1) * 100;
cm_percent(end,1:num_class) = user_acc;
cm_percent(1:num_class,end) = prod_acc;
cm_percent(end,end) = total_diag/sum(total_c) * 100;


imagesc(cm_percent([(num_class + 1):-1:1], [1:(num_class + 1)]));
Mycolors=[1 1 1; 0.3 0.3 0.3];
colormap(Mycolors);
textStrings = num2str([cm_percent(:)], '%.1f%%\n');
textStrings_total = num2str([total_cm(:)], '%i\n');

textStrings = strtrim(cellstr(textStrings));
textStrings_total = strtrim(cellstr(textStrings_total));


[x,y] = meshgrid(1:(num_class + 1),num_class + 1:-1:1);
x = [x(:), x(:)];
y = [y(:)-0.2, y(:)+0.2];
hStrings = text(x(:),y(:),[textStrings(:),textStrings_total(:)], ...
    'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));
textColors = repmat(cm_percent(:) > midValue,1,3);
set(gca,'xaxisLocation','top')
set(hStrings,{'Color'},repmat(num2cell(textColors,2),2,1));
set(gca,'XTick',1:(num_class + 1),...
    'XTickLabel',[colormap_labels,{'Producer´s Accuracy'}],... 
    'YTick',1:(num_class + 1),... 
    'YTickLabel',[{'User´s Accuracy'}, colormap_labels(end:-1:1)],...  
    'TickLength',[0 0]);
xlabel(gca, 'Classification = Predictions');
ylabel(gca, 'Ground Truth = Reference Source');
plot([6.5 6.5],get(gca, 'XLim'), 'Color', [0, 0, 0],'LineWidth',2);
plot(get(gca, 'XLim'),[1.5 1.5], 'Color', [0, 0, 0],'LineWidth',2);
xtickangle(45)

if ~isfile('slic_train.png')
saveas(gcf,'slic_train.png');
end


%
label_vector = slic_label_test;
prediction_vector = slic_prediction_test;
method = 'SLIC Test';

figure('Name',['Confusion Matrix - ', method])
hold on
title(['Confusion Matrix - ', method])
cm = confusionmat(label_vector,prediction_vector);

% Totals
total_r = sum(cm,2);
total_c = sum(cm)';
total_diag = sum(diag(cm));
total_cm = zeros(num_class + 1,num_class + 1);
total_cm(1:num_class,1:num_class) = cm;
total_cm(end,1:num_class) = total_r;
total_cm(1:num_class,end) = total_c;
total_cm(end,end) = total_diag;

prod_acc = diag(cm) ./ total_r * 100;
user_acc = diag(cm) ./ total_c * 100;

% Percentages
cm_percent = zeros(num_class + 1,num_class + 1);
cm_percent(1:num_class,1:num_class) = cm./repmat(sum(cm, 1),num_class,1) * 100;
cm_percent(end,1:num_class) = user_acc;
cm_percent(1:num_class,end) = prod_acc;
cm_percent(end,end) = total_diag/sum(total_c) * 100;


imagesc(cm_percent([(num_class + 1):-1:1], [1:(num_class + 1)]));
Mycolors=[1 1 1; 0.3 0.3 0.3];
colormap(Mycolors);
textStrings = num2str([cm_percent(:)], '%.1f%%\n');
textStrings_total = num2str([total_cm(:)], '%i\n');

textStrings = strtrim(cellstr(textStrings));
textStrings_total = strtrim(cellstr(textStrings_total));


[x,y] = meshgrid(1:(num_class + 1),num_class + 1:-1:1);
x = [x(:), x(:)];
y = [y(:)-0.2, y(:)+0.2];
hStrings = text(x(:),y(:),[textStrings(:),textStrings_total(:)], ...
    'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));
textColors = repmat(cm_percent(:) > midValue,1,3);
set(gca,'xaxisLocation','top')
set(hStrings,{'Color'},repmat(num2cell(textColors,2),2,1));
set(gca,'XTick',1:(num_class + 1),...
    'XTickLabel',[colormap_labels,{'Producer´s Accuracy'}],... 
    'YTick',1:(num_class + 1),... 
    'YTickLabel',[{'User´s Accuracy'}, colormap_labels(end:-1:1)],...  
    'TickLength',[0 0]);
xlabel(gca, 'Classification = Predictions');
ylabel(gca, 'Ground Truth = Reference Source');
plot([6.5 6.5],get(gca, 'XLim'), 'Color', [0, 0, 0],'LineWidth',2);
plot(get(gca, 'XLim'),[1.5 1.5], 'Color', [0, 0, 0],'LineWidth',2);
xtickangle(45)

if ~isfile('slic_test.png')
saveas(gcf,'slic_test.png');
end



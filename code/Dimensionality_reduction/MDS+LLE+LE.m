data=importdata('x.mat');
mappedX1=compute_mapping(data,'MDS',1);
mappedX2=compute_mapping(data,'LLE',1);
mappedX3=compute_mapping(data,'Laplacian',1);
save('MDS','mappedX1');
save('LLE','mappedX2');
save('Laplacian','mappedX3');
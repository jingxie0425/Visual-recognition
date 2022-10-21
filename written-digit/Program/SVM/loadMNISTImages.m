function images = loadMNISTImages(filename)
    fid = fopen(filename,'r'); %open the file as fid
    
    magic = fread(fid,1,'int32',0,'b'); %read the magic number
    assert(magic == 2051, ('Wrong data set!')); %check if the data set is the right one
    
    N_images = fread(fid,1,'int32',0,'b'); %number of images in the data set
    N_rows = fread(fid,1,'int32',0,'b'); %number of rows in one image
    N_cols = fread(fid,1,'int32',0,'b'); %number of columns in one image
    
    images = fread(fid,inf,'unsigned char'); %read the images
    images = reshape(images,N_cols*N_rows,N_images); %reshape every image into a column vector
    
    fclose(fid); %close the file
    
    images = double(images)/255; %normalize the image 
end
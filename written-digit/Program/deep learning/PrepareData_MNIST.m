function [opts] = PrepareData_MNIST(opts)
    imdb = GetMNISTData(); %read the data from MNIST dataset
    
    opts.train = imdb.images.data(:,:,:,imdb.images.set == 1); %training images
    opts.train_labels = imdb.images.labels(imdb.images.set == 1); %training labels

    opts.test = imdb.images.data(:,:,:,imdb.images.set == 3); %testing images
    opts.test_labels = imdb.images.labels(imdb.images.set == 3); %testing images
    
    opts.n_train=size(opts.train,4); %number of training images
    opts.n_test=size(opts.test,4); %number of testing images 
end
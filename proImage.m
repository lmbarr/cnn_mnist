function proI = proImage(image)
%     figure(1);
%     imshow(image);
    se = strel('rectangle', [5 1]);
    proI = imerode(imdilate(image, se), se);
    %figure(2);
    %imshow(proI);
    %imagesc(proI)
end
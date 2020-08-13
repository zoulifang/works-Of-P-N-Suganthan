function predicty = nnpredicty(nn, x)
    
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
  
    
    predicty = nn.a{end};
    
end

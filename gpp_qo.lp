\ Model gpp
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x5b4ba77b9b4a9ee3
Minimize
 [ 2 x(1) ^2 - 4 x(1) * x(2) + 6 x(2) ^2 - 4 x(2) * x(3) - 4 x(2) * x(4)
   + 4 x(3) ^2 - 4 x(3) * x(4) + 4 x(4) ^2 ] / 2 
Subject To
 Partition: x(1) + x(2) + x(3) + x(4) = 2
Bounds
Binaries
 x(1) x(2) x(3) x(4)
End

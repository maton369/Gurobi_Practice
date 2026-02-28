\ Model gpp model -- soco
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xbaa9a51ce32c8310
Minimize
  z(1,2) + z(2,3) + z(2,4) + z(3,4)
Subject To
 Partition: x(1) + x(2) + x(3) + x(4) = 2
 P(1,2): s(1,2) + z(1,2) = 1
 P(2,3): s(2,3) + z(2,3) = 1
 P(2,4): s(2,4) + z(2,4) = 1
 P(3,4): s(3,4) + z(3,4) = 1
 S(1,2): - 2 x(1) - 2 x(2) - s(1,2) + [ x(1) ^2 + 2 x(1) * x(2) + x(2) ^2 ]
   <= -1
 Z(1,2): - z(1,2) + [ x(1) ^2 - 2 x(1) * x(2) + x(2) ^2 ] <= 0
 S(2,3): - 2 x(2) - 2 x(3) - s(2,3) + [ x(2) ^2 + 2 x(2) * x(3) + x(3) ^2 ]
   <= -1
 Z(2,3): - z(2,3) + [ x(2) ^2 - 2 x(2) * x(3) + x(3) ^2 ] <= 0
 S(2,4): - 2 x(2) - 2 x(4) - s(2,4) + [ x(2) ^2 + 2 x(2) * x(4) + x(4) ^2 ]
   <= -1
 Z(2,4): - z(2,4) + [ x(2) ^2 - 2 x(2) * x(4) + x(4) ^2 ] <= 0
 S(3,4): - 2 x(3) - 2 x(4) - s(3,4) + [ x(3) ^2 + 2 x(3) * x(4) + x(4) ^2 ]
   <= -1
 Z(3,4): - z(3,4) + [ x(3) ^2 - 2 x(3) * x(4) + x(4) ^2 ] <= 0
Bounds
Binaries
 x(1) x(2) x(3) x(4)
End

# jigsolving

Rectangular jigsaw reconstruction program written for bachelor's thesis. 

Rearranges rectangular pieces on a white background, and is made to be robust to noise and distortion. Requires a reference image (like a photo from the back of a box).

Uses GPU-accelerated cross correlation computation to calculate similarities, finds the correct puzzle dimensions, and rearranges the pieces using discrete genetic algorithm (permutation problem).
Genetic algorithm first fixes all obvious matches, then rearranges the ambiguous pieces.

At some point used neural networks for piece detection, but that did not work as well as an algorithmic approach. Code still remains.

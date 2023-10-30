# AUGER-POP

If you use or are inspired by this code please cite:
1. Resonant Double-Core Excitations with Ultrafast, Intense Pulses\
A. E. A. Fouda, D. Koulentianos, L. Young, G. Doumy and P. J. Ho\
Submitted, arxiv pre-print available at https://doi.org/10.48550/arXiv.2208.05307

This code uses the CI vectors corresponding to the rasscf natural orbitals between the core-hole and final bound state cations, care msut be taken to check that the there is the same ordering. The code will be soon reimplemented to read from the biorthonormal orbitals in the RASSI output.

This code might require a local modification of the GRID_IT module in openmolcas that prints the Mulliken population per MO. The subroutine is already made you just have to call it.

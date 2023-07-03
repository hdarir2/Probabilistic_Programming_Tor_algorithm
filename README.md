Readme
======

This script should be run in conjuction with the Shadow simulation as well  the sbws scanner. The script pauses the shadow/client traffic after a certain amount of time specified in the Config file of the scanner. The script will then run the Probabilsitic program [1] to estimate the bandwidth of the node in the network based on the measurement of the scanner. The script will then write and publish those estimates to be read by the Tor authorities and be used by future clients.


## Authors

Hussein Darir,
PhD in mechanical engineering,
University of Illinois Urbana-Champaign.

## References


[1]: Darir, H., Borisov, N., & Dullerud, G. (2023). Probflow : Using probabilistic programming in anonymous communication networks.
In Network and distributed system security (ndss) symposium. 􀃮 doi:doi:10.14722/ndss.2023.24140

# Self-Drive-Vehicles-for-GTA5
This project aims at creating self driven cars in GTA5. It captures frames using a screen grab script rather than working within the game’s code.

It uses a Convolutional Neural Network to train the model. The model basically learns (well, attempts to learn) whatever it is fed through the screen grab script and then using that it reverses the process by giving instructions to the vehicle. 

![Lane-processing](https://raw.githubusercontent.com/akarshroot/Self-Drive-Vehicles-for-GTA5/main/lanesuccess%20-%20Copy.png)
Above image shows how the model processes the data being fed to it and reduces it as per requirement, that is, it keeps only the lanes and obstacles and ignores everything else.

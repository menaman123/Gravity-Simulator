PROJECT DESCRIPTION

Our team would like to develop a gravity simulator which can be used to estimate the position of small and large planetary bodies within a 2d solar system. Datasets will be artificially synthesized. Since small bodies like asteroids do not normally affect the position of larger bodies, it will be assumed that small bodies can only affect the position of small bodies, but large bodies can affect the position of both large and small bodies. The simulator will be able to estimate the position of 1000 to 1,000,000 bodies at once and will take into account the mass, radius, x and y coordinates, position, speed, and acceleration of bodies. If time permits, collisions between bodies will be taken into account. The output of the simulator will be the position of all bodies’ x and y coordinates within a text file. Multiple text files will be read into matlab and plotted in chronological sequence in a cartesian plane like a stop motion movie.The
gravity simulator will be produced using OpenMP or AVX2.

Team member individual tasks:
Logan Pasternak:
*Figure out how to synthesize the datasets for the gravity simulator experiments (5 pts)
*Store the resulting data for a position frame within a text file (2 pts)
*Migrate the text file data over to Matlab (2 pts)
Antonio Mena:
*Implement the gravity simulator in a single threaded context (no parallelism) (4 pts)
*Create a simple test case (Earth, Mars, Sun, a few asteroids) (2 pts)
*Create an extreme test case (Lots of planets and lots of asteroids!) (3 pts)
Felix Shames:
*Implement the gravity simulator in a parallel context (OpenMP/AVX2) (5 pts)
*Plot the position of the bodies from the text file into an cartesian plane (the radius of
points should be somewhat analogous to the radius of the individual body) (1 pts)
*Allow switching between multiple plots for stop motion effect (2 pts)
*Create a class for both large terrestrial bodies (planets/stars) and small ones
(asteroids) (1 pts)
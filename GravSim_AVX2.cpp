#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <time.h>
#include <immintrin.h>
#include <memory.h>
#include <stdlib.h>
#include "GravSim_AVX2.hpp"


__m256d load(const double* ap) {
    // Allocate 32-byte aligned memory for a double
    double* alignedMemory = (double*) aligned_alloc(32, sizeof(double));

    // Copy the double data into the allocated memory
    memcpy(alignedMemory, ap, sizeof(double));

    // Load the data into an AVX2 register
    __m256d v = _mm256_load_pd(alignedMemory);

    // Free the allocated memory
    free(alignedMemory);

    return v;
}

void updateVelocity(largeBody& obj, double dt, largeBody* objects, int num_objects) {

    obj.vx = 0.0;
    obj.vy = 0.0;

    for (int i = 0; i < num_objects; ++i) {
        largeBody* temp= &objects[i];
        if (&obj != temp) { 
            double dx = (*temp).x - obj.x;
            double dy = (*temp).y - obj.y;
            double r = std::sqrt(dx * dx + dy * dy);

            // calculate acceleration due to gravity
            double ax = G * (*temp).mass * dx / (r * r * r);
            double ay = G * (*temp).mass * dy / (r * r * r);

            // update velocity
            obj.vx += ax * dt;
            obj.vy += ay * dt;
        }
    }
}

void updateVelocityAVX2(largeBody& obj, double dt, largeBody* objects, int num_objects) {

/*
    objects[0] = {0.0, 0.0, 0.0, 0.0, 1.989e30};
    objects[1] = {147e9, 0.0, 0.0, 29783.0, 5.972e24}; 
    objects[2] = {228e9, 0.0, 0.0, 24077.0, 6.4171e23};
    objects[3] = {108e9, 0.0, 0.0, 35000.0, 4.8675e24};
    objects[4] = {778e9, 0.0, 0.0, 13070.0, 1.898e27};
    objects[5] = {57.9e9, 0.0, 0.0, 47400.0, 3.301e23};
    objects[6] = {1.429e12, 0.0, 0.0, 9660.0, 5.683e26}; 
    objects[7] = {2.871e12, 0.0, 0.0, 6810.0, 8.681e25};
*/

    double* aligned_x_coordinates = (double*)aligned_alloc(32, num_objects * sizeof(double));
    double* aligned_y_coordinates = (double*)aligned_alloc(32, num_objects * sizeof(double));
    double* aligned_mass = (double*)aligned_alloc(32, num_objects * sizeof(double));

    // Copy x coordinates to aligned memory
    for (int i = 0; i < num_objects; ++i) {
        aligned_x_coordinates[i] = objects[i].x;
    }

    // Copy x coordinates to aligned memory
    for (int i = 0; i < num_objects; ++i) {
        aligned_y_coordinates[i] = objects[i].y;
    }

    // Copy x coordinates to aligned memory
    for (int i = 0; i < num_objects; ++i) {
        aligned_mass[i] = objects[i].mass;
    }

    const __m256d zero_vec = _mm256_setzero_pd();

    __m256d vx_acc = zero_vec;
    __m256d vy_acc = zero_vec;

    /*

    double vx_test[4], vy_test[4];
    _mm256_storeu_pd(vx_test, vx_acc);
    _mm256_storeu_pd(vy_test, vy_acc);
    std::cout<<vx_test;
    std::cout<<vy_test;
    
    */

    //Print out contents of objects.x;

    for (int i = 0; i < num_objects; i += 4) 
    {
        // Load data from objects array using AVX2
        __m256d x_vec = _mm256_loadu_pd(&aligned_x_coordinates[i]);
        __m256d y_vec = _mm256_loadu_pd(&aligned_y_coordinates[i]);
        __m256d mass_vec = _mm256_loadu_pd(&aligned_mass[i]);

        //Test initial inputs
    
        /*
        double x_test[4], y_test[4], mass_test[4];
        _mm256_storeu_pd(x_test, x_vec);
        _mm256_storeu_pd(y_test, y_vec);
        _mm256_storeu_pd(mass_test, mass_vec);
        std::cout << "xTest: " << x_test[0] << " " << x_test[1] << " " << x_test[2] << " " << x_test[3] << std::endl;
        std::cout << "yTest: " << y_test[0] << " " << y_test[1] << " " << y_test[2] << " " << y_test[3] << std::endl;
        std::cout << "massTest: " << mass_test[0] << " " << mass_test[1] << " " << mass_test[2] << " " << mass_test[3] << std::endl;
        std::cout << std::endl;
        */

        // Calculate differences
        __m256d dx_vec = _mm256_sub_pd(x_vec, _mm256_set1_pd(obj.x));
        __m256d dy_vec = _mm256_sub_pd(y_vec, _mm256_set1_pd(obj.y));

        //Difference calcualtion tests

        /*

        double dx_test[4], dy_test[4];
        _mm256_storeu_pd(dx_test, dx_vec);
        _mm256_storeu_pd(dy_test, dy_vec);
        std::cout << "dxTest: " << dx_test[0] << " " << dx_test[1] << " " << dx_test[2] << " " << dx_test[3] << std::endl;
        std::cout << "dyTest: " << dy_test[0] << " " << dy_test[1] << " " << dy_test[2] << " " << dy_test[3] << std::endl;
        std::cout << std::endl;
        
        */
        
        
        // Calculate squared distance
        __m256d r_squared_vec = _mm256_add_pd(_mm256_mul_pd(dx_vec, dx_vec), _mm256_mul_pd(dy_vec, dy_vec));
        
        // Approximate reciprocal of the square root
        __m256d r_inv_sqrt_vec = _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(_mm256_max_pd(r_squared_vec, _mm256_set1_pd(1e-12))));

        // Calculate reciprocal of the cube root
        __m256d r_inv_cube_vec = _mm256_mul_pd(r_inv_sqrt_vec, _mm256_mul_pd(r_inv_sqrt_vec, r_inv_sqrt_vec));

        //Test square operations
        /*

        double square_test[4], sqrt_test[4], cube_test[4];
        _mm256_storeu_pd(square_test, r_squared_vec);
        _mm256_storeu_pd(sqrt_test, r_inv_sqrt_vec);
        _mm256_storeu_pd(cube_test, r_inv_cube_vec);
        std::cout << "square_Test: " << square_test[0] << " " << square_test[1] << " " << square_test[2] << " " << square_test[3] << std::endl;
        std::cout << "sqrt_Test: " << sqrt_test[0] << " " << sqrt_test[1] << " " << sqrt_test[2] << " " << sqrt_test[3] << std::endl;
        std::cout << "cube_Test: " << cube_test[0] << " " << cube_test[1] << " " << cube_test[2] << " " << cube_test[3] << std::endl;
        std::cout << std::endl;
        
        */

        // Calculate acceleration due to gravity
        __m256d G_mass_vec = _mm256_mul_pd(_mm256_set1_pd(G), mass_vec);
        __m256d ax_vec = _mm256_mul_pd(G_mass_vec, _mm256_mul_pd(dx_vec, r_inv_cube_vec));
        __m256d ay_vec = _mm256_mul_pd(G_mass_vec, _mm256_mul_pd(dy_vec, r_inv_cube_vec));

        //Test gravity mass vector and acceleration vectors
        /*

        double G_test[4], ax_test[4], ay_test[4];
        _mm256_storeu_pd(G_test,G_mass_vec);
        _mm256_storeu_pd(ax_test, ax_vec);
        _mm256_storeu_pd(ay_test, ay_vec);
        std::cout << "G_Test: " << G_test[0] << " " << G_test[1] << " " << G_test[2] << " " << G_test[3] << std::endl;
        std::cout << "ax_Test: " << ax_test[0] << " " << ax_test[1] << " " << ax_test[2] << " " << ax_test[3] << std::endl;
        std::cout << "ay_Test: " << ay_test[0] << " " << ay_test[1] << " " << ay_test[2] << " " << ay_test[3] << std::endl;
        std::cout << std::endl;

        */

        // Update velocity
        vx_acc = _mm256_add_pd(vx_acc, _mm256_mul_pd(ax_vec, _mm256_set1_pd(dt)));
        vy_acc = _mm256_add_pd(vy_acc, _mm256_mul_pd(ay_vec, _mm256_set1_pd(dt)));

    }

    // Free the allocated memory outside the loop
    free(aligned_x_coordinates);
    free(aligned_y_coordinates);
    free(aligned_mass);

    // Horizontal sum of the accumulated velocities
    double vx_sum[4], vy_sum[4];
    _mm256_storeu_pd(vx_sum, vx_acc);
    _mm256_storeu_pd(vy_sum, vy_acc);

    // Update the object's velocity
    obj.vx += vx_sum[0] + vx_sum[1] + vx_sum[2] + vx_sum[3];
    obj.vy += vy_sum[0] + vy_sum[1] + vy_sum[2] + vy_sum[3];
}


void updatePosition(largeBody& obj, double dt) {
    obj.x += obj.vx * dt;
    obj.y += obj.vy * dt;
}

void updatePositionAVX2(largeBody& obj, double dt) {
    // Load velocity components into AVX2 registers
    __m256d vx_vec = _mm256_set1_pd(obj.vx * dt);
    __m256d vy_vec = _mm256_set1_pd(obj.vy * dt);

    // Load position components into AVX2 registers
    __m256d x_vec = _mm256_set1_pd(obj.x);
    __m256d y_vec = _mm256_set1_pd(obj.y);

    // Update position using AVX2
    x_vec = _mm256_add_pd(x_vec, vx_vec);
    y_vec = _mm256_add_pd(y_vec, vy_vec);

    // Store updated position back to the object
    _mm256_storeu_pd(&obj.x, x_vec);
    _mm256_storeu_pd(&obj.y, y_vec);
}

void simulator(int num_objects)
{
    // dynamically allocate an array of objects
    largeBody* objects = new largeBody[num_objects];

    double maximumMass = 2.0e30;
    double minimumMass = 1.0;
    double maximumDist = 6.0e9 ;
    double minimumDist = 1.0;

    // initialize objects
    for (int i=0; i < num_objects; i++)
    {
        if (i==0)
        {
            objects[0] = {0.0, 0.0, 0.0, 0.0, 1.989e30}; // Sun 
        }
        else if(i==1)
        {
            objects[1] = {1.496e8, 0.0, 0.0, 29783.0, 5.972e24}; // Earth
        }
        else
        {
            objects[i].x=(rand()/(double)RAND_MAX)*(maximumDist-minimumDist)+minimumDist;
            objects[i].y=(rand()/(double)RAND_MAX)*(maximumDist-minimumDist)+minimumDist;
            objects[i].vx=0;
            objects[i].vy=0;
            objects[i].mass=(rand()/(double)RAND_MAX)*(maximumMass-minimumMass)+minimumMass;
        }

    }

    // simulation parameters
    double dt = 1000.0; // time step in seconds
    int num_steps = 3; // number of time steps

    std::cout << "Do you want to make a csv file containing the position data? (y/n)"<<std::endl;

    std::string input;

    std::cin >> input;

    std::cout<<"Do you want to execute the code using AVX2? (y/n)"<<std::endl;

    std::string inputB;

    std::cin >> inputB;

    if (input=="y" && inputB=="n")
    {
        //Clock starts
        clock_t t0= clock();

        std::ofstream myFile;
        myFile.open("test.csv");

        // simulation loop
        for (int step = 0; step < num_steps; ++step) {
            // update velocity and position for each object
            
            if (step==0)
            {
                // output positions of all objects at each step
                myFile << "StepNumber,Body,xPosition,yPosition,\n";
                for (int i = 0; i < num_objects; ++i) 
                {
                    std::string tempStep = std::to_string(step+1);
                    std::string bodyNum = std::to_string(i);
                    std::string positionX = std::to_string(objects[i].x);
                    std::string positionY = std::to_string(objects[i].y);

                    std::string expression = tempStep + "," + bodyNum + ","+ positionX + "," + positionY +  "," + "\n";


                    myFile << expression;
                    //std::cout << "Step " << step + 1 << ": Object" << i << " position (x, y) = (" << objects[i].x << ", " << objects[i].y << ")\n";
                }
            }
            else 
            {
                for (int i = 0; i < num_objects; ++i) 
                {
                    updateVelocity(objects[i], dt, objects, num_objects);
                    updatePosition(objects[i], dt);
                }

                // output positions of all objects at each step
                for (int i = 0; i < num_objects; ++i) 
                {
                    std::string tempStep = std::to_string(step+1);
                    std::string bodyNum = std::to_string(i);
                    std::string positionX = std::to_string(objects[i].x);
                    std::string positionY = std::to_string(objects[i].y);

                    std::string expression = tempStep + "," + bodyNum + ","+ positionX + "," + positionY +  "," + "\n";
                    myFile << expression;
                    //std::cout << "Step " << step + 1 << ": Object" << i << " position (x, y) = (" << objects[i].x << ", " << objects[i].y << ")\n";
                }
            }

        }

        myFile.close();

        //Timer stops
        clock_t t1= clock();

        //Duration calculated and time displayed
        std::cout<<"Duration: "<<(t1-t0)/(double)CLOCKS_PER_SEC <<"s"<< std::endl;

    }
    else if (input=="n" && inputB=="n")
    {

        //Clock starts
        clock_t t0= clock();

        // simulation loop
        for (int step = 0; step < num_steps; ++step) {
            // update velocity and position for each object
            
            if (step==0)
            {
                // output positions of all objects at each step
                std::cout << "StepNumber,Body,xPosition,yPosition,\n";
                for (int i = 0; i < num_objects; ++i) 
                {
                    std::string tempStep = std::to_string(step+1);
                    std::string bodyNum = std::to_string(i);
                    std::string positionX = std::to_string(objects[i].x);
                    std::string positionY = std::to_string(objects[i].y);

                    std::string expression = tempStep + "," + bodyNum + ","+ positionX + "," + positionY +  "," + "\n";


                    std::cout << expression;
                    //std::cout << "Step " << step + 1 << ": Object" << i << " position (x, y) = (" << objects[i].x << ", " << objects[i].y << ")\n";
                }
            }
            else 
            {
                for (int i = 0; i < num_objects; ++i) 
                {
                    updateVelocity(objects[i], dt, objects, num_objects);
                    updatePosition(objects[i], dt);
                }

                // output positions of all objects at each step
                for (int i = 0; i < num_objects; ++i) 
                {
                    std::string tempStep = std::to_string(step+1);
                    std::string bodyNum = std::to_string(i);
                    std::string positionX = std::to_string(objects[i].x);
                    std::string positionY = std::to_string(objects[i].y);

                    std::string expression = tempStep + "," + bodyNum + ","+ positionX + "," + positionY +  "," + "\n";
                    std::cout << expression;
                    //std::cout << "Step " << step + 1 << ": Object" << i << " position (x, y) = (" << objects[i].x << ", " << objects[i].y << ")\n";
                }
            }

        }

        //Timer stops
        clock_t t1= clock();

        //Duration calculated and time displayed
        std::cout<<"Duration: "<<(t1-t0)/(double)CLOCKS_PER_SEC <<"s"<< std::endl;

    }
    else if (input=="y" && inputB=="y")
    {

        //Clock starts
        clock_t t0= clock();

        std::ofstream myFile;
        myFile.open("test.csv");

        // simulation loop
        for (int step = 0; step < num_steps; ++step) {
            // update velocity and position for each object
            
            if (step==0)
            {
                // output positions of all objects at each step
                myFile << "StepNumber,Body,xPosition,yPosition,\n";
                for (int i = 0; i < num_objects; ++i) 
                {
                    std::string tempStep = std::to_string(step+1);
                    std::string bodyNum = std::to_string(i);
                    std::string positionX = std::to_string(objects[i].x);
                    std::string positionY = std::to_string(objects[i].y);

                    std::string expression = tempStep + "," + bodyNum + ","+ positionX + "," + positionY +  "," + "\n";


                    myFile << expression;
                    //std::cout << "Step " << step + 1 << ": Object" << i << " position (x, y) = (" << objects[i].x << ", " << objects[i].y << ")\n";
                }
            }
            else 
            {
                for (int i = 0; i < num_objects; ++i) 
                {
                    updateVelocityAVX2(objects[i], dt, objects, num_objects);
                    updatePosition(objects[i], dt);
                }

                // output positions of all objects at each step
                for (int i = 0; i < num_objects; ++i) 
                {
                    std::string tempStep = std::to_string(step+1);
                    std::string bodyNum = std::to_string(i);
                    std::string positionX = std::to_string(objects[i].x);
                    std::string positionY = std::to_string(objects[i].y);

                    std::string expression = tempStep + "," + bodyNum + ","+ positionX + "," + positionY +  "," + "\n";
                    myFile << expression;
                    //std::cout << "Step " << step + 1 << ": Object" << i << " position (x, y) = (" << objects[i].x << ", " << objects[i].y << ")\n";
                }
            }

        }

        myFile.close();

        //Timer stops
        clock_t t1= clock();

        //Duration calculated and time displayed
        std::cout<<"Duration: "<<(t1-t0)/(double)CLOCKS_PER_SEC <<"s"<< std::endl;

    }
    else if (input=="n" && inputB=="y")
    {

        //Clock starts
        clock_t t0= clock();

        // simulation loop
        for (int step = 0; step < num_steps; ++step) {
            // update velocity and position for each object
            
            if (step==0)
            {
                // output positions of all objects at each step
                std::cout << "StepNumber,Body,xPosition,yPosition,\n";
                for (int i = 0; i < num_objects; ++i) 
                {
                    std::string tempStep = std::to_string(step+1);
                    std::string bodyNum = std::to_string(i);
                    std::string positionX = std::to_string(objects[i].x);
                    std::string positionY = std::to_string(objects[i].y);

                    std::string expression = tempStep + "," + bodyNum + ","+ positionX + "," + positionY +  "," + "\n";


                    std::cout << expression;
                    //std::cout << "Step " << step + 1 << ": Object" << i << " position (x, y) = (" << objects[i].x << ", " << objects[i].y << ")\n";
                }
            }
            else 
            {
                for (int i = 0; i < num_objects; ++i) 
                {
                    updateVelocityAVX2(objects[i], dt, objects, num_objects);
                    updatePosition(objects[i], dt);
                }

                // output positions of all objects at each step
                for (int i = 0; i < num_objects; ++i) 
                {
                    std::string tempStep = std::to_string(step+1);
                    std::string bodyNum = std::to_string(i);
                    std::string positionX = std::to_string(objects[i].x);
                    std::string positionY = std::to_string(objects[i].y);

                    std::string expression = tempStep + "," + bodyNum + ","+ positionX + "," + positionY +  "," + "\n";
                    std::cout << expression;
                    //std::cout << "Step " << step + 1 << ": Object" << i << " position (x, y) = (" << objects[i].x << ", " << objects[i].y << ")\n";
                }
            }

        }

        //Timer stops
        clock_t t1= clock();

        //Duration calculated and time displayed
        std::cout<<"Duration: "<<(t1-t0)/(double)CLOCKS_PER_SEC <<"s"<< std::endl;

    }
    else
    {
        std::cout<<"HEY BUDDY THAT WAS A MISS INPUT!"<<std::endl;
    }

    
    // free the dynamically allocated memory
    delete[] objects;
}

void miniSimulator()
{
    const int num_objects = 8; // specify the number of objects

    // dynamically allocate an array of objects
    largeBody* objects = new largeBody[num_objects];

    // Sun
    objects[0] = {0.0, 0.0, 0.0, 0.0, 1.989e30}; // Mass of the Sun in kg

    // Earth (average distance: 147 million km)
    objects[1] = {147e9, 0.0, 0.0, 29783.0, 5.972e24}; // Mass of the Earth in kg

    // Mars (average distance: 228 million km)
    objects[2] = {228e9, 0.0, 0.0, 24077.0, 6.4171e23}; // Mass of Mars in kg

    // Venus (average distance: 108 million km)
    objects[3] = {108e9, 0.0, 0.0, 35000.0, 4.8675e24}; // Mass of Venus in kg

    // Jupiter (average distance: 778 million km)
    objects[4] = {778e9, 0.0, 0.0, 13070.0, 1.898e27}; // Mass of Jupiter in kg

    // Mercury (average distance: 57.9 million km)
    objects[5] = {57.9e9, 0.0, 0.0, 47400.0, 3.301e23}; // Mass of Mercury in kg

    // Saturn (average distance: 1.429 billion km)
    objects[6] = {1.429e12, 0.0, 0.0, 9660.0, 5.683e26}; // Mass of Saturn in kg

    // Uranus (average distance: 2.871 billion km)
    objects[7] = {2.871e12, 0.0, 0.0, 6810.0, 8.681e25}; // Mass of Uranus in kg

    // simulation parameters
    double dt = 1000.0; // time step in seconds
    int num_steps = 3; // number of time steps

    std::cout << "Do you want to make a csv file containing the position data? (y/n)"<<std::endl;

    std::string input;

    std::cin >> input;

    std::cout<<"Do you want to execute the code using AVX2? (y/n)"<<std::endl;

    std::string inputB;

    std::cin >> inputB;

    if (input=="y" && inputB=="n")
    {

        //Clock starts
        clock_t t0= clock();

        std::ofstream myFile;
        myFile.open("test.csv");

        // simulation loop
        for (int step = 0; step < num_steps; ++step) {
            // update velocity and position for each object
            
            if (step==0)
            {
                // output positions of all objects at each step
                myFile << "StepNumber,Body,xPosition,yPosition,\n";
                for (int i = 0; i < num_objects; ++i) 
                {
                    std::string tempStep = std::to_string(step+1);
                    std::string bodyNum = std::to_string(i);
                    std::string positionX = std::to_string(objects[i].x);
                    std::string positionY = std::to_string(objects[i].y);

                    std::string expression = tempStep + "," + bodyNum + ","+ positionX + "," + positionY +  "," + "\n";


                    myFile << expression;
                    //std::cout << "Step " << step + 1 << ": Object" << i << " position (x, y) = (" << objects[i].x << ", " << objects[i].y << ")\n";
                }
            }
            else 
            {
                for (int i = 0; i < num_objects; ++i) 
                {
                    updateVelocity(objects[i], dt, objects, num_objects);
                    updatePosition(objects[i], dt);
                }

                // output positions of all objects at each step
                for (int i = 0; i < num_objects; ++i) 
                {
                    std::string tempStep = std::to_string(step+1);
                    std::string bodyNum = std::to_string(i);
                    std::string positionX = std::to_string(objects[i].x);
                    std::string positionY = std::to_string(objects[i].y);

                    std::string expression = tempStep + "," + bodyNum + ","+ positionX + "," + positionY +  "," + "\n";
                    myFile << expression;
                    //std::cout << "Step " << step + 1 << ": Object" << i << " position (x, y) = (" << objects[i].x << ", " << objects[i].y << ")\n";
                }
            }
            

        }

        myFile.close();

        //Timer stops
        clock_t t1= clock();

        //Duration calculated and time displayed
        std::cout<<"Duration: "<<(t1-t0)/(double)CLOCKS_PER_SEC <<"s"<< std::endl;

    }
    else if (input=="n" && inputB =="n")
    {

        //Clock starts
        clock_t t0= clock();

        // simulation loop
        for (int step = 0; step < num_steps; ++step) {
            // update velocity and position for each object
            
            if (step==0)
            {
                // output positions of all objects at each step
                std::cout << "StepNumber,Body,xPosition,yPosition,\n";
                for (int i = 0; i < num_objects; ++i) 
                {
                    std::string tempStep = std::to_string(step+1);
                    std::string bodyNum = std::to_string(i);
                    std::string positionX = std::to_string(objects[i].x);
                    std::string positionY = std::to_string(objects[i].y);

                    std::string expression = tempStep + "," + bodyNum + ","+ positionX + "," + positionY +  "," + "\n";


                    std::cout << expression;
                    //std::cout << "Step " << step + 1 << ": Object" << i << " position (x, y) = (" << objects[i].x << ", " << objects[i].y << ")\n";
                }
            }
            else 
            {
                for (int i = 0; i < num_objects; ++i) 
                {
                    updateVelocity(objects[i], dt, objects, num_objects);
                    updatePosition(objects[i], dt);
                }

                // output positions of all objects at each step
                for (int i = 0; i < num_objects; ++i) 
                {
                    std::string tempStep = std::to_string(step+1);
                    std::string bodyNum = std::to_string(i);
                    std::string positionX = std::to_string(objects[i].x);
                    std::string positionY = std::to_string(objects[i].y);

                    std::string expression = tempStep + "," + bodyNum + ","+ positionX + "," + positionY +  "," + "\n";
                    std::cout << expression;
                    //std::cout << "Step " << step + 1 << ": Object" << i << " position (x, y) = (" << objects[i].x << ", " << objects[i].y << ")\n";
                }
            }

        }

        //Timer stops
        clock_t t1= clock();

        //Duration calculated and time displayed
        std::cout<<"Duration: "<<(t1-t0)/(double)CLOCKS_PER_SEC <<"s"<< std::endl;
        

    }
    else if (input=="y" && inputB=="y")
    {

        //Clock starts
        clock_t t0= clock();

        std::ofstream myFile;
        myFile.open("test.csv");

        // simulation loop
        for (int step = 0; step < num_steps; ++step) {
            // update velocity and position for each object
            
            if (step==0)
            {
                // output positions of all objects at each step
                myFile << "StepNumber,Body,xPosition,yPosition,\n";
                for (int i = 0; i < num_objects; ++i) 
                {
                    std::string tempStep = std::to_string(step+1);
                    std::string bodyNum = std::to_string(i);
                    std::string positionX = std::to_string(objects[i].x);
                    std::string positionY = std::to_string(objects[i].y);

                    std::string expression = tempStep + "," + bodyNum + ","+ positionX + "," + positionY +  "," + "\n";


                    myFile << expression;
                    //std::cout << "Step " << step + 1 << ": Object" << i << " position (x, y) = (" << objects[i].x << ", " << objects[i].y << ")\n";
                }
            }
            else 
            {
                for (int i = 0; i < num_objects; ++i) 
                {
                    updateVelocityAVX2(objects[i], dt, objects, num_objects);
                    updatePosition(objects[i], dt);
                }

                // output positions of all objects at each step
                for (int i = 0; i < num_objects; ++i) 
                {
                    std::string tempStep = std::to_string(step+1);
                    std::string bodyNum = std::to_string(i);
                    std::string positionX = std::to_string(objects[i].x);
                    std::string positionY = std::to_string(objects[i].y);

                    std::string expression = tempStep + "," + bodyNum + ","+ positionX + "," + positionY +  "," + "\n";
                    myFile << expression;
                    //std::cout << "Step " << step + 1 << ": Object" << i << " position (x, y) = (" << objects[i].x << ", " << objects[i].y << ")\n";
                }
            }
            

        }

        myFile.close();

        //Timer stops
        clock_t t1= clock();

        //Duration calculated and time displayed
        std::cout<<"Duration: "<<(t1-t0)/(double)CLOCKS_PER_SEC <<"s"<< std::endl;

    }
    else if (input=="n" && inputB=="y")
    {

        //Clock starts
        clock_t t0= clock();

        // simulation loop
        for (int step = 0; step < num_steps; ++step) {
            // update velocity and position for each object
            
            if (step==0)
            {
                // output positions of all objects at each step
                std::cout << "StepNumber,Body,xPosition,yPosition,\n";
                for (int i = 0; i < num_objects; ++i) 
                {
                    std::string tempStep = std::to_string(step+1);
                    std::string bodyNum = std::to_string(i);
                    std::string positionX = std::to_string(objects[i].x);
                    std::string positionY = std::to_string(objects[i].y);

                    std::string expression = tempStep + "," + bodyNum + ","+ positionX + "," + positionY +  "," + "\n";


                    std::cout << expression;
                    //std::cout << "Step " << step + 1 << ": Object" << i << " position (x, y) = (" << objects[i].x << ", " << objects[i].y << ")\n";
                }
            }
            else 
            {
                for (int i = 0; i < num_objects; ++i) 
                {
                    updateVelocityAVX2(objects[i], dt, objects, num_objects);
                    updatePosition(objects[i], dt);
                }

                // output positions of all objects at each step
                for (int i = 0; i < num_objects; ++i) 
                {
                    std::string tempStep = std::to_string(step+1);
                    std::string bodyNum = std::to_string(i);
                    std::string positionX = std::to_string(objects[i].x);
                    std::string positionY = std::to_string(objects[i].y);

                    std::string expression = tempStep + "," + bodyNum + ","+ positionX + "," + positionY +  "," + "\n";
                    std::cout << expression;
                    //std::cout << "Step " << step + 1 << ": Object" << i << " position (x, y) = (" << objects[i].x << ", " << objects[i].y << ")\n";
                }
            }

        }

        //Timer stops
        clock_t t1= clock();

        //Duration calculated and time displayed
        std::cout<<"Duration: "<<(t1-t0)/(double)CLOCKS_PER_SEC <<"s"<< std::endl;

    }
    else
    {
        std::cout<<"HEY BUDDY THAT WAS A MISS INPUT!"<<std::endl;
    }

    
    // free the dynamically allocated memory
    delete[] objects;
}
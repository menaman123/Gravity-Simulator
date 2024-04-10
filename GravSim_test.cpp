#include "GravSim.h"



#define _USE_MATH_DEFINES
 
#include <cmath>
#include <iostream>


SpaceObject* earth = new SpaceObject(0.0, 0.0, 1000000);

SpaceObject* moon = new SpaceObject(0.0, 10000, 100000000000000000000000.0);

std::vector<SpaceObject*> space_objects{earth, moon};

double time_seconds = 10;

void angleTest(){
    double real_angle = 90.0;
    double test_angle = calculateAngle(*earth, *moon) * 180.0 / M_PI;
    if(real_angle == test_angle){
        std::cout<< "Angle test correct \n";
    }
    else {
        std::cout<< "Angle test incorrect \n";
    }
    
}

void positionTest(){

    std::cout<< "Position for test1: " << earth->get_x() << ", " << earth->get_y() << "\n";
    std::cout<< "Position for test2: " << moon->get_x() << ", " << moon->get_y() << "\n";

    calculateAcceleration(space_objects, time_seconds);
    
    std::cout<< "Position for test1: " << earth->get_x() << ", " << earth->get_y() << "\n";
    std::cout<< "Position for test2: " << moon->get_x() << ", " << moon->get_y() << "\n";


}


int main(){
    angleTest();

    positionTest();
    
    return 0;
}

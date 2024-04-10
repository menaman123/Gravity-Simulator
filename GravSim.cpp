#include "GravSim.h"


#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>
#define _USE_MATH_DEFINES

/*
The acceleration of bodies is calculating using a single thread process. Although it is single threaded we can opitmize the code using dynamic programming. 
When going through all the bodies to calculate the acceleration between the objects we only have to calculate the acceleration once, the only thing that
changes is the acceleration. For example this empty array contains the objects [][][][][] after the first loop around the acceleration of each object is 
[a2+a3+a4][a1][a1][a1][a1]. The reason as to why the accelerations of the other objects are also filled with a1 is because we reduce the need to calculate
to calculate again. Thus each iteration to calculate the acceleration on the object is 1 less then the previous. On the first loop it will go n times,
second loop will be n-1 times, third loop will be n - 3 times till it reaches the end which will be done completely. 
*/


double SpaceObject::get_x(){
    return x;
}

double SpaceObject::get_y(){
    return y;
}

double SpaceObject::get_acceleration_x(){
    return acceleration_x;
}

double SpaceObject::get_acceleration_y(){
    return acceleration_y;
}

double SpaceObject::get_velocity_x(){
    return vx;
}

double SpaceObject::get_velocity_y(){
    return vy;
}

void SpaceObject::update_position(double& time){
    x = x + (0.5 * time * time * acceleration_x) + vx * time;
    y = y + (0.5 * time * time * acceleration_y) + vy * time;
}

void SpaceObject::update_velocity(double& time){
    vx = vx + acceleration_x * time;
    vy = vy + acceleration_y * time;
}

void SpaceObject::update_acceleration(SpaceObject& obj, double& directional_angle, double& distance){

    acceleration_x = acceleration_x + ((obj.mass * gravity_constant / pow(distance, 2))) * cos(directional_angle);
    acceleration_y = acceleration_y + ((obj.mass * gravity_constant / pow(distance, 2))) * sin(directional_angle);
}

void SpaceObject::reset_acceleration(){
    acceleration_x = 0.0;
    acceleration_y = 0.0;
}

SpaceObject::SpaceObject(double x_pos, double y_pos, double my_mass){
    mass = my_mass;
    x = x_pos;
    y = y_pos;
    acceleration_x = 0.0;
    acceleration_y = 0.0;
}


double calculateAngle(SpaceObject& obj1, SpaceObject& obj2){
    return atan2(abs(obj1.get_y() - obj2.get_y()), abs(obj1.get_x() - obj2.get_x()));
}

void calculateAcceleration(std::vector<SpaceObject*>& obj_vector, double& dt){
    for(int i = 0; i < obj_vector.size(); i++){
        SpaceObject* current = obj_vector[i];
        for(int j = i+1; j < obj_vector.size(); j++){
            SpaceObject* next = obj_vector[j];
            double angle = calculateAngle(*current, *next);

            double distance = sqrt(pow(abs(current->get_x() - next->get_x()), 2) + pow(abs(current->get_y() - next->get_y()),2));
            
            current->update_acceleration(*next, angle, distance);


            double next_angle = (M_PI) - angle;

            // Updating the acceleration so the calculation is only done once
            next->update_acceleration(*current, next_angle, distance);


            
        }

        current->update_velocity(dt);


        // Update the current position after summing all the forces affecting the object
        current->update_position(dt);

    }
}

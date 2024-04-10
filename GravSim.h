
#ifndef GRAVSIM_H
#define GRAVSIM_H


#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>




const double gravity_constant = 6.6743e-11;
class SpaceObject{

    public:

        double get_x();

        double get_y();

        double get_acceleration_x();

        double get_acceleration_y();

        double get_velocity_x();

        double get_velocity_y();

        void update_position(double& time);

        void update_velocity(double& time);

        void update_acceleration(SpaceObject& obj, double& directional_angle, double& distance);

        void reset_acceleration();

        SpaceObject(double x_pos, double y_pos, double my_mass);

    private:
        double mass;
        double x;
        double y;
        double vx;
        double vy;
        double acceleration_x;
        double acceleration_y;
};

double calculateAngle(SpaceObject& obj1, SpaceObject& obj2);

void calculateAcceleration(std::vector<SpaceObject*>& obj_vector, double& dt);




#endif
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <cmath>

bool save_animation = true;
bool follow_center_off_mass = true;
std::string file_path = "simulation2.txt";
double G = 6.67384e-11;
//mass of sun:1.9891e30, mass of earth:5.9e24
std::vector<double> mass = {1.9891e30, 1.9891e30,5.9e24, 1.9891e30, 5.9e24, 5.9e24,5.9e24,5.9e24,5.9e24,5.9e24,5.9e24}; 
//mass of sun:6.957e8, mass of earth:6.372e6
std::vector<double> radius = {6.957e8, 6.957e8,6.372e6,6.957e8,6.372e6,6.372e6,6.372e6,6.372e6,6.372e6,6.372e6,6.372e6};
//earth aphelion velocity 2.929e4
std::vector<std::vector<double>> velocity = {{0.0,-1.0e4,0},{0.0,1.0e4,0.0},{-2.0e4,2.0e4,0.0},{-2.0e4,-2.0e4,0.0},{2.0e4,2.0e4,0.0},{2.0e4,-2.0e4,0.0},{-2.0e4,2.0e4,0.0},{-2.0e4,2.0e4,0.0},{-2.0e4,2.0e4,0.0},{-2.0e4,2.0e4,0.0},{-2.0e4,2.0e4,0.0}};
//earth aphelion distance 1.5210e11
std::vector<std::vector<double>> position = {{-1.5210e11,0.0,0.0},{1.5210e11,0.0,0.0},{2.5e11,2.5e11,0.0},{2.17e11,-2.17e11,0.0},{-1.8510e11,1.8510e11,0.0},{-1.5210e11,-1.5210e11,0.0},{2.6e11,2.5e11,0.0},{2.5e11,2.6e11,0.0},{2.6e11,2.6e11,0.0},{2.6e11,2.7e11,0.0},{2.7e11,2.6e11,0.0}};
double t = 0.0;
double scale = 1.0e9; //meter/pixel
std::vector<double> upscale = {1.0e1,1.0e1,4.0e2,1.0e1,4.0e2,4.0e2,4.0e2,4.0e2,4.0e2,4.0e2,4.0e2}; //shown radius/real radius
double dt = 1.0; //time interval
std::vector<std::vector<std::vector<double>>> past_positions = {position};
double simulation_duration = 1.0e8;
double time_multiplier = 5.0e6;
double simulation_interval = std::max(1,static_cast<int>(time_multiplier/(60.0*dt)));
int number_of_objects = mass.size();

double distance(std::vector<double>& coordinate1, std::vector<double>& coordinate2){
    return sqrt(std::pow((coordinate1[0]-coordinate2[0]),2)+std::pow((coordinate1[1]-coordinate2[1]),2)+std::pow((coordinate1[2]-coordinate2[2]),2));
}

std::vector<double> gravity(int index){
    std::vector<double> acceleration = {0.0,0.0,0.0};
    for(int i = 0; i < number_of_objects; i++){
        if(i != index){
            for(int axel = 0; axel < 3; axel++){
                //newton's law of gravity
                acceleration[axel] += -G*mass[i]/(std::pow(std::max(distance(position[index],position[i]), (radius[index]+radius[i])),3))
                    *(position[index][axel]-position[i][axel]);
            }
        }
    }
    return acceleration;
}


void approximate(double delta_t){
    for(int i = 0; i < number_of_objects; i++){
        std::vector<double> acceleration = gravity(i);
        for(int axel = 0; axel < 3; axel++){
            velocity[i][axel] += delta_t*acceleration[axel];
            position[i][axel] += delta_t*velocity[i][axel];
        }
    }
    t += delta_t;
}


double total_energy(){
    double energy = 0.0;
    for(int i = 0; i < number_of_objects; i++){
        energy += 1/2.0*mass[i]*(std::pow(velocity[i][0],2)+std::pow(velocity[i][1],2)+std::pow(velocity[i][2],2));
        for(int j = 0; j < number_of_objects-i-1; j++){
            energy += -G*mass[i]*mass[i+j+1]/(distance(position[i],position[i+j+1]));
        }
    }
    return energy;
}

void simulate(double duration, double delta_t){
    auto start_time = std::chrono::high_resolution_clock::now();
    while(t < duration){
        for(int i = 0; i < simulation_interval; i++){
            approximate(delta_t);
        }
        std::cout << t*100.0/duration << "%\r";
        std::cout.flush();
        past_positions.push_back(position);
    }
    std::cout << "\nDone\n";
    auto end_time = std::chrono::high_resolution_clock::now();
    auto simulation_duration = std::chrono::duration_cast
            <std::chrono::seconds>(end_time - start_time);
    std::cout << simulation_duration.count() << "s\n";
}

int main(){
    std::cout << "time multiplier: " << 60*dt*simulation_interval << '\n';
    if(follow_center_off_mass){
        std::vector<double> momentum = {0,0,0};
        double total_mass = 0.0;
        for(int i = 0; i < number_of_objects; i++){
            total_mass += mass[i];
            for(int axel = 0; axel < 3; axel++){
                momentum[axel] += mass[i]*velocity[i][axel];
            }
        }
        std::vector<double> velocity_center_of_mass = {0.0,0.0,0.0};
        for(int axel = 0; axel < 3; axel++){
            velocity_center_of_mass[axel] = momentum[axel]/total_mass;
        }
        for(int i = 0; i < number_of_objects; i++){
            for(int axel = 0; axel < 3; axel++){
                velocity[i][axel] -= velocity_center_of_mass[axel];
            }
        }
    }
    double start_energy = total_energy();
    simulate(simulation_duration,dt);
    std::cout << "delta_energy = " << 100.0*(total_energy()-start_energy)/start_energy << "%\n";
    std::string simulation_text = "[";
    for(int i = 0; i < past_positions.size(); i++){
        simulation_text += '[';
        for(int j = 0; j < past_positions[i].size(); j++){
            simulation_text += '[';
            for(int k = 0; k < past_positions[i][j].size(); k++){
                simulation_text += std::to_string(past_positions[i][j][k]);
                if(k < past_positions[i][j].size() - 1){
                    simulation_text += ',';
                }
            }
            simulation_text += ']';
            if(j < past_positions[i].size() - 1){
                simulation_text += ',';
            }
        }
        simulation_text += ']';
        if(i < past_positions.size() - 1){
            simulation_text += ',';
        }
    }
    simulation_text += ']';
    if(save_animation){
        std::ofstream outputFile(file_path);
        outputFile << simulation_text;
        outputFile.close();
    }
    return 0;
}